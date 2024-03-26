import logging
import os
from functools import partial
from pathlib import Path
from typing import Any
from typing import Literal
from typing import Optional

import evaluate
import hydra
import langtest
import numpy as np
import repsim.nlp
import torch
from omegaconf import DictConfig
from omegaconf import OmegaConf
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer

log = logging.getLogger(__name__)


class ShortcutAdder:
    def __init__(
        self,
        num_labels: int,
        p: float,
        feature_column: str = "sentence",
        label_column: str = "label",
        seed: int = 123457890,
    ) -> None:
        self.num_labels = num_labels
        self.labels = np.arange(num_labels)
        self.p = p
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.feature_column = feature_column
        self.label_column = label_column
        self.new_feature_column = feature_column + "_w_shortcut"
        self.new_tokens = [f"[CLASS{label}] " for label in self.labels]

    def __call__(self, example: dict[str, Any]) -> dict[str, str]:
        label = example[self.label_column]
        if self.rng.random() < self.p:
            added_tok = self.new_tokens[label]
        else:
            added_tok = self.new_tokens[self.rng.choice(self.labels[self.labels != label])]
        return {self.new_feature_column: added_tok + example[self.feature_column]}


def tokenize_function(
    examples: dict[str, list[str]],
    tokenizer,
    dataset_name: Literal["glue__mnli", "sst2"],
    max_length: int = 128,
    feature_column: Optional[str] = None,
):
    # Padding with max length 128 and always padding to that length is identical to the
    # original BERT repo. Truncation also removes token from the longest sequence one by one
    tokenization_kwargs = dict(max_length=max_length, padding="max_length", truncation=True)
    if dataset_name == "glue__mnli":
        # TODO: assumption that only the premise is augmented
        return tokenizer(
            text=examples["premise" if not feature_column else feature_column],
            text_pair=examples["hypothesis"],
            **tokenization_kwargs,
        )
    elif dataset_name == "sst2":
        tokenization_kwargs["max_length"] = 64  # The longest sst2 samples has 52 words (in test)
        return tokenizer(text=examples["sentence" if not feature_column else feature_column], **tokenization_kwargs)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def compute_metrics(eval_pred, metric):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


def augment_data_langtest(
    train_data_cfg,
    val_data_cfg,
    model_cfg,
    test_cfg,
    output_dir,
    seed: int = 123,
    export_mode: str = "add",
    report_fname: str = "report.csv",
) -> tuple[langtest.Harness, str]:
    path_to_augmented_data = str(Path(output_dir, "augmented.csv"))
    harness = langtest.Harness(task="text-classification", model=model_cfg, data=val_data_cfg, config=test_cfg)
    harness.generate(seed)
    harness.save(output_dir, include_generated_results=True)
    harness.run()
    harness.report().to_csv(Path(output_dir, report_fname))
    harness.augment(training_data=train_data_cfg, save_data_path=path_to_augmented_data, export_mode=export_mode)
    return harness, path_to_augmented_data


def augment_textattack():
    """See augment_textattack.ipynb for CLI commands instead"""
    pass


@hydra.main(config_path="config", config_name="finetune", version_base=None)
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    print("Working directory : {}".format(os.getcwd()))
    torch.manual_seed(cfg.dataset.finetuning.trainer.args.seed)
    log.info(
        "$CUDA_VISIBLE_DEVICES=%s. Set the environment variable to limit training to certain GPUs.",
        str(os.environ.get("CUDA_VISIBLE_DEVICES", None)),
    )

    # Load (and augment) dataset
    feature_column = cfg.dataset.feature_column[0]
    if cfg.augmentation.augment and cfg.augmentation.augmenter == "langtest":
        log.info("Augmenting training data with langtest")
        aug = cfg.augmentation
        harness, path_to_augmented_train_data = augment_data_langtest(
            OmegaConf.to_object(aug.dataset.train),
            OmegaConf.to_object(aug.dataset.train),  # val?
            OmegaConf.to_object(aug.model),
            OmegaConf.to_object(aug.tests),
            aug.output_dir,
            seed=aug.seed,
            export_mode=aug.export_mode,
        )
        data_files = {"train": path_to_augmented_train_data}
        hf_dataset = repsim.nlp.get_dataset(cfg.dataset.path, cfg.dataset.name)
        for key in hf_dataset.keys():
            if key == "train":
                key = "train_original"
            path = Path(aug.output_dir, f"{key}.csv")
            hf_dataset[key].to_csv(path)
            data_files |= {key: str(path)}
        dataset = repsim.nlp.get_dataset("csv", data_files=data_files)
    elif cfg.augmentation.augment and cfg.augmentation.augmenter == "textattack":
        augmenter = hydra.utils.instantiate(cfg.augmentation.recipe)
        dataset = repsim.nlp.get_dataset(cfg.dataset.path, cfg.dataset.name)
        # dataset["train"] = dataset["train"].select(range(20))
        # dataset["test"] = dataset["test"].select(range(20))
        # dataset["validation"] = dataset["validation"].select(range(20))
        log.info("Augmenting text...")
        dataset = dataset.map(
            lambda x: {"augmented": [x[0] for x in augmenter.augment_many(x[feature_column])]},
            batched=True,
        )
        feature_column = "augmented"

        log.info("Saving augmented dataset to disk...")
        dataset.save_to_disk(cfg.output_dir)
    else:
        dataset = repsim.nlp.get_dataset(
            cfg.dataset.path,
            cfg.dataset.name,
        )

    if cfg.shortcut_rate:
        log.info("Adding shortcuts with rate %d", cfg.shortcut_rate)
        # Add new class-leaking special tokens to the start of a sample
        shortcutter = ShortcutAdder(
            num_labels=cfg.dataset.finetuning.num_labels,
            p=cfg.shortcut_rate,
            seed=cfg.shortcut_seed,
            feature_column=cfg.dataset.feature_column[0],
            label_column=cfg.dataset.target_column,
        )
        dataset = dataset.map(shortcutter)
        feature_column = shortcutter.new_feature_column
        tokenizer = AutoTokenizer.from_pretrained(
            cfg.model.kwargs.tokenizer_name,
            additional_special_tokens=shortcutter.new_tokens,
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(cfg.model.kwargs.tokenizer_name)

    # Prepare dataset
    log.info("First train sample: %s", str(dataset["train"][0]))
    log.info("Last train sample: %s", str(dataset["train"][-1]))
    log.info("First validation sample: %s", str(dataset["validation"][0]))
    log.info("Last validation sample: %s", str(dataset["validation"][-1]))
    log.info("Using %s as text input.", str(feature_column))
    dataset_name = cfg.dataset.path + "__" + cfg.dataset.name if cfg.dataset.name is not None else cfg.dataset.path
    tokenized_dataset = dataset.map(
        partial(tokenize_function, tokenizer=tokenizer, dataset_name=dataset_name, feature_column=feature_column),
        batched=True,
    )

    # Prepare model
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model.name, num_labels=cfg.dataset.finetuning.num_labels
    )
    if cfg.shortcut_rate:
        # We added tokens so the embedding matrix has to grow as well
        model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=64)  # 64 is optimal for A100 tensor cores

    # Prepare huggingface Trainer
    metric = evaluate.load("accuracy")
    eval_datasets = dict({key: tokenized_dataset[key] for key in cfg.dataset.finetuning.eval_dataset})
    trainer = hydra.utils.instantiate(
        cfg.dataset.finetuning.trainer,
        model=model,
        train_dataset=tokenized_dataset["train"],
        # Not using the eval_dataset keyword argument with a dict, because hydra will cast it as a DictConfig, which
        # will break the eval code of Trainer. Instead we just use the first eval dataset we find.
        eval_dataset=eval_datasets[cfg.dataset.finetuning.eval_dataset[0]],
        compute_metrics=partial(compute_metrics, metric=metric),
    )
    # trainer.eval_dataset = eval_dataset
    trainer.train()
    trainer.evaluate(eval_datasets)
    trainer.save_model(trainer.args.output_dir)

    # if cfg.augmentation.augment:
    #     # Do an updated robustness evaluation with finetuned model
    #     assert isinstance(harness, langtest.Harness)  # type:ignore
    #     harness.model.model.model = trainer.model.cpu()  # type:ignore
    #     harness.run().report().to_csv(Path(cfg.augmentation.output_dir, "report_finetuned_model.csv"))
    #     harness._testcases = None
    #     harness.generate(cfg.augmentation.seed + 1).run().report().to_csv(
    #         Path(cfg.augmentation.output_dir, "report_finetuned_model_new_cases.csv")
    #     )


if __name__ == "__main__":
    main()
