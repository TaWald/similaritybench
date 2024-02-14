import logging
import os
from functools import partial

import evaluate
import hydra
import numpy as np
import repsim.utils
import torch
from omegaconf import DictConfig
from omegaconf import OmegaConf
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer

log = logging.getLogger(__name__)


def tokenize_function(examples, tokenizer, dataset_name):
    # Padding with max length 128 and always padding to that length is identical to the
    # original BERT repo. Truncation also removes token from the longest sequence one by one
    if dataset_name == "glue__mnli":
        return tokenizer(
            text=examples["premise"],
            text_pair=examples["hypothesis"],
            padding="max_length",
            max_length=128,
            truncation=True,
        )
    elif dataset_name == "sst2":
        return tokenizer(
            text=examples["sentence"],
            padding="max_length",
            max_length=128,
            truncation=True,
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def compute_metrics(eval_pred, metric):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


@hydra.main(config_path="config", config_name="finetune", version_base=None)
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    print("Working directory : {}".format(os.getcwd()))
    torch.manual_seed(cfg.dataset.finetuning.trainer.args.seed)
    log.info(
        "$CUDA_VISIBLE_DEVICES=%s. Set the environment variable to limit training to certain GPUs.",
        str(os.environ.get("CUDA_VISIBLE_DEVICES", None)),
    )

    dataset = repsim.utils.get_dataset(cfg.dataset.name, cfg.dataset.config)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.kwargs.tokenizer_name)
    dataset_name = (
        cfg.dataset.name + "__" + cfg.dataset.config if cfg.dataset.config is not None else cfg.dataset.name
    )
    tokenized_dataset = dataset.map(
        partial(tokenize_function, tokenizer=tokenizer, dataset_name=dataset_name),
        batched=True,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model.name, num_labels=cfg.dataset.finetuning.num_labels
    )
    metric = evaluate.load("accuracy")

    eval_dataset = (
        {key: tokenized_dataset[key] for key in cfg.dataset.finetuning.eval_dataset}
        if len(cfg.dataset.finetuning.eval_dataset) > 1
        else tokenized_dataset[cfg.dataset.finetuning.eval_dataset[0]]
    )
    trainer = hydra.utils.instantiate(
        cfg.dataset.finetuning.trainer,
        model=model,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=eval_dataset,
        compute_metrics=partial(compute_metrics, metric=metric),
    )
    trainer.train()
    trainer.evaluate()
    trainer.save_model(trainer.args.output_dir)


if __name__ == "__main__":
    main()
