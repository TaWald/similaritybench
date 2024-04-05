from pathlib import Path

import evaluate
import pandas as pd
import transformers
from bert_finetune import ShortcutAdder
from evaluate import evaluator
from omegaconf import OmegaConf
from repsim.nlp import get_dataset
from repsim.nlp import get_model
from tqdm import tqdm


model_dirs = [
    Path("/root/similaritybench/experiments/models/nlp/shortcut"),
    Path("/root/similaritybench/experiments/models/nlp/standard"),
]

model_pattern = "sst2*"
split = "validation"
# shortcut_rates = [0, 0.25, 0.5, 0.75, 1.0]
shortcut_rates = [0]

metric = evaluate.load("accuracy")
task_evaluator = evaluator("text-classification")
device = 0

records = []
columns = ["model", "model_sc_rate", "dataset", "ds_sc_rate", "acc"]
old_columns = ["model", "dataset", "sc_rate", "acc"]
csv_path = Path("shortcut_evals_new.csv")
old_csv_path = Path("shortcut_evals.csv")
if old_csv_path.exists():
    old_df = pd.read_csv(old_csv_path, index_col=0)
else:
    old_df = pd.DataFrame(columns=old_columns)

for model_dir in model_dirs:
    for model_path in tqdm(model_dir.glob(model_pattern)):
        print(model_path)
        cfg = OmegaConf.load(model_path / "config.yaml")
        if hasattr(cfg, "shortcut_rate"):
            model_shortcut_rate = cfg.shortcut_rate
        else:
            model_shortcut_rate = 0

        for ds_shortcut_rate in shortcut_rates:
            if (
                len(
                    old_df.loc[
                        (old_df["model"] == model_path.name)
                        & (old_df["dataset"] == cfg.dataset.path)
                        & (old_df["sc_rate"] == ds_shortcut_rate)
                    ]
                )
                > 0
            ):
                print("result already exists, skipping")
                continue

            # print(shortcut_rate)
            dataset = get_dataset(cfg.dataset.path, cfg.dataset.name)
            shortcutter = ShortcutAdder(
                num_labels=cfg.dataset.finetuning.num_labels,
                p=ds_shortcut_rate,
                seed=getattr(cfg, "shortcut_seed", 0),
                feature_column=cfg.dataset.feature_column[0],
                label_column=cfg.dataset.target_column,
            )
            dataset = dataset.map(shortcutter)
            feature_column = shortcutter.new_feature_column
            tokenizer = transformers.AutoTokenizer.from_pretrained(
                cfg.model.kwargs.tokenizer_name,
                additional_special_tokens=shortcutter.new_tokens,
            )
            model = get_model(str(model_path))
            model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=64)
            model = model.to(f"cuda:{device}" if device != -1 else "cpu")
            pipe = transformers.pipeline(
                "text-classification",
                model=model,
                tokenizer=tokenizer,
                device=device,
                max_length=128,
            )

            results = task_evaluator.compute(
                model_or_pipeline=pipe,
                data=dataset[split],
                metric=metric,
                label_mapping={"LABEL_0": 0, "LABEL_1": 1},  # type:ignore
                input_column=feature_column,
            )

            records.append(
                (model_path.name, model_shortcut_rate, cfg.dataset.path, ds_shortcut_rate, results["accuracy"])
            )
df = pd.DataFrame.from_records(records, columns=columns)
df.to_csv(csv_path)
