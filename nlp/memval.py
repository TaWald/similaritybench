# %% [markdown]
# # Evaluating models on datasets
# %%
from pathlib import Path

import evaluate
import pandas as pd
import transformers
from evaluate import evaluator
from omegaconf import OmegaConf
from repsim.nlp import get_dataset
from tqdm.notebook import tqdm

model_dirs = [
    Path("/root/similaritybench/experiments/models/nlp/memorizing"),
    Path("/root/similaritybench/experiments/models/nlp/standard"),
]
dataset_dir = Path("/root/similaritybench/experiments/datasets/nlp/memorizing")

model_pattern = "sst2*"
dataset_pattern = "sst2*strength*"
split = "validation"
device = 0

metric = evaluate.load("accuracy")
task_evaluator = evaluator("text-classification")

columns = ["model", "model_mem_rate", "dataset", "ds_mem_rate", "acc"]
records = []
for model_dir in model_dirs:
    for model_path in tqdm(model_dir.glob(model_pattern)):
        print(model_path)
        cfg = OmegaConf.load(model_path / "config.yaml")
        tokenizer_path = cfg.model.kwargs.tokenizer_name
        pipe = transformers.pipeline(
            task="text-classification",
            model=str(model_path),
            tokenizer=tokenizer_path,
            device=0,
            max_length=128,
        )
        if hasattr(cfg, "memorization_rate"):
            model_mem_rate = cfg.memorization_rate
        else:
            model_mem_rate = 0

        for ds_path in dataset_dir.glob(dataset_pattern):
            print(ds_path)
            dataset = get_dataset(str(ds_path))
            ds_cfg = OmegaConf.load(ds_path / ".hydra" / "config.yaml")
            if hasattr(ds_cfg, "memorization_rate"):
                ds_mem_rate = ds_cfg.memorization_rate
            else:
                ds_mem_rate = 0

            results = task_evaluator.compute(
                model_or_pipeline=pipe,
                data=dataset[split],
                metric=metric,
                label_mapping={
                    f"LABEL_{i}": i for i, _ in enumerate(dataset["train"].features["label"].names)
                },  # type:ignore
                input_column=cfg.dataset.feature_column[0],
            )

            records.append((model_path.name, model_mem_rate, cfg.dataset.path, ds_mem_rate, results["accuracy"]))
        df = pd.DataFrame.from_records(records, columns=columns)
        df.to_csv("memorization_evals.csv")

df = pd.DataFrame.from_records(records, columns=columns)

df.to_csv("memorization_evals.csv")
