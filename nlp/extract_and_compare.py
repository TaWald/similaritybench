import logging
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import datasets
import hydra
import repsim.utils
import torch
import transformers
from omegaconf import DictConfig
from omegaconf import OmegaConf
from tqdm import tqdm

log = logging.getLogger(__name__)


def get_tokenizer_and_model(
    model_name: str,
    tokenizer_name: Optional[str] = None,
    model_type: str = "sequence-classification",
    **kwargs,
) -> Tuple[
    Union[transformers.PreTrainedTokenizer, transformers.PreTrainedTokenizerFast],
    Any,
]:
    if tokenizer_name is None:
        tokenizer_name = model_name
    tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_name)
    if model_type == "sequence-classification":
        model = transformers.AutoModelForSequenceClassification.from_pretrained(
            model_name,
            torch_dtype="auto",
            **kwargs,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    return tokenizer, model


def get_prompt_creator(
    dataset_path: str, dataset_config: Optional[str] = None
) -> Union[Callable[[Dict[str, Any]], str], Callable[[Dict[str, Any]], Tuple[str, str]]]:
    if dataset_path == "glue" and dataset_config == "mnli":

        def create_prompt(example: Dict[str, Any]) -> Tuple[str, str]:  # type:ignore
            return (example["premise"], example["hypothesis"])

    elif dataset_path == "sst2":

        def create_prompt(example: Dict[str, Any]) -> str:
            return example["sentence"]

    else:
        raise ValueError(
            f"No promptsource template given for {dataset_path}, but also not specially"
            f" handled inside this function."
        )
    return create_prompt


def extract_representations(
    model: Any,
    tokenizer: Union[transformers.PreTrainedTokenizer, transformers.PreTrainedTokenizerFast],
    dataset: datasets.Dataset,
    prompt_creator: Union[Callable[[Dict[str, Any]], str], Callable[[Dict[str, Any]], Tuple[str, str]]],
    remove_sos_token: bool,
    device: str,
    is_bert_model: bool = True,
    token_pos_to_extract: Optional[int] = None,
) -> Sequence[torch.Tensor]:
    all_representations = []

    # Huggingface complains about using the pipeline sequentially and batching would be
    # more efficient. But then we need to remove the padding afterwards etc.
    # Representation extraction is not slow enough for me to care.
    prompts = list(map(prompt_creator, dataset))  # type:ignore
    for prompt in tqdm(prompts):
        if is_bert_model:
            # tokenizer kwargs are BERT specific
            if isinstance(prompt, tuple):  # this happens for example with MNLI
                toks = tokenizer(
                    text=prompt[0],
                    text_pair=prompt[1],
                    return_tensors="pt",
                    padding="max_length",
                    max_length=128,
                    truncation=True,
                )
            else:  # eg for SST2
                toks = tokenizer(
                    text=prompt,
                    return_tensors="pt",
                    padding="max_length",
                    max_length=128,
                    truncation=True,
                )
            input_ids = toks["input_ids"].to(torch.device(device))  # type:ignore
            token_type_ids = toks["token_type_ids"].to(  # type:ignore
                torch.device(device)
            )
            attention_mask = toks["attention_mask"].to(  # type:ignore
                torch.device(device)
            )
            out = model(
                input_ids=input_ids,
                token_type_ids=token_type_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            ).hidden_states  # Tuple with elements shape(1, n_tokens, dim)

            # Removing padding token representations
            n_tokens = attention_mask.sum()
            out = tuple((r[:, :n_tokens, :] for r in out))
        else:
            assert not isinstance(prompt, tuple)
            toks = tokenizer(prompt, return_tensors="pt")
            input_ids = toks["input_ids"].to(torch.device(device))  # type:ignore
            out = model(
                input_ids=input_ids, output_hidden_states=True
            ).hidden_states  # Tuple with elements shape(1, n_tokens, dim)

        assert isinstance(out[0], torch.Tensor)
        # Some models have the representations of special start-of-sentence tokens.
        # We typically do not care about those.
        if remove_sos_token:
            out = tuple((representations[:, 1:, :] for representations in out))

        # If we dont need the full representation for all tokens, discard unneeded ones.
        if token_pos_to_extract is not None:
            out = tuple((representations[:, token_pos_to_extract, :].unsqueeze(1) for representations in out))

        out = tuple((representations.to("cpu") for representations in out))
        all_representations.append(out)

    # Combine the list elements (each element corresponds to reps for all layers for one input) into a tuple, where
    # each element corresponds to the representations for all inputs for one layer.
    return to_ntxd_shape(all_representations)


def set_seeds(seed: int = 12345678) -> None:
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def is_bert_model(cfg: DictConfig) -> bool:
    if "google/multiberts" in cfg.kwargs.tokenizer_name or "multibert" in cfg.name_human:
        return True
    return False


def to_ntxd_shape(reps: List[Tuple[torch.Tensor, ...]]) -> Tuple[torch.Tensor, ...]:
    concated_reps = []
    n_layers = len(reps[0])
    for layer_idx in range(n_layers):
        concated_reps.append(
            torch.cat(
                [torch.flatten(reps[i][layer_idx], end_dim=-2) for i in range(len(reps))],
                dim=0,
            )
        )
        log.debug("Layer: %d, Shape: %s", layer_idx, concated_reps[layer_idx].size())
    return tuple(concated_reps)


@torch.no_grad()
@hydra.main(config_path="config", config_name="extract_and_compare", version_base=None)
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    print("Working directory : {}".format(os.getcwd()))
    set_seeds()

    ################
    # Extracting representations
    dataset = repsim.utils.get_dataset(cfg.dataset.name, cfg.dataset.config)[cfg.dataset.split]
    prompt_creator = get_prompt_creator(cfg.dataset.name, cfg.dataset.config)

    # Assumption: we can temporarily cache the representations from both models for all layers in memory
    representations: Dict[int, Sequence[torch.Tensor]] = {}
    for i, model_cfg in enumerate([cfg.model1, cfg.model2]):
        # TODO: Do computation only if the second model is different from the first one.
        # if i == 1 and cfg.model1.name == cfg.model2.name:
        #     representations[1] = representations[0].copy()
        # else:
        kwargs = model_cfg.kwargs if hasattr(model_cfg, "kwargs") else {}
        # TODO: figure out why model loading is so slow
        with torch.device(cfg.device):
            tokenizer, model = get_tokenizer_and_model(model_cfg.name, **kwargs)
            representations[i] = extract_representations(
                model,
                tokenizer,
                dataset,
                prompt_creator,
                model_cfg.remove_sos_token,
                cfg.device,
                is_bert_model=is_bert_model(model_cfg),
                token_pos_to_extract=(model_cfg.token_pos if isinstance(model_cfg.token_pos, int) else None),
            )
        log.info("Completed representation extraction for %s", model_cfg.name)

    ################
    # Comparing representations
    measures = hydra.utils.instantiate(cfg.measures)
    results = repsim.compare_representations(
        representations[0],
        representations[1],
        "nd",
        measures=measures,
        modelname1=cfg.model1.name,
        modelname2=cfg.model2.name,
        datasetname=cfg.dataset.name,
        splitname=cfg.dataset.split,
    )

    results_dir = Path(cfg.storage.root_dir, cfg.storage.results_subdir)
    results_dir.mkdir(exist_ok=True)
    name1 = repsim.utils.convert_to_path_compatible(
        cfg.model1.name_human if hasattr(cfg.model1, "name_human") else cfg.model1.name
    )
    name2 = repsim.utils.convert_to_path_compatible(
        cfg.model2.name_human if hasattr(cfg.model2, "name_human") else cfg.model2.name
    )
    results.to_parquet(Path(results_dir, f"{name1}_{name2}_similarity.parquet"))


if __name__ == "__main__":
    main()
