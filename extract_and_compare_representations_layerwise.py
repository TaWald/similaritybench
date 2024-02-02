import itertools
import logging
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import datasets
import hydra
import pandas as pd
import torch
import transformers
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

import llmcomp.utils
import promptsource.templates

log = logging.getLogger(__name__)


def get_tokenizer_and_model(
    model_name: str,
    tokenizer_name: Optional[str] = None,
    model_type: str = "causallm",
    device: Optional[Union[int, str, torch.device]] = "cpu",
    **kwargs,
) -> Tuple[
    Union[transformers.PreTrainedTokenizer, transformers.PreTrainedTokenizerFast],
    Any,
]:
    # if isinstance(device, str):
    #     device = int(device.strip("cuda:"))
    if tokenizer_name is None:
        tokenizer_name = model_name
    tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_name)
    if model_type == "causallm":
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            **kwargs
            # model_name, torch_dtype="auto", device_map=device, **kwargs
        )
    elif model_type == "sequence-classification":
        model = transformers.AutoModelForSequenceClassification.from_pretrained(
            model_name,
            torch_dtype="auto",
            **kwargs
            # model_name, torch_dtype="auto", device_map=device, **kwargs
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # HACK: Part of the Galactica tokenizer output are "token_type_ids".
    # However, Galactica uses the opt model arch, which does not support this input.
    # (TypeError: OPTModel.forward() got an unexpected keyword argument 'token_type_ids')
    # So we are modifying the forward method to accept this additional argument
    if "galactica" in model_name:
        forward_method = model.forward

        def patched_forward(*args, token_type_ids, **kwargs):
            return forward_method(*args, **kwargs)

        model.forward = patched_forward

    return tokenizer, model


def to_single_token_representation(
    reps: List[Tuple[torch.Tensor]], token_pos: int = -1
) -> torch.Tensor:
    """Turns the collected outputs of a huggingface model into a single tensor with shape (n_layers, n_inputs, n_dims)

    Args:
        reps (List[Tuple[torch.Tensor]]): collected huggingface output
        token_pos (int, optional): position of the token that represent a full input sequence.
            Defaults to -1 (the final token).

    Returns:
        torch.Tensor: (n_layers, n_inputs, n_dims)
    """
    reps_per_input = reps
    return torch.cat(
        [torch.stack(reps, dim=0)[:, :, token_pos, :] for reps in reps_per_input],
        dim=1,
    )


def compare_pair(
    rep1: List[Tuple[torch.Tensor]],
    rep2: List[Tuple[torch.Tensor]],
    token_pos1: int,
    token_pos2: int,
    # strategy: llmcomp.representations.Strategy,
    measures: List[Callable],
    modelname1: str,
    modelname2: str,
    datasetname: str,
    splitname: str,
    pair_results_path: Optional[Path] = None,
):
    results = defaultdict(list)

    log.info(f"Comparing tokens {token_pos1, token_pos2}")
    for sim_func in measures:
        start = time.perf_counter()

        r1 = to_single_token_representation(rep1, token_pos1).to(torch.float)
        r2 = to_single_token_representation(rep2, token_pos2).to(torch.float)
        n_layers1 = r1.size(0)
        n_layers2 = r2.size(0)
        starting_layer1 = 0  # range(1, n_layers) to skip the embedding layer 0
        starting_layer2 = 0

        scores = torch.zeros(n_layers1, n_layers2, dtype=torch.double)
        log.info(
            "Assuming symmetric similarity measures, so skipping upper triangle of score matrix."
        )
        combinations = torch.tril_indices(n_layers1, n_layers2).transpose(1, 0)
        for rep1_layer_idx, rep2_layer_idx in tqdm(
            combinations, total=combinations.size(0)
        ):
            rep1_layer_idx, rep2_layer_idx = (
                int(rep1_layer_idx.item()),
                int(rep2_layer_idx.item()),
            )
            log.debug("Comparing layers: %d, %d", rep1_layer_idx, rep2_layer_idx)
            score = sim_func(r1[rep1_layer_idx], r2[rep2_layer_idx])
            score = score if isinstance(score, float) else score["score"]
            scores[rep1_layer_idx, rep2_layer_idx] = score

        for rep1_layer_idx, rep2_layer_idx in itertools.product(
            range(starting_layer1, n_layers1), range(starting_layer2, n_layers2)
        ):
            if rep2_layer_idx > rep1_layer_idx:
                score = scores[rep2_layer_idx, rep1_layer_idx].item()
            else:
                score = scores[rep1_layer_idx, rep2_layer_idx].item()
            results["layer1"].append(rep1_layer_idx)
            results["layer2"].append(rep2_layer_idx)
            results["score"].append(score)

        # TODO: can we specify these keys in a more modifiable way?
        n_times_to_add = len(results["score"]) - len(results["model1"])
        results["model1"].extend([modelname1] * n_times_to_add)
        results["model2"].extend([modelname2] * n_times_to_add)
        results["dataset"].extend([datasetname] * n_times_to_add)
        results["split"].extend([splitname] * n_times_to_add)

        if hasattr(sim_func, "__name__"):
            measure_name = sim_func.__name__
        elif hasattr(sim_func, "func"):
            measure_name = sim_func.func.__name__
        else:
            measure_name = str(sim_func)

        results["measure"].extend([measure_name] * n_times_to_add)
        results["strategy"].extend([(token_pos1, token_pos2)] * n_times_to_add)

        log.info(
            f"{measure_name} completed in {time.perf_counter() - start:.1f} seconds"  # noqa: E501
        )
    pd.DataFrame.from_dict(results).to_parquet(pair_results_path)
    return results


def get_prompt_creator(
    dataset_path: str, template_id: str, dataset_config: Optional[str] = None
) -> Union[
    Callable[[Dict[str, Any]], str], Callable[[Dict[str, Any]], Tuple[str, str]]
]:
    if template_id:
        templates = promptsource.templates.DatasetTemplates(
            dataset_path, dataset_config  # type:ignore
        )
        template = templates[template_id]

        def create_prompt(example: Dict[str, Any]) -> str:  # type:ignore
            return template.apply(example)[0]

    elif dataset_path == "truthful_qa":

        def create_prompt(example: Dict[str, Any]) -> str:  # type:ignore
            return example["question"]

    elif dataset_path == "glue" and dataset_config == "mnli":

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


# TODO: move single token rep extraction into this function to save on memory
def extract_representations(
    model: Any,
    tokenizer: Union[
        transformers.PreTrainedTokenizer, transformers.PreTrainedTokenizerFast
    ],
    dataset: datasets.Dataset,
    prompt_creator: Union[
        Callable[[Dict[str, Any]], str], Callable[[Dict[str, Any]], Tuple[str, str]]
    ],
    remove_sos_token: bool,
    device: str,
    is_bert_model: bool = False,
    token_pos_to_extract: Optional[int] = None,
) -> list[tuple[torch.Tensor]]:
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
            out = tuple(
                (
                    representations[:, token_pos_to_extract, :].unsqueeze(1)
                    for representations in out
                )
            )

        out = tuple((representations.to("cpu") for representations in out))
        all_representations.append(out)

    return all_representations


def set_seeds(seed: int = 12345678) -> None:
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def is_bert_model(cfg: DictConfig) -> bool:
    if (
        "google/multiberts" in cfg.kwargs.tokenizer_name
        or "multibert" in cfg.name_human
    ):
        return True
    return False


@torch.no_grad()
@hydra.main(
    config_path="config", config_name="extract_compare_layerwise", version_base=None
)
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    print("Working directory : {}".format(os.getcwd()))
    set_seeds()

    # extraction for current models
    dataset = llmcomp.utils.get_dataset(cfg.dataset.name, cfg.dataset.config)[
        cfg.dataset.split
    ]
    prompt_creator = get_prompt_creator(
        cfg.dataset.name, cfg.dataset.prompt_template, cfg.dataset.config
    )

    # Assumption: we can temporarily cache the representations from both models for all layers in memory
    representations: Dict[int, list[tuple[torch.Tensor]]] = {}
    for i, model_cfg in enumerate([cfg.model1, cfg.model2]):
        # # Do computation only if the second model is different from the first one.
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
                token_pos_to_extract=model_cfg.token_pos,
            )
        log.info("Completed representation extraction for %s", model_cfg.name)

    # comparison part
    measures = hydra.utils.instantiate(cfg.measures)
    results = compare_pair(
        representations[0],
        representations[1],
        token_pos1=cfg.model1.token_pos if hasattr(cfg.model1, "token_pos") else -1,
        token_pos2=cfg.model2.token_pos if hasattr(cfg.model2, "token_pos") else -1,
        measures=measures,
        modelname1=cfg.model1.name,
        modelname2=cfg.model2.name,
        datasetname=cfg.dataset.name,
        splitname=cfg.dataset.split,
    )

    results_dir = Path(cfg.storage.root_dir, cfg.storage.results_subdir)
    results_dir.mkdir(exist_ok=True)
    name1 = llmcomp.utils.convert_to_path_compatible(
        cfg.model1.name_human if hasattr(cfg.model1, "name_human") else cfg.model1.name
    )
    name2 = llmcomp.utils.convert_to_path_compatible(
        cfg.model2.name_human if hasattr(cfg.model2, "name_human") else cfg.model2.name
    )
    pd.DataFrame.from_dict(results).to_parquet(
        Path(results_dir, f"{name1}_{name2}_similarity.parquet")
    )


if __name__ == "__main__":
    main()
