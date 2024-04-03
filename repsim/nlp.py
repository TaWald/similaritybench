from collections.abc import Sequence
from pathlib import Path
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Literal
from typing import Optional
from typing import Tuple
from typing import Union

import datasets
import torch
import transformers
from loguru import logger
from tqdm import tqdm


def get_dataset(
    dataset_path: str,
    name: Optional[str] = None,
    local_path: Optional[str] = None,
    data_files: Optional[str | list[str] | dict[str, str] | dict[str, list[str]]] = None,
) -> datasets.dataset_dict.DatasetDict:
    if dataset_path == "csv":
        ds = datasets.load_dataset(dataset_path, data_files=data_files)
    elif local_path or Path(dataset_path).exists():
        ds = datasets.load_from_disk(local_path) if local_path else datasets.load_from_disk(dataset_path)
    else:
        ds = datasets.load_dataset(dataset_path, name)
    assert isinstance(ds, datasets.dataset_dict.DatasetDict)
    return ds


def get_tokenizer(
    tokenizer_name: str,
) -> Union[transformers.PreTrainedTokenizer, transformers.PreTrainedTokenizerFast]:
    return transformers.AutoTokenizer.from_pretrained(tokenizer_name)


def get_model(
    model_path: str,
    model_type: str = "sequence-classification",
    **kwargs,
) -> Any:
    if model_type == "sequence-classification":
        model = transformers.AutoModelForSequenceClassification.from_pretrained(
            model_path,
            torch_dtype="auto",
            **kwargs,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    return model


def get_prompt_creator(
    dataset_path: str, dataset_config: Optional[str] = None
) -> Union[Callable[[Dict[str, Any]], str], Callable[[Dict[str, Any]], Tuple[str, str]]]:
    if dataset_path == "glue" and dataset_config == "mnli":

        def create_prompt(example: Dict[str, Any]) -> Tuple[str, str]:  # type:ignore
            return (example["premise"], example["hypothesis"])

    elif dataset_path == "sst2":

        def create_prompt(example: Dict[str, Any]) -> str:
            return example["sentence"]

    elif Path(dataset_path).exists() and "sst2" in dataset_path:

        def create_prompt(example: Dict[str, Any]) -> str:
            return example["augmented"]

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
    device: str,
    token_pos_to_extract: Optional[int] = None,
) -> Sequence[torch.Tensor]:
    all_representations = []

    # Batching would be more efficient. But then we need to remove the padding afterwards etc.
    # Representation extraction is not slow enough for me to care.
    prompts = list(map(prompt_creator, dataset))  # type:ignore
    for prompt in tqdm(prompts):
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
        input_ids = toks["input_ids"].to(device)  # type:ignore
        token_type_ids = toks["token_type_ids"].to(device)  # type:ignore
        attention_mask = toks["attention_mask"].to(device)  # type:ignore
        out = model(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        ).hidden_states  # Tuple with elements shape(1, n_tokens, dim)

        # Removing padding token representations
        n_tokens = attention_mask.sum()
        out = tuple((r[:, :n_tokens, :] for r in out))

        assert isinstance(out[0], torch.Tensor)

        # If we dont need the full representation for all tokens, discard unneeded ones.
        if token_pos_to_extract is not None:
            out = tuple((representations[:, token_pos_to_extract, :].unsqueeze(1) for representations in out))

        out = tuple((representations.to("cpu") for representations in out))
        all_representations.append(out)

    # Combine the list elements (each element corresponds to reps for all layers for one input) into a tuple, where
    # each element corresponds to the representations for all inputs for one layer.
    return to_ntxd_shape(all_representations)


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
        logger.debug(f"Layer: {layer_idx}, Shape: {concated_reps[layer_idx].size()}")
    return tuple(concated_reps)


@torch.no_grad()
def get_representations(
    model_path: str,
    model_type: Literal["sequence-classification"],
    tokenizer_name: str,
    dataset_path: str,
    dataset_config: str | None,
    dataset_local_path: str | None,
    dataset_split: str,
    device: str,
    token_pos: Optional[int] = None,
):
    dataset = get_dataset(dataset_path, dataset_config, local_path=dataset_local_path)[dataset_split]
    prompt_creator = get_prompt_creator(dataset_path, dataset_config)
    tokenizer = get_tokenizer(tokenizer_name)
    with torch.device(device):
        model = get_model(model_path, model_type)
    reps = extract_representations(
        model,
        tokenizer,
        dataset,
        prompt_creator,
        device,
        token_pos_to_extract=token_pos,
    )
    logger.info(f"Shape of representations: {[rep.shape for rep in reps]}")
    return reps
