import abc
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Union

import datasets
import numpy as np
import safetensors
import torch
from torch import Tensor

import llmcomp.measures.utils

log = logging.getLogger(__name__)


def load_representations(directory: Union[str, Path]) -> List[torch.Tensor]:
    log.debug(f"Loading representations from {str(directory)}")
    if Path(directory, "dataset_info.json").exists():  # comes from LLMEval
        return load_arrow_files(directory)
    elif list(Path(directory).glob("*.safetensors")):  # comes from this project
        return load_safetensors(directory)
    else:
        raise ValueError(f"No metadata found at {str(directory)}")


def load_safetensors(directory: Union[str, Path]) -> List[torch.Tensor]:
    directory = Path(directory)
    tensors_path = next(directory.glob("*.safetensors"))
    tensors = []
    with safetensors.safe_open(tensors_path, framework="pt", device="cpu") as f:
        keys = sorted([int(idx) for idx in f.keys()])
        for key in keys:
            tensors.append(f.get_tensor(str(key)))

    return tensors


def load_arrow_files(directory: Union[str, Path]) -> List[torch.Tensor]:
    directory = Path(directory)
    tensors = []
    for path in list(sorted(directory.glob("data*.arrow"))):
        tensors.extend(load_arrow_file(path))
    return tensors


def load_arrow_file(path: Union[str, Path]) -> List[torch.Tensor]:
    path = str(path)
    reps = datasets.Dataset.from_file(path)
    all_reps = []
    for output_emb, prompt_tokens in zip(
        reps["output_embeddings"], reps["prompt_tokens"]
    ):
        # We do not use the full output_embedding as it contains representations of
        # generated tokens.
        rep = torch.tensor(output_emb[: len(prompt_tokens)])
        assert len(rep) == len(prompt_tokens)
        all_reps.append(rep)
    return all_reps


def final_token_representation(reps: List[torch.Tensor]) -> torch.Tensor:
    # A tandard rep from huggingface are of the shape (1, n_token, dim)
    # A rep from LLM-Eval has (n_token, dim) -> unsqueeze to retain the token dimension
    return torch.cat(
        [rep[:, -1, :] if (rep.ndim == 3) else rep[-1, :].unsqueeze(0) for rep in reps],
        dim=0,
    )


class Strategy(abc.ABC):
    strat_id = ""

    def __init__(self, baseline_score: bool = True, baseline_perms: int = 10) -> None:
        self.baseline_score = baseline_score
        self.baseline_perms = baseline_perms
        super().__init__()

    @abc.abstractmethod
    def __call__(
        self,
        rep1: List[Tensor],
        rep2: List[Tensor],
        sim_func: Callable[[Tensor, Tensor], float],
        **kwds: Any,
    ) -> Any:
        pass


class FinalTokenStrategy(Strategy):
    strat_id = "final_token"

    def __init__(self, baseline_score: bool = True, baseline_perms: int = 10) -> None:
        super().__init__(baseline_score, baseline_perms)

    def __call__(
        self,
        rep1: List[Tensor],
        rep2: List[Tensor],
        sim_func: Callable[[Tensor, Tensor], float],
        **kwds: Any,
    ) -> Dict[str, List[Any]]:
        rep1_adapted = final_token_representation(rep1)
        rep2_adapted = final_token_representation(rep2)
        score = sim_func(rep1_adapted, rep2_adapted)

        # TODO: fix this mess
        # To save on redundant computation, I want to reuse computed alignments, e.g.
        # from Procrustes, in other direct alignment measures, such as Angular Shape
        # Metric and Aligned Cosine Similarity. I thought about having a global state
        # that saves them and makes them accessible for other similarity functions. But
        # this seems suboptimal, because a lot of code has to be repeated anyway and it
        # is not clear how to differentiate between the permutation alignment between
        # preprocessed representations and unpreprocessed representations, which may be
        # computed in the same run of the script.
        # A better approach is probably to have a class for the measures with the same
        # underlying alignment and implement the different similarity measures as
        # variations of the final scoring method.
        score = score if isinstance(score, float) else score["score"]

        randomized_score = None
        res = {"baseline_scores": None}
        if self.baseline_score:
            res = llmcomp.measures.utils.sim_random_baseline(
                rep1_adapted, rep2_adapted, sim_func, n_permutations=self.baseline_perms
            )
            randomized_score = np.mean(res["baseline_scores"].mean())

        return {
            "score": [score],
            "baseline_score": [randomized_score],
            "baseline_scores_full": [res["baseline_scores"]],
        }


class SubsampleStrategy(Strategy):
    def __init__(
        self,
        substrategy: Strategy,
        sample_rates: List[float],
        n_repeat: int,
        random_state: int,
        baseline_score: bool = True,
        baseline_perms: int = 10,
    ) -> None:
        super().__init__(baseline_score, baseline_perms)
        self.substrategy = substrategy
        self.sample_rates = sample_rates
        self.n_repeat = n_repeat
        self.random_state = random_state
        self.strat_id = f"Subsample[{self.substrategy.strat_id}]"

    def __call__(
        self,
        rep1: List[Tensor],
        rep2: List[Tensor],
        sim_func: Callable[[Tensor, Tensor], float],
        **kwds: Any,
    ) -> Dict[str, List[Any]]:
        main_rng = np.random.default_rng(self.random_state)
        overall_results = defaultdict(list)
        for rate in self.sample_rates:
            for _ in range(self.n_repeat):
                sample_seed = main_rng.integers(0, 123456789)
                rng = np.random.default_rng(sample_seed)
                sample_size = int(rate * len(rep1))
                selected = rng.choice(
                    len(rep1),
                    sample_size,
                )
                rep1_sample = [rep1[i] for i in selected]
                rep2_sample = [rep2[i] for i in selected]
                results = self.substrategy(rep1_sample, rep2_sample, sim_func)
                for key, value in results.items():
                    overall_results[key].append(value)
                overall_results["sample_seed"].append(sample_seed)
                overall_results["sample_size"].append(sample_size)
                overall_results["sample_rate"].append(rate)
        return overall_results
