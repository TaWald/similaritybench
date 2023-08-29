import itertools
from dataclasses import asdict
from pathlib import Path
from warnings import warn

import torch
from ke.manual_introspection.compare import compare_models_functional
from ke.manual_introspection.comparison_helper import ModelToModelComparison
from ke.manual_introspection.comparison_helper import OutputEnsembleResults
from ke.manual_introspection.comparison_helper import SeedResult
from ke.manual_introspection.scripts.compare_representations_of_models import compare_models_representations_parallel
from ke.manual_introspection.scripts.compare_representations_of_models import get_ckpts_from_paths
from ke.manual_introspection.scripts.compare_representations_of_models import get_matching_model_dirs_of_ke_ensembles
from ke.manual_introspection.scripts.compare_representations_of_models import (
    get_models_with_ids_from_dir_and_first_model,
)
from ke.util.file_io import load_json
from ke.util.file_io import save_json
from tqdm import tqdm


def reshape(acti: torch.Tensor):
    acti = torch.flatten(acti, start_dim=1)
    return acti


def get_seed_results_of_interest(ckpt_result_path: Path, hparam: dict, model_ids: list[int]) -> list[SeedResult]:
    """Returns SeedResult of all sequences that match the requirements, specified by hparam."""
    models = get_matching_model_dirs_of_ke_ensembles(ckpt_result_path, hparam)
    model_paths: list[SeedResult] = get_models_with_ids_from_dir_and_first_model(models, model_ids)
    model_ckpt_paths: list[SeedResult] = [get_ckpts_from_paths(mp) for mp in model_paths]
    return model_ckpt_paths


def filter_all_results_that_dont_have_n_models(seed_results: list[SeedResult], n_models: int) -> list[SeedResult]:
    """Filters the seed_results if they do not have sufficient amount of models."""
    filtered_seed_results: list[SeedResult] = []
    for mcp in seed_results:
        ckpt_paths = mcp.checkpoints.values()

        if len(ckpt_paths) >= n_models:
            if all([ckpt_path.exists() for ckpt_path in ckpt_paths][:n_models]):
                filtered_seed_results.append(mcp)
    return filtered_seed_results


def remove_all_but_interesting_checkpoints(seed_results: list[SeedResult], n_models: list[int]) -> list[SeedResult]:
    """Insert only the interesting checkpoints."""
    filtered_seed_results: list[SeedResult] = []
    for mcp in seed_results:
        new_see_result = SeedResult(mcp.hparams)
        new_see_result.checkpoints = {k: v for k, v in mcp.checkpoints.items() if k in n_models}
        new_see_result.models = {k: v for k, v in mcp.models.items() if k in n_models}

    return filtered_seed_results


def get_only_first_and_second_model(seed_results: list[SeedResult]) -> list[SeedResult]:
    """Removes every Checkpoint and Model but the first and second from the list of SeedResults.""" ""
    filtered_seed_results: list[SeedResult] = []
    for mcp in seed_results:
        ckpt_paths = mcp.checkpoints.values()
        all_exist = (all([ckpt_path.exists() for ckpt_path in ckpt_paths])) and (len(ckpt_paths) >= 2)
        if all_exist:
            new_model = SeedResult(mcp.hparams)
            new_model.checkpoints = {0: mcp.checkpoints[0], 1: mcp.checkpoints[1]}
            new_model.models = {0: mcp.models[0], 1: mcp.models[1]}
            filtered_seed_results.append(new_model)
    return filtered_seed_results


def compare_representations_unregularized_models_to_each_other(
    hparam: dict,
    json_results_path: Path,
    ckpt_result_path: Path,
    overwrite=False,
):
    """
    Compares the unregularized, very first models (id=0) of the specified hparams to each other (across seeds).
    """
    for wanted_hparams_name, hparams_dict in hparam.items():
        this_output_file = json_results_path / f"{wanted_hparams_name}.json"
        if (not overwrite) and this_output_file.exists():
            continue

        model_ckpt_paths: list[SeedResult] = get_seed_results_of_interest(ckpt_result_path, hparams_dict, [0, 1])

        layer_results: list[ModelToModelComparison] = []
        all_ckpts = list(set([str(mcp.checkpoints[0]) for mcp in model_ckpt_paths]))  # only first models

        for ckpt_a, ckpt_b in tqdm(list(itertools.combinations(all_ckpts, 2))):
            res = compare_models_representations_parallel(ckpt_a, ckpt_b, hparams=hparams_dict)
            layer_results.append(res)
        save_json(
            [{**asdict(lr), **hparams_dict} for lr in layer_results],
            json_results_path / f"{wanted_hparams_name}.json",
        )


def compare_representations_same_seed_first_to_second(
    hparam: dict,
    json_results_path: Path,
    ckpt_result_path: Path,
    overwrite=False,
):
    """
    Compares the very first (original) model to the next model that was regularized based off of it.
    Only compares model 0 to model 1 of the same group_id (same sequence).
    """
    for wanted_hparams_name, hparams_dict in hparam.items():
        model_ckpt_paths: list[SeedResult] = get_seed_results_of_interest(ckpt_result_path, hparams_dict, [0, 1])

        this_output_file = json_results_path / f"{wanted_hparams_name}.json"
        if (not overwrite) and this_output_file.exists():
            continue

        layer_results: list[ModelToModelComparison] = []
        seed_result: SeedResult
        for seed_result in tqdm(model_ckpt_paths):
            combis = itertools.combinations(seed_result.checkpoints.values(), r=2)
            for a, b in combis:
                res = compare_models_representations_parallel(model_a=a, model_b=b, hparams=hparams_dict)
                layer_results.append(res)
        if len(layer_results) == 0:
            warn("Nothing to save. skipping file creation!")
        else:
            save_json(
                [{**asdict(lr), **hparams_dict} for lr in layer_results],
                json_results_path / f"{wanted_hparams_name}.json",
            )
    return


def compare_functional_same_seed_ensemble(
    hparam: dict,
    n_models: int,
    json_results_path: Path,
    ckpt_result_path: Path,
    overwrite=False,
):
    """Creates the functional comparison of the ensemble of [2, 3, ..., n_models] of the same seed (sequence)."""

    model_ids = list(range(n_models))
    for wanted_hparams_name, hparams_dict in hparam.items():
        model_ids = list(range(n_models))
        model_ckpt_paths = get_seed_results_of_interest(ckpt_result_path, hparams_dict, model_ids)
        model_ckpt_paths = filter_all_results_that_dont_have_n_models(model_ckpt_paths, n_models)

        json_results_path.mkdir(parents=True, exist_ok=True)
        this_output_file = json_results_path / f"functional_{wanted_hparams_name}.json"
        if this_output_file.exists():
            existing_json = load_json(this_output_file)
            if len(existing_json) == ((len(model_ids) - 1) * len(model_ckpt_paths)):
                print("Skipping existing file")
                continue

        # ToDo: Currently the checkpoints seem to not provide the right performance?
        #   At least the different models preform differently Whats the Issue! -> Checkpoint?/Loading?/Eval?

        layer_results: list[OutputEnsembleResults] = []
        seed_result: SeedResult
        for seed_result in tqdm(model_ckpt_paths):
            layer_results.extend(
                compare_models_functional(list(seed_result.checkpoints.values()), hparams=hparams_dict)
            )

        if len(layer_results) == 0:
            warn("Nothing to save. skipping file creation!")
            continue
        else:
            save_json(
                [{**asdict(lr), **hparams_dict} for lr in layer_results],
                json_results_path / f"functional_{wanted_hparams_name}.json",
            )
    return


def compare_representation_same_seed_ensemble(
    hparam: dict,
    n_models: int,
    json_results_path: Path,
    ckpt_result_path: Path,
    overwrite=False,
):
    for wanted_hparams_name, hparams_dict in hparam.items():
        print(f"{wanted_hparams_name}.json")

        model_ids = list(range(n_models))
        models = get_matching_model_dirs_of_ke_ensembles(ckpt_result_path, hparams_dict)
        model_paths: list[SeedResult] = get_models_with_ids_from_dir_and_first_model(models, model_ids)
        existing_ckpts: list[SeedResult] = filter_all_results_that_dont_have_n_models(model_paths, n_models)

        this_output_file = json_results_path / f"{wanted_hparams_name}.json"
        if (not overwrite) and this_output_file.exists():
            continue

        ensemble_layer_results: list[ModelToModelComparison] = []

        seed_result: SeedResult
        for seed_result in tqdm(existing_ckpts):
            combis = list(itertools.combinations_with_replacement(seed_result.checkpoints.keys(), r=2))
            for a, b in tqdm(combis):
                res = compare_models_representations_parallel(
                    model_a=seed_result.checkpoints[a], model_b=seed_result.checkpoints[b], hparams=hparams_dict
                )
                res.m_id_a = int(a)
                res.m_id_b = int(b)
                res.g_id_a = seed_result.hparams["group_id_i"]
                res.g_id_b = seed_result.hparams["group_id_i"]
                ensemble_layer_results.append(res)
                if res.accuracy_reg < 0.8:
                    print(f"bad accuracy: {res.accuracy_reg}")
                    bad_group_id = seed_result.hparams["group_id_i"]
                    # remove bad group id from list
                    tmp_res = []
                    for cnt, res in enumerate(ensemble_layer_results):
                        if res.g_id_b != bad_group_id:
                            tmp_res.append(res)
                    ensemble_layer_results = tmp_res
                    break

        save_json(
            [{**asdict(lr), **hparams_dict} for lr in ensemble_layer_results],
            json_results_path / f"{wanted_hparams_name}.json",
        )
    return


def compare_just_one_sequence_of_5_models(
    hparam: dict,
    json_results_path: Path,
    ckpt_result_path: Path,
    overwrite=False,
):
    for wanted_hparams_name, hparams_dict in hparam.items():
        # Contains path to folder and hparams of directory
        n_models = 5
        model_ids = list(range(n_models))

        model_ckpt_paths: list[SeedResult] = get_seed_results_of_interest(ckpt_result_path, hparams_dict, model_ids)

        first_seed_with_5_models: SeedResult = [mcp for mcp in model_ckpt_paths if len(mcp.checkpoints) > n_models][0]

        json_results = {"hparams": first_seed_with_5_models.hparams, "results": []}
        for i in range(n_models):
            for j in range(n_models):
                res = compare_models_representations_parallel(
                    model_a=first_seed_with_5_models.checkpoints[i],
                    model_b=first_seed_with_5_models.checkpoints[j],
                    hparams=first_seed_with_5_models.hparams,
                )
                json_results["results"].append({"id_x": i, "id_y": j, "values": asdict(res)})
        out_vals = json_results_path / f"cka_between_5_consecutive_models__{wanted_hparams_name}.json"
        save_json(json_results, out_vals)
    return
