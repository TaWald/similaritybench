import itertools
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

from ke.manual_introspection.scripts import grouped_model_results as grm
from ke.manual_introspection.scripts.compare_representations_of_models import ckpt_results
from ke.manual_introspection.scripts.compare_representations_of_models import compare_models_representations_parallel
from ke.manual_introspection.scripts.compare_representations_of_models import get_ckpts_from_paths
from ke.manual_introspection.scripts.compare_representations_of_models import get_matching_model_dirs_of_ke_ensembles
from ke.manual_introspection.scripts.compare_representations_of_models import (
    get_models_with_ids_from_dir_and_first_model,
)
from ke.manual_introspection.scripts.compare_representations_of_models import ModelToModelComparison
from ke.manual_introspection.scripts.compare_representations_of_models import SeedResult
from ke.util.file_io import save_json
from tqdm import tqdm

cur_file_path: Path = Path(__file__)
json_results_path = cur_file_path.parent.parent / "representation_comp_results"


def create_comps_between_regularized_unregularized_by_id(hparam: dict, overwrite=False):
    for wanted_hparams_name, hparams_dict in hparam.items():
        # Contains path to folder and hparams of directory
        model_dirs: list[tuple[Path, dict]] = get_matching_model_dirs_of_ke_ensembles(ckpt_results, hparams_dict)

        model_ids = [0, 1]

        model_paths: list[SeedResult] = get_models_with_ids_from_dir_and_first_model(model_dirs, model_ids)
        model_ckpt_paths: list[SeedResult] = [get_ckpts_from_paths(mp) for mp in model_paths]
        other_ckpt_paths = deepcopy(model_ckpt_paths)

        cross_seed_unregularized_paths: list[tuple[Path, Path]] = list(
            itertools.combinations([mcp.checkpoints[0] for mcp in model_ckpt_paths], r=2)
        )
        cross_seed_regularized_paths: list[tuple[Path, Path]] = list(
            itertools.combinations([mcp.checkpoints[1] for mcp in model_ckpt_paths], r=2)
        )

        cross_seed_unregularized_regularized_paths = []
        for mcp in model_ckpt_paths:
            for ocp in other_ckpt_paths:
                if mcp.hparams["group_id_i"] == ocp.hparams["group_id_i"]:
                    continue
                else:
                    cross_seed_unregularized_regularized_paths.append((mcp.checkpoints[0], ocp.checkpoints[1]))

        in_seed_regularized_unregularized = [(mcp.checkpoints[0], mcp.checkpoints[1]) for mcp in model_ckpt_paths]

        non_reg_2_non_reg = json_results_path / f"non_reg_2_non_reg__{wanted_hparams_name}.json"
        in_seed_non_reg_to_reg = json_results_path / f"in_seed_non_reg_to_reg__{wanted_hparams_name}.json"
        cross_seed_reg_2_non_reg = json_results_path / f"cross_seed_reg_2_non_reg__{wanted_hparams_name}.json"
        reg_to_reg_comp = json_results_path / f"cross_seed_reg_2_reg__{wanted_hparams_name}.json"

        # Normal comparison ----------------------------------------------------------------------
        if (not overwrite) and non_reg_2_non_reg.exists():
            pass
        else:
            layer_results: list[ModelToModelComparison] = []
            for a, b in tqdm(cross_seed_unregularized_paths[:20]):
                res = compare_models_representations_parallel(model_a=a, model_b=b, hparams=hparams_dict)
                layer_results.append(res)
            save_json(
                [{**asdict(lr), **hparams_dict} for lr in layer_results],
                non_reg_2_non_reg,
            )
        # Cross seed comparison unregularized regularizd -----------------------------------------
        if (not overwrite) and in_seed_non_reg_to_reg.exists():
            pass
        else:
            layer_results: list[ModelToModelComparison] = []
            for a, b in tqdm(in_seed_regularized_unregularized):
                res = compare_models_representations_parallel(model_a=a, model_b=b, hparams=hparams_dict)
                layer_results.append(res)
            save_json(
                [{**asdict(lr), **hparams_dict} for lr in layer_results],
                in_seed_non_reg_to_reg,
            )
        # Cross seed comparison unregularized to regularized  --------------------------------------
        if (not overwrite) and cross_seed_reg_2_non_reg.exists():
            pass
        else:
            layer_results: list[ModelToModelComparison] = []
            for a, b in tqdm(cross_seed_unregularized_regularized_paths):
                res = compare_models_representations_parallel(model_a=a, model_b=b, hparams=hparams_dict)
                layer_results.append(res)
            save_json(
                [{**asdict(lr), **hparams_dict} for lr in layer_results],
                cross_seed_reg_2_non_reg,
            )
        # Reg2Reg cross-seed comparison
        if (not overwrite) and reg_to_reg_comp.exists():
            pass
        else:
            layer_results: list[ModelToModelComparison] = []
            for a, b in tqdm(cross_seed_regularized_paths):
                res = compare_models_representations_parallel(model_a=a, model_b=b, hparams=hparams_dict)
                layer_results.append(res)
            save_json(
                [{**asdict(lr), **hparams_dict} for lr in layer_results],
                reg_to_reg_comp,
            )

    return


def create_same_seed_ensemble_comparisons(hparam: dict, overwrite=False):
    for wanted_hparams_name, hparams_dict in hparam.items():
        models = get_matching_model_dirs_of_ke_ensembles(ckpt_results, hparams_dict)
        model_paths: list[SeedResult] = get_models_with_ids_from_dir_and_first_model(models, [0, 1, 2, 3, 4])
        model_ckpt_paths: list[SeedResult] = [get_ckpts_from_paths(mp) for mp in model_paths]
        n_models_eq_5: list[bool] = [len(mp.checkpoints.values()) == 5 for mp in model_paths]
        assert all(n_models_eq_5), f"Some models did not contain 5 models. {model_paths}"

        this_output_file = json_results_path / f"{wanted_hparams_name}.json"
        if (not overwrite) and this_output_file.exists():
            continue

        ensemble_layer_results: list[ModelToModelComparison] = []

        seed_result: SeedResult
        for seed_result in tqdm(model_ckpt_paths[:20]):
            combis = itertools.combinations_with_replacement(seed_result.checkpoints.keys(), r=2)
            for a, b in tqdm(list(combis)):
                res = compare_models_representations_parallel(
                    model_a=seed_result.checkpoints[a], model_b=seed_result.checkpoints[b], hparams=hparams_dict
                )
                res.m_id_a = int(a)
                res.m_id_b = int(b)
                res.g_id_a = seed_result.hparams["group_id_i"]
                res.g_id_b = seed_result.hparams["group_id_i"]
                ensemble_layer_results.append(res)
        save_json(
            [{**asdict(lr), **hparams_dict} for lr in ensemble_layer_results],
            json_results_path / f"{wanted_hparams_name}.json",
        )
    return


if __name__ == "__main__":
    create_same_seed_ensemble_comparisons(grm.lincka_ensemble_layer_IN100_DIFF, overwrite=True)
