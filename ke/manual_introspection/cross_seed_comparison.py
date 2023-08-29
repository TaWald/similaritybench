import itertools
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

from ke.manual_introspection.compare import compare_models_parallel
from ke.manual_introspection.comparison_helper import ModelToModelComparison
from ke.manual_introspection.comparison_helper import SeedResult
from ke.manual_introspection.comparison_json_creator import get_seed_results_of_interest
from ke.util.file_io import save_json
from tqdm import tqdm


cur_file_path: Path = Path(__file__)
json_results_path = cur_file_path.parent.parent / "representation_comp_results"


def compare_models_across_seeds(
    hparam: dict,
    json_results_path: Path,
    ckpt_result_path: Path,
    overwrite=False,
):
    for wanted_hparams_name, hparams_dict in hparam.items():
        # Contains path to folder and hparams of directory
        model_ckpt_paths: list[SeedResult] = get_seed_results_of_interest(ckpt_result_path, hparam, [0, 1])
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
                res = compare_models_parallel(model_a=a, model_b=b, hparams=hparams_dict)
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
                res = compare_models_parallel(model_a=a, model_b=b, hparams=hparams_dict)
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
                res = compare_models_parallel(model_a=a, model_b=b, hparams=hparams_dict)
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
                res = compare_models_parallel(model_a=a, model_b=b, hparams=hparams_dict)
                layer_results.append(res)
            save_json(
                [{**asdict(lr), **hparams_dict} for lr in layer_results],
                reg_to_reg_comp,
            )

    return
