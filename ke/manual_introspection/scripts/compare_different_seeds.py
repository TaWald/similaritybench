import itertools
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

from tqdm import tqdm

from ke.manual_introspection.scripts import grouped_model_results as grm
from ke.manual_introspection.scripts.compare_representations_of_models import ModelToModelComparison
from ke.manual_introspection.scripts.compare_representations_of_models import ckpt_results, SeedResult
from ke.manual_introspection.scripts.compare_representations_of_models import compare_models_parallel
from ke.manual_introspection.scripts.compare_representations_of_models import get_ckpts_from_paths
from ke.manual_introspection.scripts.compare_representations_of_models import get_matching_model_dirs_of_ke_ensembles
from ke.manual_introspection.scripts.compare_representations_of_models import (
    get_models_with_ids_from_dir_and_first_model,
)
from ke.util.file_io import save_json
from IPython import embed

cur_file_path: Path = Path(__file__)
json_results_path = cur_file_path.parent.parent / "plots"


def create_comps_between_regularized_unregularized_by_id(hparam: dict, overwrite=False):
    for wanted_hparams_name, hparams_dict in hparam.items():
        # Contains path to folder and hparams of directory
        model_dirs: list[tuple[Path, dict]] = get_matching_model_dirs_of_ke_ensembles(ckpt_results, hparams_dict)

        model_ids = [0, 1]

        model_paths: list[SeedResult] = get_models_with_ids_from_dir_and_first_model(model_dirs, model_ids)
        model_ckpt_paths: list[SeedResult] = [get_ckpts_from_paths(mp) for mp in model_paths]
        other_ckpt_paths = deepcopy(model_ckpt_paths)

        cross_seed_unregularized_paths: list[tuple[Path, Path]] = list(itertools.combinations([mcp.checkpoints[0] for mcp in model_ckpt_paths], r=2))
        cross_seed_regularized_paths: list[tuple[Path, Path]] = list(itertools.combinations([mcp.checkpoints[1] for mcp in model_ckpt_paths], r=2))

        cross_seed_unregularized_regularized_paths = []
        for mcp in model_ckpt_paths:
            for ocp in other_ckpt_paths:
                if mcp.hparams['group_id_i'] == ocp.hparams['group_id_i']:
                    continue
                else:
                    cross_seed_unregularized_regularized_paths.append((mcp.checkpoints[0], ocp.checkpoints[1]))

        in_seed_regularized_unregularized = [(mcp.checkpoints[0], mcp.checkpoints[1]) for mcp in model_ckpt_paths]

        hparams = model_ckpt_paths[0].hparams

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
            for a,b  in tqdm(cross_seed_regularized_paths):
                res = compare_models_parallel(model_a=a, model_b=b, hparams=hparams_dict)
                layer_results.append(res)
            save_json(
                [{**asdict(lr), **hparams_dict} for lr in layer_results],
                reg_to_reg_comp,
            )

    return


def create_comps_between_single_5_consecutive_models(hparam: dict, overwrite=False):
    for wanted_hparams_name, hparams_dict in hparam.items():
        # Contains path to folder and hparams of directory
        model_dirs: list[tuple[Path, dict]] = get_matching_model_dirs_of_ke_ensembles(ckpt_results, hparams_dict)

        n_models = 5

        model_paths: list[SeedResult] = get_models_with_ids_from_dir_and_first_model(model_dirs, list(range(n_models)))
        model_ckpt_paths: list[SeedResult] = [get_ckpts_from_paths(mp) for mp in model_paths]

        first_seed_with_5_models: SeedResult = [mcp for mcp in model_ckpt_paths if len(mcp.checkpoints) > n_models][0]

        json_results = {
            "hparams": first_seed_with_5_models.hparams,
            "results": []}
        for i in range(n_models):
            for j in range(n_models):
                res = compare_models_parallel(
                    model_a=first_seed_with_5_models.checkpoints[i],
                    model_b=first_seed_with_5_models.checkpoints[j],
                    hparams=first_seed_with_5_models.hparams)
                json_results['results'].append({
                    "id_x": i,
                    "id_y": j,
                    "values": asdict(res)})
        out_vals = json_results_path / f"cka_between_5_consecutive_models__{wanted_hparams_name}.json"
        save_json(json_results, out_vals)
    return




if __name__ == "__main__":
    create_comps_between_regularized_unregularized_by_id(grm.layer_9_tdepth_1_expvar_1)
    # ToDo: Find a run that had more than 5 models in it.
    create_comps_between_single_5_consecutive_models(grm.layer_9_tdepth_1_expvar_1)