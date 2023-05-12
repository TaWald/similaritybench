import itertools
from dataclasses import asdict
from pathlib import Path

from ke.manual_introspection.scripts import grouped_model_results as grm
from ke.manual_introspection.scripts.compare_representations_of_models import ckpt_results
from ke.manual_introspection.scripts.compare_representations_of_models import compare_models_parallel
from ke.manual_introspection.scripts.compare_representations_of_models import get_ckpts_from_paths
from ke.manual_introspection.scripts.compare_representations_of_models import get_models_of_ke_ensembles
from ke.manual_introspection.scripts.compare_representations_of_models import (
    get_models_with_ids_from_dir_and_first_model,
)
from ke.manual_introspection.scripts.compare_representations_of_models import ModelToModelComparison
from ke.util.file_io import save_json
from tqdm import tqdm

json_results_path = Path(__file__.parent.parent / "difference_between_two_models")


def create_comparisons(hparam: dict, overwrite=False):
    for wanted_hparams_name, hparams_dict in hparam.items():

        models = get_models_of_ke_ensembles(ckpt_results, hparams_dict)

        model_paths: list[dict[int, Path]] = get_models_with_ids_from_dir_and_first_model(models, [0, 1])
        model_ckpt_paths: list[dict[int, Path]] = [get_ckpts_from_paths(mp) for mp in model_paths]

        # ToDo: 3 Combinations of comparisons need to be conducted
        # Create quadruplets:
        #   All normal models to each other.
        #   Normal model to the one it was regularized to
        #   Normal other models to the regularized ones (that were not regularized on them)
        #   Regularized models to each other
        # --> This shows if they learn similar things.

        this_output_file = json_results_path / f"{wanted_hparams_name}.json"
        if (not overwrite) and this_output_file.exists():
            continue

        layer_results: list[ModelToModelComparison] = []
        for model in tqdm(model_ckpt_paths[:20]):
            combis = itertools.combinations(model.values(), r=2)
            for a, b in combis:
                res = compare_models_parallel(model_a=a, model_b=b, hparams=hparams_dict)
                layer_results.append(res)
        save_json(
            [{**asdict(lr), **hparams_dict} for lr in layer_results],
            json_results_path / f"{wanted_hparams_name}.json",
        )
    return


if __name__ == "__main__":
    create_comparisons(grm.layer_9_tdepth_1_expvar_1)
