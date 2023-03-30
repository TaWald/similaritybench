import argparse
import sys
from pathlib import Path

import numpy as np
from ke.arch.arch_loading import load_model_from_info_file
from ke.training.ke_train_modules.CalibrationModule import Calibrator
from ke.util import data_structs as ds
from ke.util import default_parser_args as dpa
from ke.util import file_io
from ke.util import name_conventions as nc
from ke.util.default_params import get_default_parameters
from ke.util.file_io import load_json
from ke.util.file_io import save_json
from ke.util.gpu_cluster_worker_nodes import get_workers_for_current_node
from ke.util.load_own_objects import load_datamodule_from_info
from ke.util.status_check import output_json_has_nans
from scripts.post_train_additions.utils import clean_up_after_processing
from scripts.post_train_additions.utils import should_process_a_dir
from tqdm import tqdm


def load_temperature_from_info(model_info: ds.FirstModelInfo):
    if model_info.is_calibrated():
        return load_json(model_info.path_calib_json)["val"]["temperature"]
    else:
        return np.NAN


def calibrate_model(model_info: ds.FirstModelInfo) -> None:
    """
    Calibrates a model based on the info file given.
    :param model_info: Model info parametrization file.
    """

    if model_info.is_calibrated():
        return
    elif model_info.is_trained():
        output_json = load_json(model_info.path_output_json)
        if output_json_has_nans(output_json):
            return

        val_dataloader_kwargs = {
            "shuffle": False,
            "drop_last": False,
            "pin_memory": False,
            "batch_size": 100,
            "num_workers": get_workers_for_current_node(),
            "persistent_workers": False,
        }
        model = load_model_from_info_file(model_info, load_ckpt=True)
        dataloader = load_datamodule_from_info(model_info)

        calib = Calibrator(model)
        calib.calibrate(dataloader.val_dataloader(model_info.split, ds.Augmentation.VAL, **val_dataloader_kwargs))
        validation_calib = calib.calculate_calibration_effect(
            dataloader.val_dataloader(model_info.split, ds.Augmentation.VAL, **val_dataloader_kwargs)
        )
        test_calib = calib.calculate_calibration_effect(
            dataloader.test_dataloader(ds.Augmentation.VAL, **val_dataloader_kwargs)
        )
        calibration_results = {"val": validation_calib, "test": test_calib}
        save_json(calibration_results, model_info.path_calib_json)


def main():
    parser = argparse.ArgumentParser(description="Specify model hyperparams.")
    dpa.dir_parser_arguments(parser)
    args = parser.parse_args()
    ke_dirname = args.ke_dir_name

    base_data_path = Path(file_io.get_experiments_data_root_path())
    base_ckpt_path = Path(file_io.get_experiments_checkpoints_root_path())

    ke_data_path = base_data_path / ke_dirname
    ke_ckpt_path = base_ckpt_path / ke_dirname

    paths = list(sorted(ke_data_path.iterdir()))

    for res in paths:

        dir_name = res.name

        all_training_infos: list[ds.FirstModelInfo] = []
        kedp = ke_data_path / dir_name
        kecp = ke_ckpt_path / dir_name

        if dir_name.startswith(nc.KE_FIRST_MODEL_DIR.split("__")[0]):
            _, dataset_name, architecture_name = dir_name.split("__")
            p: ds.Params = get_default_parameters(architecture_name, ds.Dataset(dataset_name))
            for sub_dir in res.iterdir():
                group_id = int(sub_dir.name.split("_")[-1])
                all_training_infos.append(
                    file_io.get_first_model(
                        ke_data_path=ke_data_path,
                        ke_ckpt_path=ke_ckpt_path,
                        params=p,
                        group_id=group_id,
                    )
                )
        else:
            if ke_dirname == nc.KNOWLEDGE_EXTENSION_DIRNAME:
                all_training_infos = file_io.get_trained_ke_models(kedp, kecp)
            elif ke_dirname == nc.KE_OUTPUT_REGULARIZATION_DIRNAME:
                all_training_infos = file_io.get_trained_keo_models(kedp, kecp)
            elif ke_dirname == nc.KE_ADVERSARIAL_LENSE_DIRNAME:
                all_training_infos = file_io.get_trained_adversarial_lense_models(kedp, kecp)
            elif ke_dirname == nc.KNOWLEDGE_ADVERSARIAL_DIRNAME:
                all_training_infos = file_io.get_trained_ke_adv_models(kedp, kecp)
            elif ke_dirname == nc.KNOWLEDGE_UNUSEABLE_DIRNAME:
                all_training_infos = file_io.get_trained_ke_unuseable_models(kedp, kecp)
            else:
                raise NotImplementedError

            # Do the baseline model creation if it not already exists!
        if all([ti.is_calibrated() for ti in all_training_infos]):
            print(f"{res} is already calibrated.")
            continue

        if not should_process_a_dir(res):
            continue

        print(f"Calibrating {res}")
        for train_info in tqdm(all_training_infos):
            calibrate_model(train_info)
        clean_up_after_processing(res)
    return


if __name__ == "__main__":
    main()
    sys.exit(0)
