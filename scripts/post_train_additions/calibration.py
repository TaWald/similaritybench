import argparse
import sys
from pathlib import Path

from rep_trans.arch.arch_loading import load_model_from_info_file
from rep_trans.training.ke_train_modules.CalibrationModule import Calibrator
from rep_trans.util import data_structs as ds
from rep_trans.util import default_parser_args as dpa
from rep_trans.util import file_io
from rep_trans.util import name_conventions as nc
from rep_trans.util.default_params import get_default_parameters
from rep_trans.util.file_io import load_datamodule_from_info
from rep_trans.util.file_io import save_json
from rep_trans.util.gpu_cluster_worker_nodes import get_workers_for_current_node
from rep_trans.util.status_check import is_calibrated
from scripts.post_train_additions.utils import clean_up_after_processing
from scripts.post_train_additions.utils import should_process_a_file


def calibrate_model(model_info: ds.BasicTrainingInfo) -> None:
    """
    Calibrates a model based on the info file given.
    :param model_info: Model info parametrization file.
    """

    if is_calibrated(model_info.path_data_root):
        return
    else:
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

    if ke_dirname == nc.KNOWLEDGE_EXTENSION_DIRNAME:
        encoder = nc.KENameEncoder
    elif ke_dirname == nc.KE_OUTPUT_REGULARIZATION_DIRNAME:
        encoder = nc.KEOutputNameEncoder
    elif ke_dirname == nc.KE_ADVERSARIAL_LENSE_DIRNAME:
        encoder = nc.KEAdversarialLenseOutputNameEncoder
    elif ke_dirname == nc.KNOWLEDGE_ADVERSARIAL_DIRNAME:
        encoder = nc.KEAdversarialNameEncoder
    elif ke_dirname == nc.KNOWLEDGE_UNUSEABLE_DIRNAME:
        encoder = nc.KEUnusableDownstreamNameEncoder
    else:
        raise NotImplementedError

    base_data_path = Path(file_io.get_experiments_data_root_path())
    base_ckpt_path = Path(file_io.get_experiments_checkpoints_root_path())

    ke_data_path = base_data_path / ke_dirname
    ke_ckpt_path = base_ckpt_path / ke_dirname

    for res in ke_data_path.iterdir():
        dir_name = res.name

        kedp = ke_data_path / dir_name
        kecp = ke_ckpt_path / dir_name

        decoded = encoder.decode(dir_name)
        exp_name = decoded[0]
        dataset_name = decoded[1]
        architecture_name = decoded[2]
        prev_training_infos: list[ds.BasicTrainingInfo]
        if ke_dirname == nc.KNOWLEDGE_EXTENSION_DIRNAME:
            group_id = decoded[6]
            prev_training_infos = file_io.get_trained_ke_models(kedp, kecp)
        elif ke_dirname == nc.KE_OUTPUT_REGULARIZATION_DIRNAME:
            group_id = decoded[3]
            prev_training_infos = file_io.get_trained_keo_models(kedp, kecp)
        elif ke_dirname == nc.KE_ADVERSARIAL_LENSE_DIRNAME:
            group_id = decoded[3]
            prev_training_infos = file_io.get_trained_adversarial_lense_models(kedp, kecp)
        elif ke_dirname == nc.KNOWLEDGE_ADVERSARIAL_DIRNAME:
            group_id = decoded[6]
            prev_training_infos = file_io.get_trained_ke_adv_models(kedp, kecp)
        elif ke_dirname == nc.KNOWLEDGE_UNUSEABLE_DIRNAME:
            group_id = decoded[6]
            prev_training_infos = file_io.get_trained_ke_unuseable_models(kedp, kecp)
        else:
            raise NotImplementedError

        # Do the baseline model creation if it not already exists!
        p: ds.Params = get_default_parameters(architecture_name, ds.Dataset(dataset_name))

        first_model = file_io.get_first_model(
            experiment_description=exp_name,
            ke_data_path=ke_data_path,
            ke_ckpt_path=ke_ckpt_path,
            architecture=architecture_name,
            dataset=dataset_name,
            learning_rate=p.learning_rate,
            split=p.split,
            weight_decay=p.weight_decay,
            batch_size=p.batch_size,
            group_id=group_id,
        )

        all_training_infos: list[ds.BasicTrainingInfo]
        all_training_infos = [first_model] + prev_training_infos

        if not should_process_a_file(res):
            continue

        for train_info in all_training_infos:
            calibrate_model(train_info)
        clean_up_after_processing(res)
    return


if __name__ == "__main__":
    main()
    sys.exit(0)
