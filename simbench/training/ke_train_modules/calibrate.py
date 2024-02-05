from simbench.arch.arch_loading import load_model_from_info_file
from simbench.training.ke_train_modules.CalibrationModule import Calibrator
from simbench.util import data_structs as ds
from simbench.util.file_io import load_json
from simbench.util.file_io import save_json
from simbench.util.gpu_cluster_worker_nodes import get_workers_for_current_node
from simbench.util.load_own_objects import load_datamodule_from_info
from simbench.util.status_check import output_json_has_nans


def calibrate_model(model_info: ds.FirstModelInfo) -> None:
    """
    Calibrates a model based on the info file given.
    :param model_info: Model info parametrization file.
    """

    if model_info.is_calibrated() or (not model_info.model_converged()):
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
