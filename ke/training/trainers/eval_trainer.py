from __future__ import annotations

import os
from copy import deepcopy

import numpy as np
from augmented_cifar.scripts.get_dataloaders import get_augmented_cifar100_test_dataloader
from augmented_cifar.scripts.get_dataloaders import get_augmented_cifar10_test_dataloader
from ke.training.ke_train_modules.EvaluationLightningModule import EvaluationLightningModule
from ke.util import data_structs as ds
from ke.util import file_io
from ke.util.load_own_objects import load_datamodule_from_info
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader


def scalarize_robustness(all_robustness_tests: dict, is_ensemble: bool) -> dict:
    """
    Creates a scalar measure of the robustness and the mean of all the values for each dimension.

    Thanks CoPilot ... Takes a dict of dicts of dicts and returns a dict of dicts with the mean of the innermost dict.
    :param all_robustness_tests: Dict of dicts of dicts. Outermost dict contains augmentation type,
     the innermost dict is parameter strength
     :param is_ensemble: If the robustness is calculated for an ensemble or a single model.
    """
    means: dict[str, float] = {}
    eval_key = "ensemble_accuracy" if is_ensemble else "accuracy"
    for k in sorted(all_robustness_tests.keys()):
        means[k] = float(np.mean([res_dict[eval_key] for res_dict in all_robustness_tests[k].values()]))
    all_mean_value = float(np.mean(list(means.values())))
    return {"mean_robustness": all_mean_value, "dimensionwise_mean": means}


class EvalTrainer:
    def __init__(self, model_infos: list[ds.FirstModelInfo]):
        self.model: EvaluationLightningModule = EvaluationLightningModule(
            model_infos, model_infos[0].architecture, model_infos[0].dataset
        )
        self.model_infos = model_infos
        self.num_workers = 0  # get_workers_for_current_node()

        # Create them. Should not exist though or overwrite would happen!

        if "RAW_DATA" in os.environ:
            dataset_path = os.environ["RAW_DATA"]
        elif "data" in os.environ:
            dataset_path = os.environ["data"]
        else:
            raise EnvironmentError

        self.dataset_path = dataset_path
        self.test_kwargs = {
            "shuffle": False,
            "drop_last": False,
            "pin_memory": True,
            "batch_size": 128,
            "num_workers": self.num_workers,
            "persistent_workers": False,
        }

    def _eval_performance(
        self, dataloader: DataLoader, single: bool, ensemble: bool, also_calibrated: bool
    ) -> dict[str, dict]:
        """
        Measures the generalization of the model by evaluating it on an augmented version of the test set.
        """
        test_dataloader = dataloader

        trainer = Trainer(
            enable_checkpointing=False,
            max_epochs=None,
            accelerator="gpu",
            devices=1,
            precision=32,
            default_root_dir=None,
            enable_progress_bar=False,
            logger=False,
            profiler=None,
        )
        self.model.cuda()
        self.model.eval()

        trainer.validate(self.model, test_dataloader)
        ret_dict: dict = {}
        if single:
            single_metrics = deepcopy(self.model.all_single_metrics)
            ret_dict["single"] = single_metrics
        if ensemble:
            ensemble_metrics = deepcopy(self.model.all_ensemble_metrics)
            ret_dict["ensemble"] = ensemble_metrics
            if also_calibrated:
                with self.model.calibration_mode():
                    trainer.validate(self.model, test_dataloader)
                    ret_dict["calibrated_ensemble"] = deepcopy(self.model.all_ensemble_metrics)

        return ret_dict

    def measure_performance(self, single: bool, ensemble: bool, also_calibrated: bool) -> None:
        """
        Measures the performance of the model(s) on the test set. If Ensemble is passed it calcualtes
        the performance of the various ensemble combinations that exist. (e.g. n = 2, 3, 4, 5, 6, 7, 8, 9, 10 Models)
        If also_calibrated is passed it also calculates the performance of the ensemble with calibrated
         members on the test set.

        :param single: If true, the performance of the single models is calculated.
        :param ensemble: If true, the performance of the ensembles is calculated.
        :param also_calibrated: If true, the performance of the calibrated ensembles is calculated
        (only if ensemble is true).
        :return: None - Saves the results to a json
        """
        datamodule = load_datamodule_from_info(self.model_infos[-1])
        test_dataloader = datamodule.test_dataloader(ds.Augmentation.VAL, **self.test_kwargs)

        if self.model.infos[-1].sequence_performance_exists(single, ensemble, also_calibrated):
            print("Performance already exists. Skipping")
            return
        perf = self._eval_performance(test_dataloader, single, ensemble, also_calibrated)
        if single:
            file_io.save_json(perf["single"], self.model.infos[-1].sequence_single_json)
        if ensemble:
            file_io.save_json(perf["ensemble"], self.model.infos[-1].sequence_ensemble_json)
        if also_calibrated:
            file_io.save_json(perf["calibrated_ensemble"], self.model.infos[-1].sequence_calibrated_ensemble_json)
        return

    def measure_robustness(self, single: bool, ensemble: bool, also_calibrated: bool) -> None:
        """
        Measures the generalization of the model by evaluating it on an augmented version of the test set.

        """
        if self.model.infos[-1].robust_sequence_performance_exists(single, ensemble, also_calibrated):
            print("Performance already exists. Skipping")
            return

        if self.model_infos[0].dataset == "CIFAR10":
            dataloaders = get_augmented_cifar10_test_dataloader(self.dataset_path, self.test_kwargs)
        elif self.model_infos[0].dataset == "CIFAR100":
            dataloaders = get_augmented_cifar100_test_dataloader(self.dataset_path, self.test_kwargs)
        else:
            raise ValueError(
                f"Trying to measure generalization of unknown dataset! Got {self.model_infos[0].dataset}"
            )

        n_models = len(self.model.models)
        all_results = {i: {} for i in range(n_models)}
        all_ensemble_results = {i: {} for i in range(1, n_models)}
        all_calibrated_ensemble_results = {i: {} for i in range(1, n_models)}
        for dl in dataloaders:
            perf = self._eval_performance(dl.dataloader, True, True, True)
            single_metrics = perf["single"]
            ensemble_metrics = perf["ensemble"]
            calibrated_ensemble_metrics = perf["calibrated_ensemble"]

            for i in range(n_models):
                if single:
                    if dl.name in all_results[i].keys():
                        all_results[i][dl.name].update({str(dl.value): single_metrics[i]})
                    else:
                        all_results[i][dl.name] = {str(dl.value): single_metrics[i]}
                if i > 0:
                    if ensemble:
                        if dl.name in all_ensemble_results[i].keys():
                            all_ensemble_results[i][dl.name].update({str(dl.value): ensemble_metrics[i]})
                        else:
                            all_ensemble_results[i][dl.name] = {str(dl.value): ensemble_metrics[i]}
                    if also_calibrated:
                        if dl.name in all_calibrated_ensemble_results[i].keys():
                            all_calibrated_ensemble_results[i][dl.name].update(
                                {str(dl.value): calibrated_ensemble_metrics[i]}
                            )
                        else:
                            all_calibrated_ensemble_results[i][dl.name] = {
                                str(dl.value): calibrated_ensemble_metrics[i]
                            }

        for i in range(n_models):
            if single:
                single_robustness_result = scalarize_robustness(all_results[i], is_ensemble=False)
                all_results[i].update({"specific_values": single_robustness_result})
            if i > 0:
                if ensemble:
                    ensemble_robustness_result = scalarize_robustness(all_ensemble_results[i], is_ensemble=True)
                    all_ensemble_results[i].update({"specific_values": ensemble_robustness_result})
                if also_calibrated:
                    calibrated_ensemble_robustness_result = scalarize_robustness(
                        all_calibrated_ensemble_results[i], is_ensemble=True
                    )
                    all_calibrated_ensemble_results[i].update(
                        {"specific_values": calibrated_ensemble_robustness_result}
                    )

        if single:
            file_io.save(all_results, self.model.infos[-1].robust_sequence_single_json)
        if ensemble:
            file_io.save(all_ensemble_results, self.model.infos[-1].robust_sequence_ensemble_json)
        if also_calibrated:
            file_io.save(all_calibrated_ensemble_results, self.model.infos[-1].robust_calib_sequence_ensemble_json)
