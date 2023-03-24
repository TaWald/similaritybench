from __future__ import annotations

import os

import numpy as np
from augmented_cifar.scripts.get_dataloaders import get_augmented_cifar100_test_dataloader
from augmented_cifar.scripts.get_dataloaders import get_augmented_cifar10_test_dataloader
from ke.data.cifar10_dm import CIFAR10DataModule
from ke.training.ke_train_modules.EvaluationLightningModule import EvaluationLightningModule
from ke.util import data_structs as ds
from ke.util import file_io
from ke.util import name_conventions as nc
from pytorch_lightning import Trainer


def scalarize_robustness(all_robustness_tests: dict) -> dict:
    """
    Creates a scalar measure of the robustness and the mean of all the values for each dimension.

    Thanks CoPilot ... Takes a dict of dicts of dicts and returns a dict of dicts with the mean of the innermost dict.
    :param all_robustness_tests: Dict of dicts of dicts. Outermost dict contains augmentation type,
     the innermost dict is parameter strength
    """
    means: dict[str, float] = {}
    for k in sorted(all_robustness_tests.keys()):
        means[k] = float(np.mean([res_dict["accuracy"] for res_dict in all_robustness_tests[k].values()]))
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
            "persistent_workers": True,
        }

    def measure_performance(self):
        """
        Measures the generalization of the model by evaluating it on an augmented version of the test set.

        """
        if self.model_infos[0].dataset == "CIFAR10":
            dataloaders = CIFAR10DataModule().test_dataloader(ds.Augmentation.VAL)
        elif self.model_infos[0].dataset == "CIFAR100":
            dataloaders = get_augmented_cifar100_test_dataloader(self.dataset_path, self.test_kwargs)
        else:
            raise ValueError(
                f"Trying to measure generalization of unknown dataset! Got {self.model_infos[0].dataset}"
            )

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
        self.model.clear_outputs = False

        n_models = len(self.model.models)
        all_results = [{} for _ in range(n_models)]
        for dl in dataloaders:
            trainer.validate(self.model, dl.dataloader)
            final_metrics = self.model.all_metrics
            for i in range(n_models):
                if dl.name in all_results[i].keys():
                    all_results[i][dl.name].update({str(dl.value): final_metrics[i]})
                else:
                    all_results[i][dl.name] = {str(dl.value): final_metrics[i]}

        for i in range(n_models):
            robustness_result = scalarize_robustness(all_results[i])
            robustness_result.update({"specific_values": all_results[i]})
            file_io.save(robustness_result, self.model.infos[i].path_ckpt_root, nc.GNRLZ_OUT_RESULTS)

    def measure_generalization(self):
        """
        Measures the generalization of the model by evaluating it on an augmented version of the test set.

        """
        if self.model_infos[0].dataset == "CIFAR10":
            dataloaders = get_augmented_cifar10_test_dataloader(self.dataset_path, self.test_kwargs)
        elif self.model_infos[0].dataset == "CIFAR100":
            dataloaders = get_augmented_cifar100_test_dataloader(self.dataset_path, self.test_kwargs)
        else:
            raise ValueError(
                f"Trying to measure generalization of unknown dataset! Got {self.model_infos[0].dataset}"
            )

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
        self.model.clear_outputs = False

        n_models = len(self.model.models)
        all_results = [{} for _ in range(n_models)]
        for dl in dataloaders:
            trainer.validate(self.model, dl.dataloader)
            final_metrics = self.model.all_metrics
            for i in range(n_models):
                if dl.name in all_results[i].keys():
                    all_results[i][dl.name].update({str(dl.value): final_metrics[i]})
                else:
                    all_results[i][dl.name] = {str(dl.value): final_metrics[i]}

        for i in range(n_models):
            robustness_result = scalarize_robustness(all_results[i])
            robustness_result.update({"specific_values": all_results[i]})
            file_io.save(robustness_result, self.model.infos[i].path_ckpt_root, nc.GNRLZ_OUT_RESULTS)
