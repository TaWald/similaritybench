from __future__ import annotations

import os
from abc import abstractmethod
from dataclasses import asdict
from pathlib import Path
from warnings import warn

import torch
from augmented_cifar.scripts.get_dataloaders import get_augmented_cifar100_test_dataloader
from augmented_cifar.scripts.get_dataloaders import get_augmented_cifar10_test_dataloader
from ke.data.base_datamodule import BaseDataModule
from ke.training.ke_train_modules.base_training_module import BaseLightningModule
from ke.util import data_structs as ds
from ke.util import file_io
from ke.util import name_conventions as nc
from ke.util.gpu_cluster_worker_nodes import get_workers_for_current_node
from pytorch_lightning import Trainer


class BaseTrainer:
    def __init__(
        self,
        model: BaseLightningModule,
        datamodule: BaseDataModule,
        params: ds.Params,
        basic_training_info: ds.FirstModelInfo,
        arch_params: dict,
    ):
        self.model: BaseLightningModule = model
        self.datamodule: BaseDataModule = datamodule
        self.params = params
        self.arch_params = arch_params
        self.training_info = basic_training_info
        self.basic_training_info = basic_training_info
        self.num_workers = get_workers_for_current_node()
        self.prog_bar = False if "data" in os.environ else True

        # Create them. Should not exist though or overwrite would happen!
        self.basic_training_info.path_ckpt_root.mkdir(exist_ok=True, parents=True)
        self.basic_training_info.path_activations.mkdir(exist_ok=True, parents=True)
        self.basic_training_info.path_ckpt.parent.mkdir(exist_ok=True, parents=True)

        if "RAW_DATA" in os.environ:
            dataset_path = os.environ["RAW_DATA"]
        elif "data" in os.environ:
            dataset_path = os.environ["data"]
        else:
            raise EnvironmentError

        self.dataset_path = dataset_path

        self.train_kwargs = {
            "shuffle": True,
            "drop_last": False,
            "pin_memory": True,
            "batch_size": self.params.batch_size,
            "num_workers": self.num_workers,
            "persistent_workers": True,
        }
        self.val_kwargs = {
            "shuffle": False,
            "drop_last": False,
            "pin_memory": True,
            "batch_size": self.params.batch_size,
            "num_workers": self.num_workers,
            "persistent_workers": True,
        }
        self.test_kwargs = {
            "shuffle": False,
            "drop_last": False,
            "pin_memory": True,
            "batch_size": self.params.batch_size,
            "num_workers": self.num_workers,
            "persistent_workers": True,
        }

    def post_train_eval(self):
        """
        Function if potentially a model finished training but did not write output.json or info.json accordingly.
        Intended to only do the final eval with the given model and save it.
        """
        trainer = Trainer(
            enable_checkpointing=False,
            max_epochs=self.params.num_epochs,
            accelerator="gpu",
            devices=1,
            precision=16,
            default_root_dir=str(self.basic_training_info.path_data_root),
            enable_progress_bar=self.prog_bar,
            logger=False,
            profiler=None,
        )

        # Points to final checkpoint.
        self.model.load_latest_checkpoint()

        self.model.cuda()
        self.model.eval()
        self.model.final_validation = True
        trainer.validate(
            self.model,
            dataloaders=self.datamodule.val_dataloader(
                self.params.split, transform=ds.Augmentation.VAL, **self.val_kwargs
            ),
        )
        val_metrics = self.model.final_metrics
        trainer.test(self.model, dataloaders=self.datamodule.test_dataloader(ds.Augmentation.VAL, **self.val_kwargs))
        test_metrics = self.model.final_metrics
        output = {
            "val": val_metrics,
            "test": test_metrics,
            **vars(self.params),
            **self.arch_params,
        }

        file_io.save(
            output,
            path=self.training_info.path_ckpt_root,
            filename=nc.OUTPUT_TMPLT,
        )
        file_io.save(
            output,
            path=self.training_info.path_data_root,
            filename=nc.OUTPUT_TMPLT,
        )

        tbt_ke_dict = {}
        for k, v in asdict(self.training_info).items():
            if isinstance(v, Path):
                tbt_ke_dict[k] = str(v)
            else:
                tbt_ke_dict[k] = v
        file_io.save_json(tbt_ke_dict, self.training_info.path_train_info_json)

    def train(self):
        """Trains a model and keeps it as attribute self.model
         After finishing training saves checkpoint and a short Hyperparameter summary
         to the model directory.

        :return:
        """
        trainer = Trainer(
            enable_checkpointing=False,
            max_epochs=self.params.num_epochs,
            accelerator="gpu",
            devices=1,
            precision=16,
            default_root_dir=str(self.basic_training_info.path_data_root),
            enable_progress_bar=self.prog_bar,
            logger=False,
            profiler=None,
        )
        self.model.cuda()
        trainer.fit(
            self.model,
            train_dataloaders=self.datamodule.train_dataloader(
                split=self.params.split,
                transform=ds.Augmentation.TRAIN,
                **self.train_kwargs,
            ),
            val_dataloaders=self.datamodule.val_dataloader(
                split=self.params.split,
                transform=ds.Augmentation.VAL,
                **self.val_kwargs,
            ),
        )
        self.model.cuda()
        self.model.eval()
        self.model.final_validation = True
        trainer.validate(
            self.model,
            dataloaders=self.datamodule.val_dataloader(
                self.params.split, transform=ds.Augmentation.VAL, **self.val_kwargs
            ),
        )
        val_metrics = self.model.final_metrics
        trainer.test(self.model, dataloaders=self.datamodule.test_dataloader(ds.Augmentation.VAL, **self.val_kwargs))
        test_metrics = self.model.final_metrics
        output = {
            "val": val_metrics,
            "test": test_metrics,
            **vars(self.params),
            **self.arch_params,
        }

        file_io.save(
            output,
            path=self.training_info.path_ckpt_root,
            filename=nc.OUTPUT_TMPLT,
        )
        file_io.save(
            output,
            path=self.training_info.path_data_root,
            filename=nc.OUTPUT_TMPLT,
        )

        tbt_ke_dict = {}
        for k, v in asdict(self.training_info).items():
            if isinstance(v, Path):
                tbt_ke_dict[k] = str(v)
            else:
                tbt_ke_dict[k] = v
        file_io.save_json(tbt_ke_dict, self.training_info.path_train_info_json)

    # ToDo: Outsource this call to
    def measure_generalization(self):
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
        self.model.load_latest_checkpoint()
        self.model.cuda()
        self.model.eval()
        self.model.clear_outputs = False
        if self.params.dataset == "CIFAR10":
            dataloaders = get_augmented_cifar10_test_dataloader(self.dataset_path, self.test_kwargs)
        elif self.params.dataset == "CIFAR100":
            dataloaders = get_augmented_cifar100_test_dataloader(self.dataset_path, self.test_kwargs)
        else:
            warn(f"Trying to measure generalization of unknown dataset! Got {self.params.dataset}")
            return

        all_results = {}
        for dl in dataloaders:
            trainer.validate(self.model, dl.dataloader)
            out = self.model.get_outputs()
            final_metrics = self.model.final_metrics
            if dl.name in all_results.keys():
                all_results[dl.name].update({str(dl.value): final_metrics})
            else:
                all_results[dl.name] = {str(dl.value): final_metrics}
            file_io.save(
                out["outputs"], self.training_info.path_activations, nc.GNRLZ_PD_TMPLT.format(dl.name, dl.value)
            )
            file_io.save(
                out["groundtruths"], self.training_info.path_activations, nc.GNRLZ_GT_TMPLT.format(dl.name, dl.value)
            )
        file_io.save(all_results, self.training_info.path_ckpt_root, nc.GNRLZ_OUT_RESULTS)
        self.model.clear_outputs = True

    def save_outputs(self, mode: str):
        assert mode in ["test", "val"], f"Expected only 'test' or 'val' as mode. Got: {mode}"

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
        self.model.load_latest_checkpoint()
        self.model.cuda()
        self.model.eval()
        self.model.clear_outputs = False
        if mode == "test":
            trainer.validate(
                self.model, self.datamodule.test_dataloader(**self.test_kwargs, transform=ds.Augmentation.VAL)
            )
        else:
            trainer.validate(
                self.model, self.datamodule.val_dataloader(**self.test_kwargs, transform=ds.Augmentation.VAL)
            )
        out = self.model.get_outputs()
        self.model.clear_outputs = True

        if mode == "test":
            file_io.save(out["outputs"], self.training_info.path_activations / nc.MODEL_TEST_PD_TMPLT)
            file_io.save(out["groundtruths"], self.training_info.path_activations / nc.MODEL_TEST_GT_TMPLT)
        else:
            file_io.save(out["outputs"], self.training_info.path_activations / nc.MODEL_VAL_PD_TMPLT)
            file_io.save(out["groundtruths"], self.training_info.path_activations / nc.MODEL_VAL_GT_TMPLT)

        return out

    def save_activations(self, mode="test"):
        """
        Method for saving intermediate feature map activations.
        Can be called when the model is of the right class. If not it raises a NotImplementedError
        """
        assert mode in ["test", "val"], f"Expected only 'test' or 'val' as mode. Got: {mode}"
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
        self.model.load_latest_checkpoint()
        self.model.cuda()
        self.model.eval()

        new_model = self.model.net.get_new_model()
        for h in new_model.hooks:
            new_model.register_rep_hook(h)
            trainer.validate(
                self.model, self.datamodule.test_dataloader(**self.test_kwargs, transform=ds.Augmentation.VAL)
            )
            acti = new_model.activations
            if mode == "test":
                file_io.save(acti, nc.TEST_ACTI_TMPLT.format(h.name))
            else:
                file_io.save(acti, nc.VAL_ACTI_TMPLT.format(h.name))
        return

    @abstractmethod
    def calculate_acc(self, mode="test"):
        assert mode in ["test", "val"], f"Expected only 'test' or 'val' as mode. Got: {mode}"

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
        self.model.load_latest_checkpoint()
        self.model.cuda()
        self.model.eval()
        self.model.clear_outputs = False
        if mode == "test":
            trainer.validate(self.model, self.datamodule.test_dataloader(**self.test_kwargs))
        else:
            trainer.validate(self.model, self.datamodule.val_dataloader(**self.test_kwargs))
        out = self.model.get_outputs()
        self.model.clear_outputs = True

        acc = torch.mean(torch.argmax(out["outputs"], dim=-1) == out["groundtruths"], dtype=torch.float)
        return acc
