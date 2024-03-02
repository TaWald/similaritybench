from __future__ import annotations

import os
from abc import abstractmethod
from dataclasses import asdict
from pathlib import Path

from torch.utils.data import DataLoader
from vision.data.base_datamodule import BaseDataModule
from vision.training.ke_train_modules.base_training_module import BaseLightningModule
from vision.util import data_structs as ds
from vision.util import file_io
from vision.util import name_conventions as nc
from vision.util.gpu_cluster_worker_nodes import get_workers_for_current_node
from pytorch_lightning import Trainer


class ShortcutTrainer:
    def __init__(
        self,
        model: BaseLightningModule,
        datamodule: BaseDataModule,
        no_sc_datamodule: BaseDataModule,  # Shortcut datamodule with no shortcut
        full_sc_datamodule: BaseDataModule,  # Shortcut datamodule with full shortcut
        model_info: ds.ModelInfo,
        arch_params: dict,
    ):
        self.model: BaseLightningModule = model
        self.datamodule: BaseDataModule = datamodule
        self.no_sc_datamodule: BaseDataModule = no_sc_datamodule  # 0 % correlation
        self.full_sc_datamodule: BaseDataModule = full_sc_datamodule  # 100 % correlation
        self.params = model.params
        self.arch_params = arch_params
        self.model_info = model_info
        self.num_workers = get_workers_for_current_node()
        self.prog_bar = False if "data" in os.environ else True

        # Create them. Should not exist though or overwrite would happen!
        self.model_info.path_root.mkdir(exist_ok=True, parents=True)
        self.model_info.path_activations.mkdir(exist_ok=True, parents=True)
        self.model_info.path_ckpt.parent.mkdir(exist_ok=True, parents=True)

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
            default_root_dir=str(self.model_info.path_root),
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
        self.model.clear_outputs = False
        trainer.validate(
            self.model,
            dataloaders=self.datamodule.val_dataloader(
                self.params.split, transform=ds.Augmentation.VAL, **self.val_kwargs
            ),
        )
        val_metrics_in_domain = self.model.final_metrics

        trainer.validate(
            self.model,
            dataloaders=self.no_sc_datamodule.val_dataloader(
                self.params.split, transform=ds.Augmentation.VAL, **self.val_kwargs
            ),
        )
        val_metrics_no_sc = self.model.final_metrics

        trainer.validate(
            self.model,
            dataloaders=self.full_sc_datamodule.val_dataloader(
                self.params.split, transform=ds.Augmentation.VAL, **self.val_kwargs
            ),
        )
        val_metrics_full_sc = self.model.final_metrics

        diff_full_sc_to_in_domain = val_metrics_full_sc["accuracy"] - val_metrics_in_domain["accuracy"]
        diff_no_sc_to_in_domain = val_metrics_no_sc["accuracy"] - val_metrics_in_domain["accuracy"]

        trainer.test(self.model, dataloaders=self.datamodule.test_dataloader(ds.Augmentation.VAL, **self.val_kwargs))
        test_metrics_in_domain = self.model.final_metrics
        trainer.test(
            self.model, dataloaders=self.no_sc_datamodule.test_dataloader(ds.Augmentation.VAL, **self.val_kwargs)
        )
        test_metrics_no_sc = self.model.final_metrics
        trainer.test(
            self.model, dataloaders=self.full_sc_datamodule.test_dataloader(ds.Augmentation.VAL, **self.val_kwargs)
        )
        test_metrics_full_sc = self.model.final_metrics

        diff_full_sc_to_in_domain_test = test_metrics_full_sc["accuracy"] - test_metrics_in_domain["accuracy"]
        diff_no_sc_to_in_domain_test = test_metrics_no_sc["accuracy"] - test_metrics_in_domain["accuracy"]

        output = {
            "val_changes": {
                "full_sc_to_in_domain": diff_full_sc_to_in_domain,
                "no_sc_to_in_domain": diff_no_sc_to_in_domain,
            },
            "test_changes": {
                "full_sc_to_in_domain": diff_full_sc_to_in_domain_test,
                "no_sc_to_in_domain": diff_no_sc_to_in_domain_test,
            },
            "val": val_metrics_in_domain,
            "test": test_metrics_in_domain,
            "val_no_sc": val_metrics_no_sc,
            "test_no_sc": test_metrics_no_sc,
            "val_full_sc": val_metrics_full_sc,
            "test_full_sc": test_metrics_full_sc,
            **vars(self.params),
            **self.arch_params,
        }

        file_io.save(
            output,
            path=self.model_info.path_root,
            filename=nc.OUTPUT_TMPLT,
        )

        tbt_ke_dict = {}
        for k, v in asdict(self.model_info).items():
            if isinstance(v, Path):
                tbt_ke_dict[k] = str(v)
            else:
                tbt_ke_dict[k] = v
        file_io.save_json(tbt_ke_dict, self.model_info.path_train_info_json)

    def evaluate(self, eval_datamodule: BaseDataModule, mode: str):
        trainer = Trainer(
            enable_checkpointing=False,
            max_epochs=self.params.num_epochs,
            accelerator="gpu",
            devices=1,
            precision=16,
            default_root_dir=str(self.model_info.path_root),
            enable_progress_bar=self.prog_bar,
            logger=False,
            profiler=None,
        )
        self.model.cuda()
        self.model.eval()
        self.model.clear_outputs = False
        if mode == "val":
            dataloader = eval_datamodule.val_dataloader
        elif mode == "test":
            dataloader = eval_datamodule.test_dataloader
        else:
            raise ValueError()

        trainer.validate(
            self.model,
            dataloaders=dataloader(self.params.split, transform=ds.Augmentation.VAL, **self.val_kwargs),
        )
        val_metrics = self.model.final_metrics
        output = {
            "val": val_metrics,
            **vars(self.params),
            **self.arch_params,
        }
        return output

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
