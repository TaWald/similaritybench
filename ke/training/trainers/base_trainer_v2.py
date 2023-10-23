from __future__ import annotations

import os
from dataclasses import asdict
from pathlib import Path

from ke.data.base_datamodule import BaseDataModule
from ke.training.ke_train_modules import parallel_ensemble
from ke.training.ke_train_modules.base_training_module import BaseLightningModule
from ke.util import data_structs as ds
from ke.util import file_io
from ke.util import name_conventions as nc
from ke.util.gpu_cluster_worker_nodes import get_workers_for_current_node
from pytorch_lightning import Trainer


class BaseTrainerV2:
    def __init__(
        self,
        model: BaseLightningModule,
        datamodule: BaseDataModule,
        params: ds.Params,
        root_dir: Path,
        arch_params: dict,
    ):
        self.model: parallel_ensemble = model
        self.datamodule: BaseDataModule = datamodule
        self.params = params
        self.arch_params = arch_params
        self.root_dir = root_dir
        self.num_workers = get_workers_for_current_node()
        self.prog_bar = False if "data" in os.environ else True

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

    def write_info(self):
        info_json = asdict(self.params)
        info_json["arch_params"] = self.arch_params
        for k, v in info_json.items():
            if isinstance(v, Path):
                info_json[k] = str(v)
            else:
                info_json[k] = v
        file_io.save_json(info_json, self.root_dir / nc.KE_INFO_FILE)

    def train(self):
        """Trains a model and keeps it as attribute self.model
        After finishing training saves checkpoint and a short Hyperparameter summary
        to the model directory.

        :return:
        """
        self.write_info()
        trainer = Trainer(
            enable_checkpointing=False,
            max_epochs=self.params.num_epochs,
            accelerator="gpu",
            devices=1,
            precision=16,
            default_root_dir=str(self.root_dir),
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
            path=self.root_dir,
            filename=nc.OUTPUT_TMPLT,
        )
