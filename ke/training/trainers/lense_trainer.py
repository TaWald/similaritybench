from __future__ import annotations

from abc import abstractmethod
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from ke.training.ke_train_modules.AdversarialLenseLightningModule import AdversarialLenseLightningModule
from ke.training.trainers.base_trainer import BaseTrainer
from ke.util import data_structs as ds
from ke.util import file_io
from ke.util import name_conventions as nc
from PIL import Image
from pytorch_lightning import Trainer


class LenseTrainer(BaseTrainer):
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
        self.training_info: ds.KEAdversarialLenseOutputTrainingInfo
        self.model: AdversarialLenseLightningModule
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
        created_samples = self.model.batch_of_created_samples
        original_samples = self.model.batch_of_original_samples
        self.training_info.path_lense_examples.mkdir(exist_ok=True)
        human_interpretable_created = self.datamodule.revert_base_transform(created_samples)[:20]
        human_interpretable_original = self.datamodule.revert_base_transform(original_samples)[:20]
        # make sure its between 0 and 255
        diff = (human_interpretable_original - human_interpretable_created + 255.0) / 2
        for cnt, (hio, hic, dif) in enumerate(zip(human_interpretable_original, human_interpretable_created, diff)):
            created = Image.fromarray(hic.astype(np.uint8).transpose(1, 2, 0))
            orig = Image.fromarray(hio.astype(np.uint8).transpose(1, 2, 0))
            d = Image.fromarray(dif.astype(np.uint8).transpose(1, 2, 0))
            created.save(str(self.training_info.path_lense_examples / f"lense_{cnt:04}.png"))
            orig.save(str(self.training_info.path_lense_examples / f"orig_{cnt:04}.png"))
            d.save(str(self.training_info.path_lense_examples / f"diff_{cnt:04}.png"))

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
            filename=nc.LENSE_TMPLT,
        )
        file_io.save(
            output,
            path=self.training_info.path_data_root,
            filename=nc.LENSE_TMPLT,
        )

        tbt_ke_dict = {}
        for k, v in asdict(self.training_info).items():
            if isinstance(v, Path):
                tbt_ke_dict[k] = str(v)
            else:
                tbt_ke_dict[k] = v
        file_io.save(tbt_ke_dict, self.training_info.path_data_root, filename=nc.LENSE_INFO)

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
