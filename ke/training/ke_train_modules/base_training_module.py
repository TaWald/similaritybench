from __future__ import annotations

from abc import ABC
from abc import abstractmethod
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

import pytorch_lightning as pl
import torch
from ke.arch.abstract_acti_extr import AbsActiExtrArch
from ke.util import data_structs as ds
from ke.util.file_io import save_json
from torch.optim.lr_scheduler import _LRScheduler as LRScheduler  # noqa
from torch.utils.tensorboard.writer import SummaryWriter


class BaseLightningModule(pl.LightningModule, ABC):
    def __init__(
        self,
        model_info: ds.FirstModelInfo,
        save_checkpoints: bool,
        params: ds.Params,
        hparams: dict,
        skip_n_epochs: int | None = None,
        log: bool = True,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.mode_info: ds.FirstModelInfo = model_info
        self.params = params
        self.ke_hparams = hparams
        self.do_log = log

        if self.do_log:
            self.tb_logger_tr: SummaryWriter = SummaryWriter(log_dir=str(model_info.path_train_log / "train"))
            self.tb_logger_val: SummaryWriter = SummaryWriter(log_dir=str(model_info.path_train_log / "val"))

        self.checkpoint_path: Path = model_info.path_ckpt
        self.checkpoint_dir_path: Path = model_info.path_ckpt.parent
        self.save_checkpoints = save_checkpoints
        torch.backends.cudnn.benchmark = True  # noqa
        self.skip_n_epochs: int | None = skip_n_epochs

        # For the final validation epoch we want to aggregate all activation maps and approximations
        # to calculate the metrics in a less noisy manner.
        self.old_intermediate_reps: torch.Tensor | None = None
        self.new_intermediate_reps: torch.Tensor | None = None
        self.old_y_outs: torch.Tensor | None = None
        self.y_transferred_outs: torch.Tensor | None = None
        self.new_y_out: torch.Tensor | None = None
        self.gts: torch.Tensor | None = None
        self.not_too_large = True

        self.clear_outputs = True

        self.max_data_points = 3e8

        self.final_metrics: dict = {}

    def get_new_arch(self) -> AbsActiExtrArch:
        """Returns the to be trained (new) model"""
        return self.net.get_new_model()

    def on_fit_end(self) -> None:
        """
        Writes the last validation metrics and closes summary writers.
        """
        serializable_metrics = deepcopy(self.final_metrics)
        for key, val in self.final_metrics.items():
            if isinstance(val, dict):
                for k, v in val.items():
                    if isinstance(v, (dict, list)):
                        continue
                    else:
                        serializable_metrics[key + "/" + k] = float(v)
            else:
                serializable_metrics[key] = val

        # Create the final metrics instead here!
        if self.do_log:
            self.tb_logger_tr.add_hparams(self.ke_hparams, asdict(self.params))
            self.tb_logger_val.add_hparams(self.ke_hparams, asdict(self.params))
            save_json(self.final_metrics, self.mode_info.path_last_metrics_json)
            self.tb_logger_tr.close()
            self.tb_logger_val.close()

    def forward(self, x):
        return self.net.forward(x)

    def log_message(self, tensorboard_dict: dict, is_train: bool):
        if self.do_log:
            if is_train:
                sm_wr = self.tb_logger_tr
            else:
                sm_wr = self.tb_logger_val
            for key, val in tensorboard_dict.items():
                if isinstance(val, dict):
                    for k, v in val.items():
                        if isinstance(v, (dict, list)):
                            continue
                        else:
                            sm_wr.add_scalar(key + "/" + k, scalar_value=v, global_step=self.global_step)
                else:
                    sm_wr.add_scalar(key, scalar_value=val, global_step=self.global_step)

    @abstractmethod
    def save_checkpoint(self):
        """Save the checkpoint and (optionally) save the checkpoint of the transfer layers as well."""

    @abstractmethod
    def load_latest_checkpoint(self):
        """Loads the latest checkpoint (only of the to be trained architecture)"""

    @abstractmethod
    def training_step(self, batch, batch_idx):
        """
        Does the training step here
        """

    @abstractmethod
    def training_epoch_end(self, outputs: list[dict]):
        """Does training epoch end stuff"""

    def on_validation_start(self) -> None:
        """
        Empty potential remaining results from before.
        """
        self.old_intermediate_reps = None
        self.new_intermediate_reps = None
        self.old_y_outs = None
        self.y_transferred_outs = None
        self.new_y_out = None
        self.gts = None
        self.not_too_large = True

    def on_validation_end(self) -> None:
        """
        Empty potential remaining results from before.
        """
        if self.clear_outputs:
            self.old_intermediate_reps = None
            self.new_intermediate_reps = None
            self.old_y_outs = None
            self.y_transferred_outs = None
            self.new_y_out = None
            self.gts = None
            self.not_too_large = True

    def get_outputs(self) -> dict[str, torch.Tensor]:
        return {"outputs": self.new_y_out.detach().cpu().numpy(), "groundtruths": self.gts.detach().cpu().numpy()}

    def is_not_too_large(self, t: list[torch.Tensor]) -> bool:
        num_ele = sum([torch.numel(te) for te in t])
        if self.max_data_points >= num_ele:
            return True
        else:
            return False

    def save_validation_values(
        self,
        old_intermediate_reps: list[torch.Tensor] | None,
        new_intermediate_reps: list[torch.Tensor] | None,
        groundtruths: torch.Tensor | None,
        old_y_hats: torch.Tensor | None,
        new_y_hat: torch.Tensor | None,
        transferred_y_hats: torch.Tensor | None,
    ):
        # Save Groundtruths:
        if groundtruths is not None:
            if self.gts is None:
                self.gts = groundtruths
            else:
                self.gts = torch.concatenate([self.gts, groundtruths], dim=0)

        # Aggregate new models outputs
        if new_y_hat is not None:
            detached_y_hat = new_y_hat.detach()
            if self.new_y_out is None:
                self.new_y_out = detached_y_hat
            else:
                self.new_y_out = torch.cat([self.new_y_out, detached_y_hat], dim=0)

        # Aggregate old models outputs
        if old_y_hats is not None:
            detached_y_hats = old_y_hats.detach()
            if self.old_y_outs is None:
                self.old_y_outs = detached_y_hats
            else:
                self.old_y_outs = torch.cat([self.old_y_outs, detached_y_hats], dim=1)

        if transferred_y_hats is not None:
            detached_y_trans_hats = transferred_y_hats.detach()
            if self.y_transferred_outs is None:
                self.y_transferred_outs = detached_y_trans_hats
            else:
                self.y_transferred_outs = torch.cat([self.y_transferred_outs, detached_y_trans_hats], dim=1)

        # Aggregate new and old models intermediate layers
        if (old_intermediate_reps is not None) and (new_intermediate_reps is not None):
            detached_approx = [a.detach() for a in old_intermediate_reps]
            if self.old_intermediate_reps is not None:
                self.not_too_large = self.is_not_too_large(self.old_intermediate_reps)

            if self.old_intermediate_reps is None:
                self.old_intermediate_reps = detached_approx
            else:
                if self.not_too_large:
                    self.old_intermediate_reps = [
                        torch.cat([saved_a, a], dim=1)
                        for saved_a, a in zip(self.old_intermediate_reps, detached_approx)
                    ]

            detached_true = [tr.detach() for tr in new_intermediate_reps]
            if self.new_intermediate_reps is None:
                self.new_intermediate_reps = detached_true
            else:
                if self.not_too_large:
                    self.new_intermediate_reps = [
                        torch.cat([saved_tr, tr], dim=1)
                        for saved_tr, tr in zip(self.new_intermediate_reps, detached_true)
                    ]

    @abstractmethod
    def validation_step(self, batch, batch_idx):
        pass

    @abstractmethod
    def validation_epoch_end(self, outputs):
        pass

    def on_train_epoch_start(self) -> None:
        self.on_validation_epoch_start()

    def on_validation_epoch_start(self) -> None:
        self.old_intermediate_reps = None
        self.new_intermediate_reps = None
        self.old_y_outs = None
        self.y_transferred_outs = None
        self.new_y_out = None
        self.gts = None
        self.final_metrics = {}  # Make sure this doesn't contain something from validation
        self.not_too_large = True

    def on_test_start(self) -> None:
        """
        Empty potential remaining results form before.
        """
        self.on_validation_epoch_start()

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def test_epoch_end(self, outputs):
        self.validation_epoch_end(outputs)

    def configure_optimizers(self):
        has_get_parameters = hasattr(self.net, "get_trainable_parameters")
        if has_get_parameters:
            parameters_to_train = self.net.get_trainable_parameters()
        else:
            parameters_to_train = self.net.parameters()

        optim = torch.optim.SGD(
            params=parameters_to_train,
            lr=self.params.learning_rate,
            momentum=self.params.momentum,
            weight_decay=self.params.weight_decay,
            nesterov=self.params.nesterov,
        )
        if self.params.cosine_annealing:
            total_epochs = self.params.num_epochs
            # This is ugly AF, I know. But weigh rewinding with CosineLR is a pain.
            #  Simulates normal # of epochs as previously to the scheduler.
            #   Skips n epochs to adjust the LR to the value and have the slope be equal
            if self.skip_n_epochs is not None:
                total_epochs = total_epochs + self.skip_n_epochs
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=total_epochs, eta_min=0)
            if self.skip_n_epochs is not None:
                for i in range(self.skip_n_epochs):
                    self.scheduler.step()
            return [optim], [scheduler]
        else:
            return [optim]
