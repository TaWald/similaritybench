from __future__ import annotations

from dataclasses import asdict

import torch
from ke.arch.ke_architectures.unusable_downstream_gradient_reversal import (
    FeatureTransferUnuseableDownstreamArch,
)
from ke.metrics.ke_metrics import multi_output_metrics
from ke.training.ke_train_modules.base_training_module import BaseLightningModule
from ke.util import data_structs
from ke.util import name_conventions as nc
from ke.util.data_structs import BasicTrainingInfo
from ke.util.data_structs import KEUnuseableDownstreamTrainingInfo
from ke.util.file_io import save_json
from torch import nn
from torch.optim.lr_scheduler import _LRScheduler as LRScheduler  # noqa


class KEUnusableDownstreamLightningModule(BaseLightningModule):
    def __init__(
        self,
        model_info: KEUnuseableDownstreamTrainingInfo | BasicTrainingInfo,
        network: FeatureTransferUnuseableDownstreamArch,
        save_checkpoints: bool,
        params: data_structs.Params,
        hparams: dict,
        ce_loss_weight: float,
        trans_loss_weight: float,
        skip_n_epochs: int | None = None,
        log: bool = True,
        save_approx: bool = False,
    ):
        super().__init__(model_info, save_checkpoints, params, hparams, skip_n_epochs, log)
        self.ce_loss_weight: float = ce_loss_weight
        self.trans_loss_weight: float = trans_loss_weight
        self.net: FeatureTransferUnuseableDownstreamArch = network

        self.save_approx = save_approx
        self.ce_loss = nn.CrossEntropyLoss()

    def save_checkpoint(self):
        if self.save_checkpoints:
            if self.current_epoch != 0:
                state_dict = self.net.get_state_dict()
                torch.save(state_dict, self.checkpoint_path)
                if self.save_approx:
                    approx_state_dict: list[tuple[dict, dict]] = self.net.get_approx_state_dict()
                    for cnt, asd in enumerate(approx_state_dict):
                        info = self.checkpoint_dir_path / nc.APPROX_CKPT_INFO_NAME.format(asd[0]["count"])
                        ckpt = self.checkpoint_dir_path / nc.APPROX_CKPT_NAME.format(asd[0]["count"])
                        save_json(asd[0], info)
                        torch.save(asd[1], ckpt)
        return

    def load_latest_checkpoint(self):
        """Loads the latest checkpoint (only of the to be trained architecture)"""
        ckpt = torch.load(self.checkpoint_path)
        self.net.get_new_model().load_state_dict(ckpt)

    def calculate_loss(
        self, new_out: torch.Tensor, old_transferred_outs: list[torch.Tensor] | None, y: torch.Tensor
    ) -> dict:
        new_ce_loss = self.ce_loss(new_out, y)
        trans_ce_loss = torch.mean(torch.stack([self.ce_loss(o, y) for o in old_transferred_outs]))

        class_weighted = new_ce_loss * self.ce_loss_weight
        trans_weighted_ce = trans_ce_loss * self.trans_loss_weight
        loss = class_weighted + trans_weighted_ce
        output = {
            "loss": loss,
            "loss_info": {
                "total_loss": loss,
                "classificiation_weighted": class_weighted,
                "transfer_weighted": trans_weighted_ce,
            },
        }
        return output

    def training_step(self, batch, batch_idx):
        x, y = batch
        new_out, old_transferred_outs, old_original_outs = self(x)
        output = self.calculate_loss(new_out, old_transferred_outs, y)
        return output

    def training_epoch_end(self, outputs: list[dict]):
        # Scheduler steps are done automatically! (No step is needed)
        with torch.no_grad():
            mean_loss = torch.mean(torch.stack([o["loss"] for o in outputs]))
            tensorboard_log = {"tr/loss": mean_loss}
            self.log_message(tensorboard_log, is_train=True)
        return None

    def validation_step(self, batch, batch_idx):
        x, y = batch  # ["data"], batch["label"]
        with torch.no_grad():
            new_out, old_transferred_outs, old_original_outs = self(x)
            self.save_validation_values(
                old_intermediate_reps=None,
                new_intermediate_reps=None,
                groundtruths=y,
                old_y_hats=torch.stack(old_original_outs),
                new_y_hat=new_out,
                transferred_y_hats=torch.stack(old_transferred_outs),
            )

    def validation_epoch_end(self, outputs):
        with torch.no_grad():
            self.save_checkpoint()
            loss_values = self.calculate_loss(self.new_y_out, self.y_transferred_outs, self.gts)
            out_metrics = multi_output_metrics(self.new_y_out, self.old_y_outs, self.gts)
            trans_metrics = multi_output_metrics(self.new_y_out, self.y_transferred_outs, self.gts)

            metrics = asdict(out_metrics)
            metrics["mean_transfer_accuracy"] = trans_metrics.mean_old_acc
            self.final_metrics = metrics

            tensorboard_dict = {f"metrics/{k}": v for k, v in metrics.items()}
            tensorboard_dict.update({f"loss/{k}": v for k, v in loss_values["loss_info"].items()})
            self.log_message(tensorboard_dict, is_train=False)

            prog_bar_log = {}
            for k, v in metrics.items():
                if k in ["mean_transfer_accuracy", "accuracy", "cohens_kappa", "mean_old_acc"]:
                    prog_bar_log[f"val/{k}"] = v
            self.log_dict(prog_bar_log, prog_bar=True, logger=False)
        return None
