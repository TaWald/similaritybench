from __future__ import annotations

import re
from dataclasses import asdict
from typing import Any
from warnings import warn

import pytorch_lightning as pl
import torch
from ke.arch.ke_architectures.parallel_model_arch import ParallelModelArch
from ke.losses.dummy_loss import DummyLoss
from ke.losses.parallel_output_loss import AbstractParallelOutputLoss
from ke.metrics.ke_metrics import parallel_multi_output_metrics
from ke.util import data_structs as ds
from ke.util import name_conventions as nc
from torch.optim.lr_scheduler import _LRScheduler as LRScheduler  # noqa


class ParallelEnsembleLM(pl.LightningModule):
    def __init__(
        self,
        model_infos: list[ds.ParallelModelInfo],
        network: ParallelModelArch,
        params: ds.Params,
        hparams: dict,
        loss: AbstractParallelOutputLoss | DummyLoss,
        log: bool = True,
    ):
        super().__init__()
        self.model_infos = model_infos
        self.net: ParallelModelArch = network
        self.loss: AbstractParallelOutputLoss = loss
        self.params = params

        self.final_metrics = None
        self.gts = None
        self.logits = None
        self.save_checkpoints = params.save_last_checkpoint

    def save_checkpoint(self):
        """Saves the checkpoint of all models of the current epoch"""
        if (self.current_epoch == 0) or (not self.save_checkpoints):
            return  # Skip if first epoch
        for cnt, m in enumerate(self.net.all_models):
            state_dict = m.state_dict()
            torch.save(state_dict, self.model_infos[0].path_ckpt / nc.CKPT_PARALLEL_TMPLT.format(cnt))
        return

    def remove_cached_values(self):
        self.gts = None
        self.logits = None

    def load_latest_checkpoint(self):
        """Loads the latest checkpoint (only of the to be trained architecture)"""
        found_ckpts = []
        for content in self.checkpoint_path.iterdir():
            if re.match(nc.CKPT_PARALLEL_RE, content.name):
                found_ckpts.append(content)
        warn(f"Found: {found_ckpts} Checkpoints. Loading all.")
        for cnt, ckpt in enumerate(found_ckpts):
            state_dict = torch.load(ckpt)
            self.net.modules[cnt].load_state_dict(state_dict)

    def save_validation_values(self, logits: torch.Tensor | None, targets: torch.Tensor | None):
        # Save Groundtruths:
        if targets is not None:
            if self.gts is None:
                self.gts = targets
            else:
                self.gts = torch.concatenate([self.gts, targets], dim=0)

        # Aggregate new models outputs
        if logits is not None:
            logits = logits.detach()
            if self.logits is None:
                self.logits = logits
            else:
                self.logits = torch.cat([self.logits, logits], dim=1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.net(x)
        ke_fwd = self.loss.forward(target=y, logits=logits)
        return ke_fwd

    def validation_step(self, batch, batch_idx):
        x, y = batch
        with torch.no_grad():
            logits = self.net(x)
            self.save_validation_values(logits=logits, targets=y)
            return self.loss.forward(target=y, logits=logits)

    def validation_epoch_end(self, outputs):
        with torch.no_grad():
            self.save_checkpoint()
            out_metrics = parallel_multi_output_metrics(outputs=self.logits, groundtruth=self.gts)
            self.final_metrics = asdict(out_metrics)  # Join dicts
            self.remove_cached_values()
            prog_bar_log = {
                "val/mean_acc": out_metrics.mean_accuracy,
                "val/ens_acc": out_metrics.ensemble_accuracy,
                "val/CoKa": out_metrics.cohens_kappa.last_to_others_mean,
            }
            self.log_dict(prog_bar_log, prog_bar=True, logger=False)
        return None

    def on_test_start(self) -> None:
        # Make sure the final metrics are empty and we don't mess up.
        self.final_metrics = None

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def test_epoch_end(self, outputs) -> None:
        return self.validation_epoch_end(outputs)

    def configure_optimizers(self) -> Any:
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
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=total_epochs, eta_min=1e-6)
            return [optim], [scheduler]
        else:
            return [optim]
