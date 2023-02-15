from __future__ import annotations

from dataclasses import asdict

import torch
from rep_trans.arch.ke_architectures.feature_approximation import FAArch
from rep_trans.arch.ke_architectures.feature_approximation_gradient_reversal import (
    FAGradientReversalArch,
)
from rep_trans.arch.ke_architectures.feature_approximation_subtraction import FASubtractionArch
from rep_trans.losses.ke_adversarial_loss import KEAdversarialTrainLoss
from rep_trans.losses.ke_loss import KETrainLoss
from rep_trans.losses.ke_sub_loss import KESubTrainLoss
from rep_trans.metrics.ke_metrics import multi_output_metrics
from rep_trans.metrics.ke_metrics import representation_metrics
from rep_trans.training.ke_train_modules.base_training_module import BaseLightningModule
from rep_trans.util import data_structs
from rep_trans.util import name_conventions as nc
from rep_trans.util.data_structs import BaseArchitecture
from rep_trans.util.data_structs import Dataset
from rep_trans.util.data_structs import KEAdversarialTrainingInfo
from rep_trans.util.data_structs import KESubTrainingInfo
from rep_trans.util.data_structs import KETrainingInfo
from rep_trans.util.file_io import save_json
from torch.optim.lr_scheduler import _LRScheduler as LRScheduler  # noqa


class IntermediateRepresentationLightningModule(BaseLightningModule):
    def __init__(
        self,
        model_info: KETrainingInfo | KESubTrainingInfo | KEAdversarialTrainingInfo,
        network: FAGradientReversalArch | FAArch | FASubtractionArch,
        save_checkpoints: bool,
        params: data_structs.Params,
        hparams: dict,
        loss: KETrainLoss | KESubTrainLoss | KEAdversarialTrainLoss,
        skip_n_epochs: int | None = None,
        log: bool = True,
        save_approx: bool = False,
    ):
        super().__init__(
            model_info=model_info,
            save_checkpoints=save_checkpoints,
            params=params,
            hparams=hparams,
            skip_n_epochs=skip_n_epochs,
            log=log,
        )
        self.net: FAGradientReversalArch | FAArch | FASubtractionArch = network
        self.loss = loss
        self.save_approx: bool = save_approx

    def save_checkpoint(self):
        if self.current_epoch == 0:
            return
        if self.save_checkpoints:
            if self.current_epoch == (self.params.num_epochs - 1):
                state_dict = self.net.get_new_model_state_dict()
                torch.save(state_dict, self.checkpoint_path)
                if self.save_approx:
                    approx_state_dict: list[tuple[dict, dict]] = self.net.get_approx_state_dict()
                    for cnt, asd in enumerate(approx_state_dict):
                        info = self.checkpoint_dir_path / nc.APPROX_CKPT_INFO_NAME.format(asd[0]["count"])
                        ckpt = self.checkpoint_dir_path / nc.APPROX_CKPT_NAME.format(asd[0]["count"])
                        save_json(asd[0], info)
                        torch.save(asd[1], ckpt)
            # torch.save(transfer_dict, self.checkpoint_path.parent / "transfer_layer.ckpt")
        return

    def load_latest_checkpoint(self):
        """Loads the latest checkpoint (only of the to be trained architecture)"""
        ckpt = torch.load(self.checkpoint_path)
        self.net.new_arch.load_state_dict(ckpt)

    def training_step(self, batch, batch_idx):
        x, y = batch
        old_inters, new_inter, old_outs, new_out = self(x)
        ke_fwd = self.loss.forward(
            label=y,
            new_out=new_out,
            new_inter=new_inter,
            old_inters=old_inters,
            epoch_num=self.current_epoch,
            global_step=self.global_step,
        )
        return ke_fwd

    def training_epoch_end(self, outputs: list[dict]):
        with torch.no_grad():
            loss_values = self.loss.on_epoch_end(outputs)
            prog_bar_log = {"tr/loss": loss_values["loss/total"]}

            self.log_dict(prog_bar_log, prog_bar=True, logger=False)
            self.log_message(loss_values, is_train=True)
        return None

    def validation_step(self, batch, batch_idx):
        x, y = batch  # ["data"], batch["label"]
        with torch.no_grad():
            old_inters, new_inter, old_outs, new_out = self(x)
            self.save_validation_values(
                old_intermediate_reps=old_inters,
                new_intermediate_reps=new_inter,
                groundtruths=y,
                old_y_hats=old_outs,
                new_y_hat=new_out,
                transferred_y_hats=None,
            )
            return self.loss.forward(
                label=y,
                new_out=new_out,
                new_inter=new_inter,
                old_inters=old_inters,
                epoch_num=self.current_epoch,
                global_step=self.global_step,
            )

    def validation_epoch_end(self, outputs):
        with torch.no_grad():
            self.save_checkpoint()
            loss_dict = self.loss.on_epoch_end(outputs)

            rep_metrics = representation_metrics(
                self.new_intermediate_reps,
                self.old_intermediate_reps,
                sample_size=128,
                calc_r2=True,
                calc_corr=True,
                calc_cka=True,
                calc_cos=True,
            )
            out_metrics = multi_output_metrics(
                self.new_y_out,
                self.old_y_outs,
                self.gts,
                Dataset(self.params.dataset),
                BaseArchitecture(self.params.architecture_name),
            )
            self.final_metrics = asdict(rep_metrics) | asdict(out_metrics)  # Join dicts

            tensorboard_dict = {f"metrics/{k}": v for k, v in self.final_metrics.items()}
            tensorboard_dict.update(loss_dict)

            prog_bar_log = {
                "val/acc": out_metrics.accuracy,
                "val/corr": rep_metrics.corr,
                "val/celu_r2": rep_metrics.celu_r2,
                "val/CoKa": out_metrics.cohens_kappa,
                "val/rrCosSim": rep_metrics.rel_rep_cosine_similarity,
            }
            self.log_dict(prog_bar_log, prog_bar=True, logger=False)
            self.log_message(tensorboard_dict, is_train=False)
        return None
