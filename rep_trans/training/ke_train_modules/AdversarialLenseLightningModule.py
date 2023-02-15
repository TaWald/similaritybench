from __future__ import annotations

from typing import Any
from typing import Callable
from typing import Optional
from typing import Union

import numpy as np
import rep_trans.util.name_conventions as nc
import torch
from pytorch_lightning.core.optimizer import LightningOptimizer
from rep_trans.arch.ke_architectures.adversarial_lense_training_arch import AdversarialLenseTrainingArch
from rep_trans.losses.ke_lense_output_loss import KELenseOutputTrainLoss
from rep_trans.metrics.ke_metrics import single_output_metrics
from rep_trans.training.ke_train_modules.base_training_module import BaseLightningModule
from rep_trans.util import data_structs
from rep_trans.util import data_structs as ds
from rep_trans.util.file_io import save_json
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler as LRScheduler  # noqa


class AdversarialLenseLightningModule(BaseLightningModule):
    def __init__(
        self,
        model_info: ds.KEAdversarialLenseOutputTrainingInfo,
        network: AdversarialLenseTrainingArch,
        save_checkpoints: bool,
        params: data_structs.Params,
        hparams: dict,
        loss: KELenseOutputTrainLoss,
        skip_n_epochs: int | None = None,
        log: bool = False,
    ):
        super().__init__(model_info, save_checkpoints, params, hparams, skip_n_epochs, log)
        self.loss: KELenseOutputTrainLoss = loss
        self.net: AdversarialLenseTrainingArch = network  # To overwrite typehints
        self.batch_of_created_samples: np.ndarray | None = None
        self.batch_of_original_samples: np.ndarray | None = None

    def save_checkpoint(self):
        if self.current_epoch == 0:
            return
        if self.save_checkpoints:
            if self.current_epoch == (self.params.num_epochs - 1):
                lense_dict = self.net.get_state_dict()
                torch.save(lense_dict, self.checkpoint_dir_path / nc.KE_LENSE_CKPT_NAME)
        return

    def load_latest_checkpoint(self):
        """Loads the latest checkpoint (only of the to be trained architecture)"""
        ckpt = torch.load(self.checkpoint_dir_path / nc.KE_LENSE_CKPT_NAME)
        self.net.lense.load_state_dict(ckpt)

    def on_fit_end(self) -> None:
        """
        Writes the last validation metrics and closes summary writers.
        """
        serializable_log = {k: float(v) for k, v in self.final_metrics.items()}
        # Create the final metrics instead here!
        if self.do_log:
            self.tb_logger_tr.add_hparams(self.ke_hparams, serializable_log)
            self.tb_logger_val.add_hparams(self.ke_hparams, serializable_log)
            save_json(serializable_log, self.mode_info.path_data_root / "lense.json")
            self.tb_logger_tr.close()
            self.tb_logger_val.close()

    def training_step(self, batch, batch_idx):
        x, y = batch
        x_lense, aug_y_hats = self(x)
        output = self.loss.forward(
            original_image=x, reconstructed_image=x_lense, groundtruth=y, aug_predictions=aug_y_hats
        )
        return output

    def training_epoch_end(self, outputs: list[dict]):
        # Scheduler steps are done automatically! (No step is needed)
        loss_values = self.loss.on_epoch_end(outputs)

        prog_bar_log = {"tr/l_reco": loss_values["loss/reconstruction"], "tr/l_adv": loss_values["loss/adversarial"]}

        self.log_dict(prog_bar_log, prog_bar=True, logger=False)
        self.log_message(loss_values, is_train=True)

        return None

    def validation_step(self, batch, batch_idx):
        x, y = batch  # ["data"], batch["label"]

        with torch.no_grad():
            clean_y_outs = self.net.clean_forward(x)
            x_lense, aug_y_hats = self.net.forward(x)

            self.save_validation_values(
                old_intermediate_reps=None,
                new_intermediate_reps=None,
                groundtruths=y,
                old_y_hats=torch.stack(clean_y_outs),
                new_y_hat=None,
                transferred_y_hats=torch.stack(aug_y_hats),
            )

            output = self.loss.forward(
                original_image=x, reconstructed_image=x_lense, groundtruth=y, aug_predictions=aug_y_hats
            )

            return output

    def validation_epoch_end(self, outputs):
        self.save_checkpoint()
        out = self.loss.on_epoch_end(outputs)

        aug_metrics = single_output_metrics(new_output=self.y_transferred_outs[0, ...], groundtruth=self.gts)
        cln_metrics = single_output_metrics(new_output=self.old_y_outs[0, ...], groundtruth=self.gts)

        out["metrics/augmented_accuracy"] = aug_metrics.accuracy
        out["metrics/clean_accuracy"] = cln_metrics.accuracy
        self.final_metrics = {k: float(v) for k, v in out.items()}

        prog_bar_log = {
            "val/aug_acc": aug_metrics.accuracy,
            "val/clean_acc": cln_metrics.accuracy,
        }
        self.log_dict(prog_bar_log, prog_bar=True, logger=False)
        self.log_message(out, is_train=False)
        return None

    def test_step(self, batch, batch_idx):
        x, y = batch  # ["data"], batch["label"]

        with torch.no_grad():
            clean_y_outs = self.net.clean_forward(x)
            x_lense, aug_y_hats = self.net.forward(x)
            if batch_idx == 0:
                self.batch_of_created_samples = x_lense.detach().cpu().numpy()
                self.batch_of_original_samples = x.detach().cpu().numpy()

            self.save_validation_values(
                old_intermediate_reps=None,
                new_intermediate_reps=None,
                groundtruths=y,
                old_y_hats=torch.stack(clean_y_outs),
                new_y_hat=None,
                transferred_y_hats=torch.stack(aug_y_hats),
            )

            output = self.loss.forward(
                original_image=x, reconstructed_image=x_lense, groundtruth=y, aug_predictions=aug_y_hats
            )

            return output

    def optimizer_step(
        self,
        epoch: int,
        batch_idx: int,
        optimizer: Union[Optimizer, LightningOptimizer],
        optimizer_idx: int = 0,
        optimizer_closure: Optional[Callable[[], Any]] = None,
        on_tpu: bool = False,
        using_native_amp: bool = False,
        using_lbfgs: bool = False,
    ) -> None:
        # update params
        optimizer.step(closure=optimizer_closure)

        # skip the first 500 steps
        if self.trainer.global_step < 500:
            lr_scale = min(1.0, float(self.trainer.global_step + 1) / 500.0)
            for pg in optimizer.param_groups:
                pg["lr"] = lr_scale * self.params.learning_rate
