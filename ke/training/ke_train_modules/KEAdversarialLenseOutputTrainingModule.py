from __future__ import annotations

from dataclasses import asdict

import ke.util.name_conventions as nc
import torch
from ke.arch.ke_architectures.adversarial_lense_training_arch import AdversarialLenseTrainingArch
from ke.losses.ke_lense_output_loss import KELenseOutputTrainLoss
from ke.metrics.ke_metrics import multi_output_metrics
from ke.training.ke_train_modules.base_training_module import BaseLightningModule
from ke.util import data_structs
from ke.util import data_structs as ds
from torch.optim.lr_scheduler import _LRScheduler as LRScheduler  # noqa


class KEAdversarialLenseOutputLightningModule(BaseLightningModule):
    def __init__(
        self,
        model_info: ds.KEOutputTrainingInfo,
        network: AdversarialLenseTrainingArch,
        save_checkpoints: bool,
        params: data_structs.Params,
        hparams: dict,
        loss: KELenseOutputTrainLoss,
        skip_n_epochs: int | None = None,
        log: bool = True,
    ):
        super().__init__(model_info, save_checkpoints, params, hparams, skip_n_epochs, log)
        self.loss: KELenseOutputTrainLoss = loss
        self.net: AdversarialLenseTrainingArch = network  # To overwrite typehints

    def save_checkpoint(self):
        if self.current_epoch == 0:
            return
        if self.save_checkpoints:
            if self.current_epoch == (self.params.num_epochs - 1):
                state_dict = self.net.get_state_dict()
                torch.save(state_dict, self.checkpoint_path)
                lense_dict = self.net.get_lense_state_dict()
                torch.save(lense_dict, self.checkpoint_dir_path / nc.KE_LENSE_CKPT_NAME)
        return

    def load_latest_checkpoint(self):
        """Loads the latest checkpoint (only of the to be trained architecture)"""
        ckpt = torch.load(self.checkpoint_path)
        self.net.get_new_model().load_state_dict(ckpt)

    def training_step(self, batch, batch_idx):
        x, y = batch
        x_lense, y_hat, source_y_hats = self(x)
        ke_fwd = self.loss.ke_forward(
            label=y,
            new_out=y_hat,
            old_outs=source_y_hats,
            epoch_num=self.current_epoch,
            global_step=self.global_step,
        )
        return ke_fwd

    def training_epoch_end(self, outputs: list[dict]):
        # Scheduler steps are done automatically! (No step is needed)
        loss_values = self.loss.on_ke_epoch_end(outputs)
        prog_bar_log = {"tr/loss": loss_values["loss/total_loss"]}

        self.log_dict(prog_bar_log, prog_bar=True, logger=False)
        self.log_message(loss_values, is_train=True)

        return None

    def is_not_too_large(self, t: list[torch.Tensor]) -> bool:
        num_ele = sum([torch.numel(te) for te in t])
        if self.max_data_points >= num_ele:
            return True
        else:
            return False

    def validation_step(self, batch, batch_idx):
        x, y = batch  # ["data"], batch["label"]
        with torch.no_grad():
            y_hat, y_hats = self(x)
            self.save_validation_values(
                old_intermediate_reps=None,
                new_intermediate_reps=None,
                groundtruths=y,
                old_y_hats=y_hats,
                new_y_hat=y_hat,
                transferred_y_hats=None,
            )
            return self.loss.ke_forward(
                label=y,
                new_out=y_hat,
                old_outs=y_hats,
                epoch_num=self.current_epoch,
                global_step=self.global_step,
            )

    def validation_epoch_end(self, outputs):
        self.save_checkpoint()
        loss_values = self.loss.on_ke_epoch_end(outputs)

        tensorboard_dict = loss_values
        tensor_metrics = multi_output_metrics(
            new_output=self.newly_trained_y_out,
            old_outputs=torch.stack(self.already_trained_y_outs, dim=0),
            groundtruth=self.gts,
            dataset=self.params.dataset,
            architecture=self.params.architecture_name,
            n_cls=self.ke_hparams["n_cls"],
        )
        self.final_metrics = {k: float(v) for k, v in asdict(tensor_metrics).items()}

        tensorboard_dict.update({f"metrics/{k}": v for k, v in self.final_metrics.items()})
        prog_bar_log = {
            "val/acc": self.final_metrics["acc"],
            "val/CoKa": self.final_metrics["cohens_kappa"],
        }
        self.log_dict(prog_bar_log, prog_bar=True, logger=False)
        self.log_message(tensorboard_dict, is_train=False)
        return None
