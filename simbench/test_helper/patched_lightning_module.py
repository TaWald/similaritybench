import pytorch_lightning as pl
import torch
from simbench.arch.ke_architectures.feature_approximation import FAArch
from simbench.losses.dummy_loss import DummyLoss
from simbench.training.ke_train_modules.IntermediateRepresentationLightningModule import (
    IntermediateRepresentationLightningModule,
)
from simbench.training.ke_train_modules.single_lightning_module import SingleLightningModule


def patched_init(self, params, loss, network):
    pl.LightningModule.__init__(self)
    self.net: FAArch = network
    self.loss = loss
    self.save_approx: bool = False
    self.params = params
    torch.backends.cudnn.benchmark = True  # noqa
    self.skip_n_epochs: int | None = None

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


def patched_single_init(self, params, network):
    pl.LightningModule.__init__(self)
    self.net: FAArch = network
    self.loss = DummyLoss()
    self.params = params
    torch.backends.cudnn.benchmark = True  # noqa

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


def getPatchedIntermediateRepresentationLightningModule() -> type[IntermediateRepresentationLightningModule]:
    PatchedIntermediateRepresentationLightningModule = IntermediateRepresentationLightningModule
    PatchedIntermediateRepresentationLightningModule.__init__ = patched_init
    PatchedIntermediateRepresentationLightningModule.log_message = lambda *args, **kwargs: None
    PatchedIntermediateRepresentationLightningModule.training_epoch_end = lambda *args, **kwargs: None
    PatchedIntermediateRepresentationLightningModule.validation_epoch_end = lambda *args, **kwargs: None
    PatchedIntermediateRepresentationLightningModule.validation_step = lambda *args, **kwargs: None
    PatchedIntermediateRepresentationLightningModule.save_checkpoint = lambda *args, **kwargs: None
    PatchedIntermediateRepresentationLightningModule.load_latest_checkpoint = lambda *args, **kwargs: None
    PatchedIntermediateRepresentationLightningModule.on_fit_end = lambda *args, **kwargs: None
    PatchedIntermediateRepresentationLightningModule.save_validation_values = lambda *args, **kwargs: None
    return PatchedIntermediateRepresentationLightningModule


def getPatchedSingleLightningModule() -> type[SingleLightningModule]:
    PatchedSingleLightningModule = SingleLightningModule
    PatchedSingleLightningModule.__init__ = patched_single_init
    PatchedSingleLightningModule.log_message = lambda *args, **kwargs: None
    PatchedSingleLightningModule.training_epoch_end = lambda *args, **kwargs: None
    PatchedSingleLightningModule.validation_epoch_end = lambda *args, **kwargs: None
    PatchedSingleLightningModule.validation_step = lambda *args, **kwargs: None
    PatchedSingleLightningModule.save_checkpoint = lambda *args, **kwargs: None
    PatchedSingleLightningModule.load_latest_checkpoint = lambda *args, **kwargs: None
    PatchedSingleLightningModule.save_validation_values = lambda *args, **kwargs: None
    PatchedSingleLightningModule.on_fit_end = lambda *args, **kwargs: None
    return PatchedSingleLightningModule
