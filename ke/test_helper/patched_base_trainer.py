from ke.data.base_datamodule import BaseDataModule
from ke.training.ke_train_modules.base_training_module import BaseLightningModule
from ke.training.trainers.base_trainer import BaseTrainer
from ke.util import data_structs as ds
from pytorch_lightning import Trainer


def patched_base_trainer_init(
    self,
    model: BaseLightningModule,
    datamodule: BaseDataModule,
    params: ds.Params,
):
    self.model: BaseLightningModule = model
    self.datamodule: BaseDataModule = datamodule
    self.params = params
    self.params.batch_size = 100
    self.num_workers = 0  # Single threaded
    self.prog_bar = True

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


def patched_train(self):
    trainer = Trainer(
        enable_checkpointing=False,
        benchmark=True,
        deterministic=True,
        max_epochs=1,
        accelerator="cpu",
        devices=1,
        precision=16,
        default_root_dir=None,
        enable_progress_bar=True,
        logger=False,
        profiler=None,
    )
    trainer.fit(self.model, train_dataloaders=self.datamodule.train_dataloader(), val_dataloaders=None)


def get_patched_trainer() -> type[BaseTrainer]:
    PatchedBaseTrainer = BaseTrainer
    PatchedBaseTrainer.__init__ = patched_base_trainer_init
    PatchedBaseTrainer.post_train_eval = lambda *args, **kwargs: None
    PatchedBaseTrainer.save_outputs = lambda *args, **kwargs: None
    PatchedBaseTrainer.save_activations = lambda *args, **kwargs: None
    PatchedBaseTrainer.train = patched_train
    PatchedBaseTrainer.calculate_accuracy = lambda *args, **kwargs: None
    return PatchedBaseTrainer
