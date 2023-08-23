import unittest

import numpy as np
from ke.arch.ke_architectures.feature_approximation import FAArch
from ke.arch.ke_architectures.single_model import SingleModel
from ke.data.test_dm import TestDataModule
from ke.losses.ke_loss import KETrainLoss
from ke.losses.representation_similarity_losses.ke_exp_var import ExpVarLoss
from ke.test_helper.patched_base_trainer import get_patched_trainer
from ke.test_helper.patched_lightning_module import getPatchedIntermediateRepresentationLightningModule
from ke.test_helper.patched_lightning_module import getPatchedSingleLightningModule
from ke.training.ke_train_modules import IntermediateRepresentationLightningModule
from ke.training.ke_train_modules.single_lightning_module import SingleLightningModule
from ke.training.trainers.base_trainer import BaseTrainer
from ke.util import data_structs as ds
from ke.util import find_architectures as fa
from ke.util.default_params import get_default_arch_params
from ke.util.default_params import get_default_parameters


class TestTrainerBehaveIdenticallyWhenOnlyCELoss(unittest.TestCase):
    def setUp(self):
        """Initialize KETrainLoss to only weight the ce_loss and nothing else."""
        architecture: ds.BaseArchitecture = ds.BaseArchitecture.RESNET18
        dataset: ds.Dataset = ds.Dataset.TEST

        arch_params = get_default_arch_params(dataset)
        # Just get some random parameters. Does not really matter
        p: ds.Params = get_default_parameters(architecture.value, ds.Dataset.CIFAR10)

        tbt_arch = fa.get_base_arch(architecture)
        tmp_arch = tbt_arch(**arch_params)
        pseudo_hook_position = 2
        pseudo_hooks = (tmp_arch.hooks[pseudo_hook_position],)

        # Create the FA arch.
        old_arch_infos = [
            ds.ArchitectureInfo(
                arch_type_str="ResNet18", arch_kwargs=arch_params, checkpoint=None, hooks=pseudo_hooks
            )
        ]
        new_arch_info = ds.ArchitectureInfo(
            arch_type_str="ResNet18", arch_kwargs=arch_params, checkpoint=None, hooks=pseudo_hooks
        )
        faarch = FAArch(old_arch_infos, new_arch_info, True, 1, 1)
        faarch.load_individual_state_dicts(tmp_arch.state_dict(), None, None)

        # The above line is unneeded here, but is kept as it is done for the parallel comparison of the
        #  SimpleModel and the FAArch.

        # THIS HERE IS DIFFERENT! No CE Weight only SIM Weight --> Make sure gradients are Zero for new model!
        faa_loss = KETrainLoss(
            ExpVarLoss(False),
            ExpVarLoss(False),
            ce_weight=1.0,
            dissim_weight=0,
            sim_weight=0,
            regularization_epoch_start=-1,
            n_classes=10,
        )

        datamodule = TestDataModule()

        patched_inter_lightning_module = getPatchedIntermediateRepresentationLightningModule()
        patched_intermediate_lightning_module = patched_inter_lightning_module(p, network=faarch, loss=faa_loss)
        self.ilm_trainer: BaseTrainer = get_patched_trainer()(
            model=patched_intermediate_lightning_module,
            datamodule=datamodule,
            params=p,
        )

        # ======= Here the initialization of the SingleModel starts =======

        # Init to the same state as the FAAarch starts with!
        single_model_arch = SingleModel(tbt_arch(**arch_params))
        single_model_arch.tbt_arch.load_state_dict(tmp_arch.state_dict())
        single_lightning_module = getPatchedSingleLightningModule()(params=p, network=single_model_arch)
        self.single_trainer: BaseTrainer = get_patched_trainer()(
            model=single_lightning_module, datamodule=datamodule, params=p
        )

    def tearDown(self) -> None:
        del self.faarch, self.patched_intermediate_lightning_module

    def get_single_trainers_arch_state_dict(self, single_trainer: SingleLightningModule):
        """Retrieves the AbsActiArch state dict from the SingleLightningModule"""
        return {k: v.numpy() for k, v in single_trainer.model.net.tbt_arch.state_dict().items()}

    def get_inter_trainers_arch_state_dict(self, inter_trainer: IntermediateRepresentationLightningModule):
        """Retrieves the AbsActiArch state dict from the IntermediateRepresentationLightningModule"""
        return {k: v.numpy() for k, v in inter_trainer.model.net.new_arch.state_dict().items()}

    def weights_are_identical(self, state_dict_a: dict[str : np.ndarray], state_dict_b: dict[str : np.ndarray]):
        """Returns true if all weights are identical."""
        for k, v in state_dict_a.items():
            vals_are_equal = v == state_dict_b[k]
            if not np.all(vals_are_equal):
                return False
        return True

    def all_weights_changed(self, state_dict_a: dict, state_dict_b: dict):
        """Measure if every tensor changed at least a little bit."""
        for k, v in state_dict_a.items():
            if np.all(np.equal(v.numpy(), state_dict_b[k].numpy())):
                return False
        return True

    def test_weights_change_the_same_way(self):
        """Does one epoch across the TestDataset and checks if all weights changed."""
        t0_single_state_dict = self.get_single_trainers_arch_state_dict(self.single_trainer)
        t0_inter_state_dict = self.get_inter_trainers_arch_state_dict(self.ilm_trainer)
        self.assertTrue(
            self.weights_are_identical(t0_single_state_dict, t0_inter_state_dict),
            msg=f"Weights at t0 are not identical between inter and single!",
        )
        # Train the single trainer
        self.single_trainer.train()
        t1_single_state_dict = self.get_single_trainers_arch_state_dict(self.single_trainer)
        t1_inter_state_dict = self.get_inter_trainers_arch_state_dict(self.ilm_trainer)
        self.assertTrue(
            self.weights_are_identical(t0_inter_state_dict, t1_inter_state_dict),
            msg=f"Weights at t0 and t1 are not identical of inter state dict despite not training!",
        )
        self.assertTrue(
            self.all_weights_changed(t0_single_state_dict, t1_single_state_dict),
            msg=f"Weights at t0 and t1 are identical of single state dict despite training!",
        )

        self.ilm_trainer.train()
        t2_single_state_dict = self.get_single_trainers_arch_state_dict(self.single_trainer)
        t2_inter_state_dict = self.get_inter_trainers_arch_state_dict(self.ilm_trainer)
        self.assertTrue(
            self.all_weights_changed(t1_inter_state_dict, t2_inter_state_dict),
            msg=f"Weights at t1 and t2 are identical of inter state dict despite training!",
        )
        self.assertTrue(
            self.weights_are_identical(t2_single_state_dict, t2_inter_state_dict),
            msg=f"Weights at t2 are not identical between inter and single, should be though!!",
        )


if __name__ == "__main__":
    unittest.main()
