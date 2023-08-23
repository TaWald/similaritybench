import unittest
from copy import deepcopy

import numpy as np
from ke.arch.ke_architectures.feature_approximation import FAArch
from ke.data.test_dm import TestDataModule
from ke.losses.ke_loss import KETrainLoss
from ke.losses.representation_similarity_losses.ke_exp_var import ExpVarLoss
from ke.test_helper.patched_base_trainer import get_patched_trainer
from ke.test_helper.patched_lightning_module import getPatchedIntermediateRepresentationLightningModule
from ke.util import data_structs as ds
from ke.util import find_architectures as fa
from ke.util.default_params import get_default_arch_params
from ke.util.default_params import get_default_parameters


class TestGradientsWhenOnlyCELoss(unittest.TestCase):
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
        self.faarch = FAArch(old_arch_infos, new_arch_info, True, 1, 1)
        self.faarch.load_individual_state_dicts(tmp_arch.state_dict(), None, None)

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
        self.patched_intermediate_lightning_module = patched_inter_lightning_module(
            p, network=self.faarch, loss=faa_loss
        )
        self.trainer = get_patched_trainer()(
            model=self.patched_intermediate_lightning_module,
            datamodule=datamodule,
            params=p,
        )

    def tearDown(self) -> None:
        del self.faarch, self.patched_intermediate_lightning_module

    def test_all_weights_change_during_training(self):
        """Does one epoch across the TestDataset and checks if all weights changed."""

        pre_state_dict = self.patched_intermediate_lightning_module.net.new_arch.state_dict()
        weights_pre_train = {k: deepcopy(v.numpy()) for k, v in pre_state_dict.items()}
        self.trainer.train()
        post_state_dict = self.patched_intermediate_lightning_module.net.new_arch.state_dict()
        weights_post_train = {k: deepcopy(v.numpy()) for k, v in post_state_dict.items()}

        weight_diff = {k: v - weights_post_train[k] for k, v in weights_pre_train.items()}
        changed = {~np.all(np.isclose(v, 0)) for v in weight_diff.values()}
        all_changed = all(changed)
        # Assert the weights are not close
        self.assertTrue(all_changed, msg=f"Not all weights changed during training! {weight_diff}")

    def test_all_new_arch_gradients_are_trainable(self):
        """
        Assure all gradients are trainable.
        """

        new_models_trainable = all(
            [p.requires_grad for p in self.patched_intermediate_lightning_module.net.new_arch.parameters()]
        )
        self.assertTrue(
            new_models_trainable,
            msg=f"""All new_model gradients are trainable!""",
        )

    def test_all_new_partial_arch_gradients_are_trainable(self):
        """
        Assure all gradients of the partial new models  are trainable
        (Should be the subset of parameters from `test_all_new_arch_gradients_are_trainable` but
        somehow other test is throws).
        """

        new_models_trainable = all(
            [p.requires_grad for p in self.patched_intermediate_lightning_module.net.partial_new_modules.parameters()]
        )
        self.assertTrue(
            new_models_trainable,
            msg=f"""All partial new_model gradients are trainable!""",
        )

    def test_new_models_gradients_are_non_zero(self):
        """Test that all gradients of the new model are non zero"""
        self.patched_intermediate_lightning_module.net.train()
        tmp = self.patched_intermediate_lightning_module.configure_optimizers()
        optim = tmp[0][0]
        for cnt, batch in enumerate(self.trainer.datamodule.train_dataloader()):
            optim.zero_grad()
            with self.subTest(i=cnt):
                loss = self.patched_intermediate_lightning_module.training_step(batch, cnt)["loss"]
                loss.backward()

                new_models_trainable = all(
                    [
                        p.requires_grad
                        for p in self.patched_intermediate_lightning_module.net.partial_new_modules.parameters()
                    ]
                )
                self.assertTrue(
                    new_models_trainable,
                    msg=f"""All partial new_model gradients are trainable!""",
                )

                new_models_trainable = all(
                    [p.requires_grad for p in self.patched_intermediate_lightning_module.net.new_arch.parameters()]
                )
                self.assertTrue(
                    new_models_trainable,
                    msg=f"""All partial new_model gradients are trainable!""",
                )

                faa_grads_new = {
                    k: v.grad.data
                    for k, v in self.patched_intermediate_lightning_module.net.new_arch.named_parameters()
                }

                faa_grads_partial_new = {
                    k: v
                    for k, v in self.patched_intermediate_lightning_module.net.partial_new_modules.named_parameters()
                }
                faa_grads_linear_new = {
                    k: v for k, v in self.patched_intermediate_lightning_module.net.linear_layer.named_parameters()
                }

                for k, v in faa_grads_linear_new.items():
                    self.assertTrue(v.requires_grad, msg=f"Param of linear layer {k} does not require grad!")

                for k, v in faa_grads_partial_new.items():
                    self.assertTrue(v.requires_grad, f"Param of partial new model {k} does not require grad!")

                # Make sure some gradients are not zero for the old architectuer somewhere.
                grads = [g.data.cpu().numpy() for g in faa_grads_new.values()]
                self.assertTrue(
                    np.all([not np.all(g == 0) for g in grads]),
                    msg=f"New models gradients are zero but shouldnt be!",
                )

    def test_all_old_arch_gradients_are_none(self):
        """Test that all gradients of the old model are None"""
        tmp = self.patched_intermediate_lightning_module.configure_optimizers()
        optim = tmp[0][0]
        for cnt, batch in enumerate(self.trainer.datamodule.train_dataloader()):
            optim.zero_grad()
            with self.subTest(i=cnt):
                self.patched_intermediate_lightning_module.training_step(batch, cnt)["loss"].backward()

                faa_grads_old = [
                    p.grad for p in self.patched_intermediate_lightning_module.net.old_archs[0].parameters()
                ]
                # Make sure some gradients are not zero for the old architectuer somewhere.
                self.assertTrue(all([g is None for g in faa_grads_old]), msg=f"Gradients are not None!")

    def test_all_transfer_gradients_are_zero(self):
        """Test that all gradients of the transfer model are zero"""
        tmp = self.patched_intermediate_lightning_module.configure_optimizers()
        optim = tmp[0][0]
        for cnt, batch in enumerate(self.trainer.datamodule.train_dataloader()):
            optim.zero_grad()
            with self.subTest(i=cnt):
                self.patched_intermediate_lightning_module.training_step(batch, cnt)["loss"].backward()

                faa_grads_transfer = [
                    p.grad.data.cpu().numpy()
                    for p in self.patched_intermediate_lightning_module.net.all_transfer_modules.parameters()
                ]
                # Make sure some gradients are not zero for the old architectuer somewhere.
                self.assertTrue(all([np.all(g == 0) for g in faa_grads_transfer]), msg=f"Gradients are not 0!")


if __name__ == "__main__":
    unittest.main()
