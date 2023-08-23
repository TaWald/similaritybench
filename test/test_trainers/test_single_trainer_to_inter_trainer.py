import copy
import unittest

import numpy as np
import torch
from ke.arch.ke_architectures.feature_approximation import FAArch
from ke.arch.ke_architectures.single_model import SingleModel
from ke.losses.dummy_loss import DummyLoss
from ke.losses.ke_loss import KETrainLoss
from ke.losses.representation_similarity_losses.ke_exp_var import ExpVarLoss
from ke.training.ke_train_modules.base_training_module import BaseLightningModule
from ke.training.ke_train_modules.IntermediateRepresentationLightningModule import (
    IntermediateRepresentationLightningModule,
)
from ke.training.ke_train_modules.single_lightning_module import SingleLightningModule
from ke.util import data_structs as ds
from ke.util import find_architectures as fa
from ke.util import find_datamodules as fd
from ke.util.default_params import get_default_arch_params
from ke.util.default_params import get_default_parameters


class Pseudo_single_model_trainer:
    def __init__(self, network: SingleModel, params: ds.Params, loss: DummyLoss):
        self.net = network
        self.params = params
        self.loss = loss
        self.current_epoch = 0
        self.global_step = 0


Pseudo_single_model_trainer.__call__ = SingleLightningModule.forward
Pseudo_single_model_trainer.forward = SingleLightningModule.forward
Pseudo_single_model_trainer.configure_optimizers = SingleLightningModule.configure_optimizers
Pseudo_single_model_trainer.training_step = SingleLightningModule.training_step


class Pseudo_faa_trainer:
    def __init__(self, network: FAArch, params: ds.Params, loss: KETrainLoss):
        self.net = network
        self.params = params
        self.skip_n_epochs = None
        self.loss = loss
        self.current_epoch = 0
        self.global_step = 0


# Patch
Pseudo_faa_trainer.__call__ = IntermediateRepresentationLightningModule.forward
Pseudo_faa_trainer.forward = IntermediateRepresentationLightningModule.forward
Pseudo_faa_trainer.configure_optimizers = BaseLightningModule.configure_optimizers
Pseudo_faa_trainer.training_step = IntermediateRepresentationLightningModule.training_step

"""
In here we are testing the following:

1. Gradients of the FAArch and SingleModel arch are identical when no weight is
 put on similarity or dissimilarity.
2. When having the FAArch and one sets only one of the weights to non-zero that the gradients behave as expected
"""


class TestCompareKELoss(unittest.TestCase):
    def setUp(self):
        """Here we initialize the two trainers."""
        architecture: ds.BaseArchitecture = ds.BaseArchitecture.RESNET18
        dataset: ds.Dataset = ds.Dataset.TEST

        arch_params = get_default_arch_params(dataset)
        # Just get some random parameters. Does not really matter
        p: ds.Params = get_default_parameters(architecture.value, ds.Dataset.CIFAR10)

        tbt_arch = fa.get_base_arch(architecture)
        tmp_arch = tbt_arch(**arch_params)
        pseudo_hook_position = 2
        pseudo_hooks = (tmp_arch.hooks[pseudo_hook_position],)

        self.single_net: SingleModel = SingleModel(copy.deepcopy(tmp_arch))
        self.single_net.train()
        self.single_pseudo_trainer = Pseudo_single_model_trainer(self.single_net, p, DummyLoss())
        single_tmp = self.single_pseudo_trainer.configure_optimizers()
        self.single_optim, self.single_scheduler = single_tmp[0][0], single_tmp[1][0]

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
        faa_loss = KETrainLoss(ExpVarLoss(False), ExpVarLoss(False), 1, 0, 0, -1, 10)

        self.faarch.train()
        self.faa_pseudo_trainer = Pseudo_faa_trainer(self.faarch, p, faa_loss)
        faa_tmp = self.faa_pseudo_trainer.configure_optimizers()

        self.faa_optim, self.faa_scheduler = faa_tmp[0][0], faa_tmp[1][0]

        self.datamodule = fd.get_datamodule(dataset=dataset)
        # KE specific values.

    def tearDown(self) -> None:
        del self.single_net, self.single_pseudo_trainer, self.single_optim, self.single_scheduler
        del self.faarch, self.faa_pseudo_trainer, self.faa_optim, self.faa_scheduler
        del self.datamodule

    def test_forward_passes_are_identical(self):
        """Assert that the two forward passes have same outcome."""
        for cnt, batch in enumerate(self.datamodule.train_dataloader()):
            with self.subTest(i=cnt):
                x, y = batch
                single_out = self.single_net(x)
                _, _, _, faa_new_out = self.faarch(x)
                self.assertTrue(torch.isclose(single_out, faa_new_out).all())

    def test_correct_faa_parameters_are_trainable(self):
        """Test if all parameters are correctly trainable as intended!"""

        # Everthing new/transferable should be true. (Also since they are object the partials should be identical to the whole arch.)
        all_new_trainable = all([p.requires_grad for p in self.faa_pseudo_trainer.net.new_arch.parameters()])
        self.assertTrue(
            all_new_trainable, "Expect all parameters of the new architecture to be trainable, but not all are!"
        )
        all_new_trainable_partial = all(
            [p.requires_grad for p in self.faa_pseudo_trainer.net.partial_new_modules.parameters()]
        )
        self.assertTrue(
            all_new_trainable_partial,
            "Expect all parameters of the partial new architecture to be trainable, but not all are!",
        )
        all_new_linear = all([p.requires_grad for p in self.faa_pseudo_trainer.net.linear_layer.parameters()])
        self.assertTrue(
            all_new_linear, "Expect all parameters of the partial new architecture to be trainable, but not all are!"
        )
        all_transfer_trainable = all(
            [p.requires_grad for p in self.faa_pseudo_trainer.net.all_transfer_modules.parameters()]
        )
        self.assertTrue(all_transfer_trainable, "Expect all parameters of transfer to be trainable, but not all are!")
        # All old should be not-trainable and therefore not require grad.
        all_old_non_trainable = all(
            [not p.requires_grad for p in self.faa_pseudo_trainer.net.old_archs[0].parameters()]
        )
        self.assertTrue(
            all_old_non_trainable,
            "Expect all parameters of the old architecture to be non-trainable, but not all are!",
        )
        all_partial_old_trainable = all(
            [not p.requires_grad for p in self.faa_pseudo_trainer.net.all_partial_old_models_t.parameters()]
        )
        self.assertTrue(
            all_partial_old_trainable, "Expect all parameters of the partial old architecture to be trainable!"
        )
        all_partial_old_linear = all(
            [not p.requires_grad for p in self.faa_pseudo_trainer.net.all_partial_old_models_linears.parameters()]
        )
        self.assertTrue(
            all_partial_old_linear, "Expect all parameters of the partial old architecture to be trainable!"
        )

    def test_gradients_are_identical(self):
        """Assure the gradients of the FAArch and the SingleModel are identical when no weight is put on
        similarity or dissimilarity."""
        for cnt, batch in enumerate(self.datamodule.train_dataloader()):
            with self.subTest(i=cnt):
                self.faa_pseudo_trainer.training_step(batch, cnt)["loss"].backward()
                self.single_pseudo_trainer.training_step(batch, cnt)["loss"].backward()

                faa_grads = [p.grad.data.cpu().numpy() for p in self.faa_pseudo_trainer.net.new_arch.parameters()]
                all_faa_non_zero = any([np.equal(grad, np.zeros_like(grad)).any() for grad in faa_grads])
                self.assertTrue(all_faa_non_zero, msg=f"Some gradients of the FAArch are all zero but shouldnt be!")

                single_grads = [p.grad.data.cpu().numpy() for p in self.single_pseudo_trainer.net.parameters()]
                all_single_non_zero = any([np.equal(grad, np.zeros_like(grad)).any() for grad in single_grads])
                self.assertTrue(all_single_non_zero, msg=f"Some gradients of the FAArch are all zero!")

                matches = []
                for faa_grad, single_grad in zip(faa_grads, single_grads):
                    matches.append(np.isclose(faa_grad, single_grad).all())
                self.assertTrue(all(matches), msg=f"Gradients are not identical!")

    def test_old_gradients_are_none(self):
        """Assure the old gradients are zero, when we pass a scaling factor of 0 for the similarity loss.
        This makes sure there is not accidentally some gradient flow to the old arch"""
        for cnt, batch in enumerate(self.datamodule.train_dataloader()):
            self.faa_optim.zero_grad()  # Make sure grads are initialized at the beginning
            with self.subTest(i=cnt):
                self.faa_pseudo_trainer.training_step(batch, cnt)["loss"].backward()
                old_grads_are_none = [p.grad is None for p in self.faa_pseudo_trainer.net.old_archs[0].parameters()]

                self.assertTrue(all(old_grads_are_none), msg=f"Gradients are not None!")


if __name__ == "__main__":
    unittest.main()
