import unittest

import numpy as np
import torch
from ke.arch.ke_architectures.feature_approximation import FAArch
from ke.losses.ke_loss import KETrainLoss
from ke.losses.representation_similarity_losses.ke_exp_var import ExpVarLoss
from ke.training.ke_train_modules.base_training_module import BaseLightningModule
from ke.training.ke_train_modules.IntermediateRepresentationLightningModule import (
    IntermediateRepresentationLightningModule,
)
from ke.util import data_structs as ds
from ke.util import find_architectures as fa
from ke.util import find_datamodules as fd
from ke.util.default_params import get_default_arch_params
from ke.util.default_params import get_default_parameters


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


class TestGradientsWhenOnlySimilarity(unittest.TestCase):
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

        # THIS HERE IS DIFFERENT! No CE Weight only SIM Weight --> Make sure gradients are Zero for new model!
        faa_loss = KETrainLoss(
            ExpVarLoss(False),
            ExpVarLoss(False),
            ce_weight=0,
            dissim_weight=0,
            sim_weight=1.0,
            regularization_epoch_start=-1,
        )

        self.faa_pseudo_trainer = Pseudo_faa_trainer(self.faarch, p, faa_loss)
        faa_tmp = self.faa_pseudo_trainer.configure_optimizers()

        self.faa_optim: torch.optim.Optimizer
        self.faa_optim, self.faa_scheduler = faa_tmp[0][0], faa_tmp[1][0]

        self.datamodule = fd.get_datamodule(dataset=dataset)
        # KE specific values.

    def tearDown(self) -> None:
        del self.faarch, self.faa_pseudo_trainer, self.faa_optim, self.faa_scheduler
        del self.datamodule

    def test_all_new_arch_gradients_are_trainable(self):
        """
        Assure all gradients are trainable.
        """

        new_models_trainable = all([p.requires_grad for p in self.faa_pseudo_trainer.net.new_arch.parameters()])
        self.assertTrue(
            new_models_trainable,
            msg=f"""All new_model gradients are trainable!""",
        )

    def test_all_transfer_gradients_are_trainable(self):
        """
        Assure all transfer gradients are trainable.
        """
        trans_models_trainable = all(
            [p.requires_grad for p in self.faa_pseudo_trainer.net.all_transfer_modules.parameters()]
        )
        self.assertTrue(
            trans_models_trainable,
            msg=f"""Transfer Gradients are not all trainable.""",
        )

    def test_transfer_gradients_are_non_zero(self):
        """
        Assure the gradients of the new architecture are zero.
        When no dissimilarity or CE loss is weighted the new arch should not be updated.
        This makes sure there is not accidentally some gradient flow to the old arch"""
        for cnt, batch in enumerate(self.datamodule.train_dataloader()):
            self.faa_optim.zero_grad()  # Make sure they are zero at the beginning
            with self.subTest(i=cnt):
                self.faa_pseudo_trainer.training_step(batch, cnt)["loss"].backward()

                transfer_module_gradients = [
                    p.grad.data.cpu().numpy() for p in self.faa_pseudo_trainer.net.all_transfer_modules.parameters()
                ]
                non_zero = [np.not_equal(grads, np.zeros_like(grads)).all() for grads in transfer_module_gradients]
                self.assertTrue(
                    all(non_zero),
                    msg=f"""New Gradients are non-zero when they should be Zero.
                        Similarity gradients leaking into new_arch weight updates!""",
                )

    def test_old_arch_gradients_are_none(self):
        """
        Assure that the old architecture gradients are zero,
        when similarity is to be enforced."""
        for cnt, batch in enumerate(self.datamodule.train_dataloader()):
            self.faa_optim.zero_grad()  # Make sure they are zero at the beginning
            with self.subTest(i=cnt):
                self.faa_pseudo_trainer.training_step(batch, cnt)["loss"].backward()

                named_params = {k: v for k, v in self.faa_pseudo_trainer.net.old_archs[0].named_parameters()}
                all_none = all([v.grad is None for v in named_params.values()])
                # Make sure some gradients are not zero for the old architectuer somewhere.

                self.assertTrue(
                    all_none,
                    msg=f"""New Gradients are non-zero when they should be Zero.
                        Similarity gradients leaking into new_arch weight updates!""",
                )

    def test_new_arch_gradients_are_zero(self):
        """
        Assure that the old architecture gradients are zero,
        when similarity is to be enforced."""
        for cnt, batch in enumerate(self.datamodule.train_dataloader()):
            self.faa_optim.zero_grad()  # Make sure they are zero at the beginning
            with self.subTest(i=cnt):
                self.faa_pseudo_trainer.training_step(batch, cnt)["loss"].backward()

                new_grads = [p.grad.data.cpu().numpy() for p in self.faa_pseudo_trainer.net.new_arch.parameters()]
                # Make sure some gradients are not zero for the old architectuer somewhere.
                all_zero = all([(np.equal(new_grad, np.zeros_like(new_grad))).all() for new_grad in new_grads])
                self.assertTrue(
                    all_zero,
                    msg=f"""All Gradients of the new architecture should be 0""",
                )


if __name__ == "__main__":
    unittest.main()
