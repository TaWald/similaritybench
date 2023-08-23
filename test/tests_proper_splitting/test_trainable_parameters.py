import unittest

import numpy as np
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


class TestTrainableParametersAreIdentical(unittest.TestCase):
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

        self.faa_pseudo_trainer = Pseudo_faa_trainer(self.faarch, p, faa_loss)
        faa_tmp = self.faa_pseudo_trainer.configure_optimizers()

        self.faa_optim, self.faa_scheduler = faa_tmp[0][0], faa_tmp[1][0]

        self.datamodule = fd.get_datamodule(dataset=dataset)
        # KE specific values.

    def tearDown(self) -> None:
        del self.faarch, self.faa_pseudo_trainer, self.faa_optim, self.faa_scheduler
        del self.datamodule

    def test_trainable_parameters_of_faa_arch_are_same_to_new_arch(self):
        """
        Assure all gradients are trainable.
        """

        new_arch_named_params = dict(self.faa_pseudo_trainer.net.new_arch.named_parameters())
        faa_named_params = dict(self.faa_pseudo_trainer.net.partial_new_modules.named_parameters())
        faa_named_params.update(dict(self.faa_pseudo_trainer.net.linear_layer.named_parameters()))

        self.assertTrue(len(new_arch_named_params) == len(faa_named_params), msg=f"""Lengths are not equal!""")
        for (name_a, param_a), (name_b, param_b) in zip(new_arch_named_params.items(), faa_named_params.items()):
            self.assertTrue(
                np.all(np.equal(param_a.data.cpu().numpy(), param_b.data.cpu().numpy())),
                msg=f"""Parameter {name_a} is not identical to {name_b}!""",
            )


if __name__ == "__main__":
    unittest.main()
