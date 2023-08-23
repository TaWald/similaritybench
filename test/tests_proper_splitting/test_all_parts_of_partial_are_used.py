import unittest

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


def was_called_hook(was_called: set, name: str) -> bool:
    def hook(*args):
        """
        Attaches a forward hook that sets `was_called` to true when it is hit.
        """
        was_called.add(name)

    return hook


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


class TestAllPartialsAreUsed_ResNet18(unittest.TestCase):
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

        self.were_called: set[str] = set()

    def tearDown(self) -> None:
        del self.faarch, self.faa_pseudo_trainer, self.faa_optim, self.faa_scheduler
        del self.datamodule

    def test_first_part_is_hit(self):
        """
        Assure all gradients are trainable.
        """

        first_relu_module = self.faa_pseudo_trainer.net.partial_new_modules[0].relu
        first_layer_basic_block_module = self.faa_pseudo_trainer.net.partial_new_modules[0].layer1[0]
        last_layer_basic_block_module = self.faa_pseudo_trainer.net.partial_new_modules[0].layer1[-1].identity

        first_relu_module.register_forward_hook(was_called_hook(self.were_called, "first_relu"))
        first_layer_basic_block_module.register_forward_hook(was_called_hook(self.were_called, "first_basic"))
        last_layer_basic_block_module.register_forward_hook(was_called_hook(self.were_called, "last_basic_identity"))

        for cnt, batch in enumerate(self.datamodule.train_dataloader()):
            self.faa_pseudo_trainer.training_step(batch, cnt)["loss"]

        self.assertTrue(
            self.were_called.issuperset(["first_relu", "first_basic", "last_basic_identity"]),
            msg=f"""Not all first_parts were called!""",
        )

    def test_second_part_is_hit(self):
        """
        Assure all gradients are trainable.
        """

        layer1 = self.faa_pseudo_trainer.net.partial_new_modules[1].layer1
        layer2_block = self.faa_pseudo_trainer.net.partial_new_modules[1].layer2[0]
        layer2_block_conv2 = self.faa_pseudo_trainer.net.partial_new_modules[1].layer2[0].conv2
        layer2_block2 = self.faa_pseudo_trainer.net.partial_new_modules[1].layer2[2]
        layer3_block0 = self.faa_pseudo_trainer.net.partial_new_modules[1].layer3[0]
        layer3_block0_downsample = self.faa_pseudo_trainer.net.partial_new_modules[1].layer3[0].downsample
        layer3_block2 = self.faa_pseudo_trainer.net.partial_new_modules[1].layer3[2]
        layer4_block2 = self.faa_pseudo_trainer.net.partial_new_modules[1].layer4[2]

        layer1.register_forward_hook(was_called_hook(self.were_called, "a"))
        layer2_block.register_forward_hook(was_called_hook(self.were_called, "b"))
        layer2_block_conv2.register_forward_hook(was_called_hook(self.were_called, "c"))
        layer2_block2.register_forward_hook(was_called_hook(self.were_called, "d"))
        layer3_block0.register_forward_hook(was_called_hook(self.were_called, "e"))
        layer3_block0_downsample.register_forward_hook(was_called_hook(self.were_called, "f"))
        layer3_block2.register_forward_hook(was_called_hook(self.were_called, "g"))
        layer4_block2.register_forward_hook(was_called_hook(self.were_called, "h"))

        for cnt, batch in enumerate(self.datamodule.train_dataloader()):
            self.faa_pseudo_trainer.training_step(batch, cnt)["loss"]

        self.assertTrue(
            self.were_called.issuperset(["a", "b", "c", "d", "e", "f", "g", "h"]),
            msg=f"""Not all first_parts were called!""",
        )

    def test_assert_partial_linear_layer_is_hit(self):
        """
        Assure linear_layer_is_hit are trainable.
        """

        linear_layer = self.faa_pseudo_trainer.net.linear_layer

        linear_layer.register_forward_hook(was_called_hook(self.were_called, "linear"))

        for cnt, batch in enumerate(self.datamodule.train_dataloader()):
            self.faa_pseudo_trainer.training_step(batch, cnt)["loss"]

        self.assertTrue(
            self.were_called.issuperset(["linear"]),
            msg=f"""Not all first_parts were called!""",
        )


if __name__ == "__main__":
    unittest.main()
