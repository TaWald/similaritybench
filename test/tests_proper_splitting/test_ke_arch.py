import unittest
from abc import ABC
from abc import abstractmethod
from typing import Sequence

import numpy as np
import torch
from ke.arch.ke_architectures.feature_approximation import FAArch
from ke.arch.resnet import ResNet101
from ke.arch.resnet import ResNet18
from ke.arch.resnet import ResNet34
from ke.util.data_structs import ArchitectureInfo
from ke.util.data_structs import Hook
from torch.nn import functional as F

batch = 64
width = 160
height = width
channel = 3


def create_random_tensor(shape: Sequence) -> torch.Tensor:
    random_np = np.random.random(shape)
    return torch.from_numpy(random_np)


class TestAbstractResNetSplitting(ABC):
    @abstractmethod
    def setUp(self) -> None:
        self.arch_kwargs = {
            "n_cls": 10,
            "in_ch": 3,
            "input_resolution": (32, 32),
            "early_downsampling": False,
            "global_average_pooling": 4,
        }
        self.resnet = ResNet18(**self.arch_kwargs)

    def create_faa_arch_for_hook_id(self, hook_id: int) -> Hook:
        return self.resnet.hooks[hook_id]

    def create_faa_of_interest(self, hook_id: int) -> FAArch:
        hook = self.create_faa_arch_for_hook_id(hook_id)
        arch_info = ArchitectureInfo(
            self.resnet.__class__.__name__,
            self.arch_kwargs,
            checkpoint=None,
            hooks=(hook,),
        )

        return FAArch([arch_info], arch_info, aggregate_old_reps=True, transfer_depth=1, transfer_kernel_width=1)

    def test_partial_parameters_are_identical_to_new_arch(self):
        """Tests if each parameter object is identical to its counterpart in the new_arch."""
        faa_arch = self.create_faa_of_interest(0)

        new_arch_of_faa_arch = faa_arch.new_arch
        initial_part, last_part = faa_arch.partial_new_modules
        linear_layer = faa_arch.linear_layer

        new_arch_params = list(new_arch_of_faa_arch.parameters())
        partialized_params = (
            list(initial_part.parameters()) + list(last_part.parameters()) + list(linear_layer.parameters())
        )

        for new_arch_param, partialized_param in zip(new_arch_params, partialized_params):
            self.assertTrue(
                new_arch_param is partialized_param,
                "Splitting went wrong. Expected parameters objects to be identical, but got different ones.",
            )

    def test_forward_is_equal(self):
        """Test if output of original new_arch and split one are the same."""
        pseudo_input = torch.rand(
            1,
            self.arch_kwargs["in_ch"],
            self.arch_kwargs["input_resolution"][0],
            self.arch_kwargs["input_resolution"][1],
        )
        for i in range(len(self.resnet.hooks)):
            with self.subTest(i=i):
                faa = self.create_faa_of_interest(i)

                with torch.no_grad():
                    original_output = faa.new_arch(pseudo_input)
                    partial_output = pseudo_input
                    for partial in faa.partial_new_modules:
                        partial_output = partial(partial_output)
                    partial_output = faa.linear_layer(partial_output)

                    self.assertTrue(
                        torch.isclose(partial_output, original_output).all(),
                        "Splitting went wrong. Expected identical outputs, but got different ones.",
                    )

    def test_trainable_params_is_identical(self):
        """Test if output of original new_arch and split one are the same."""

        for i in range(len(self.resnet.hooks)):
            with self.subTest(i=i):
                faa = self.create_faa_of_interest(i)

                original_n_params = sum(p.numel() for p in faa.new_arch.parameters() if p.requires_grad)
                # Split parts
                partial_n_params = sum(p.numel() for p in faa.partial_new_modules.parameters() if p.requires_grad)
                linear_n_params = sum(p.numel() for p in faa.linear_layer.parameters() if p.requires_grad)
                all_split_n_params = partial_n_params + linear_n_params
                self.assertTrue(
                    original_n_params == all_split_n_params
                ), "Splitting went wrong. Expected identical number of trainable parameters"

    def test_grad_is_equal(self):
        """Test if gradient is identical when passing through the `new_arch` or through the `partial and split one are the same."""
        pseudo_input = torch.rand(
            1,
            self.arch_kwargs["in_ch"],
            self.arch_kwargs["input_resolution"][0],
            self.arch_kwargs["input_resolution"][1],
        )
        pseudo_label = torch.randint(0, self.arch_kwargs["n_cls"], (1,))
        for i in range(len(self.resnet.hooks)):
            with self.subTest(i=i):
                faa = self.create_faa_of_interest(i)

                # Create Original Gradients
                original_output = faa.new_arch(pseudo_input)
                # Forward
                _ = F.cross_entropy(original_output, pseudo_label).backward()
                original_grads = [p.grad.data.numpy() for p in faa.new_arch.parameters() if p.requires_grad]
                faa.new_arch.zero_grad()

                # Create Gradients when going through partial path.
                partial_output = pseudo_input
                for partial in faa.partial_new_modules:
                    partial_output = partial(partial_output)
                partial_output = faa.linear_layer(partial_output)
                # Grad Calc
                _ = F.cross_entropy(partial_output, pseudo_label).backward()
                partial_grads = [p.grad.data.numpy() for p in faa.new_arch.parameters() if p.requires_grad]
                faa.new_arch.zero_grad()

                all_close = np.all([np.all(np.isclose(og, pg)) for og, pg in zip(original_grads, partial_grads)])

                self.assertTrue(
                    all_close,
                    "Splitting went wrong. Expected close gradients, but got different ones.",
                )

    def test_assure_outputs_are_identical(self):
        """Test if output of original new_arch and split one are the same."""
        pseudo_input = torch.rand(
            1,
            self.arch_kwargs["in_ch"],
            self.arch_kwargs["input_resolution"][0],
            self.arch_kwargs["input_resolution"][1],
        )
        for i in range(len(self.resnet.hooks)):
            with self.subTest(i=i):
                faa = self.create_faa_of_interest(i)

                with torch.no_grad():
                    pseudo_output = faa.simple_forward(pseudo_input)
                    original_pseudo_output = faa.new_arch(pseudo_input)
                    assert torch.isclose(
                        pseudo_output, original_pseudo_output
                    ).all(), "Splitting went wrong. Expected identical outputs, but got different ones."

    def test_assure_trainable_parameters_are_identical(self):
        """Assures the trainable parameters of the split and the normal one are the same."""

        for i in range(len(self.resnet.hooks)):
            with self.subTest(i=i):
                faa = self.create_faa_of_interest(i)

                original_n_params = sum(p.numel() for p in faa.new_arch.parameters() if p.requires_grad)
                # Split parts
                partial_n_params = sum(p.numel() for p in faa.partial_new_modules.parameters() if p.requires_grad)
                linear_n_params = sum(p.numel() for p in faa.linear_layer.parameters() if p.requires_grad)
                all_split_n_params = partial_n_params + linear_n_params
                assert (
                    original_n_params == all_split_n_params
                ), "Splitting went wrong. Expected identical number of trainable parameters"


class TestResNet18_32x32(unittest.TestCase, TestAbstractResNetSplitting):
    def setUp(self) -> None:
        self.arch_kwargs = {
            "n_cls": 10,
            "in_ch": 3,
            "input_resolution": (32, 32),
            "early_downsampling": False,
            "global_average_pooling": 4,
        }
        self.resnet = ResNet18(**self.arch_kwargs)


class TestResNet18_160x160(unittest.TestCase, TestAbstractResNetSplitting):
    def setUp(self) -> None:
        self.arch_kwargs = {
            "n_cls": 1000,
            "in_ch": 3,
            "input_resolution": (160, 160),
            "early_downsampling": True,
            "global_average_pooling": 5,
        }
        self.resnet = ResNet18(**self.arch_kwargs)


class TestResNet34_32x32(unittest.TestCase, TestAbstractResNetSplitting):
    def setUp(self) -> None:
        self.arch_kwargs = {
            "n_cls": 10,
            "in_ch": 3,
            "input_resolution": (32, 32),
            "early_downsampling": False,
            "global_average_pooling": 4,
        }
        self.resnet = ResNet34(**self.arch_kwargs)


class TestResNet34_160x160(unittest.TestCase, TestAbstractResNetSplitting):
    def setUp(self) -> None:
        self.arch_kwargs = {
            "n_cls": 1000,
            "in_ch": 3,
            "input_resolution": (160, 160),
            "early_downsampling": True,
            "global_average_pooling": 5,
        }
        self.resnet = ResNet34(**self.arch_kwargs)


class TestResNet101_32x32(unittest.TestCase, TestAbstractResNetSplitting):
    def setUp(self) -> None:
        self.arch_kwargs = {
            "n_cls": 10,
            "in_ch": 3,
            "input_resolution": (32, 32),
            "early_downsampling": False,
            "global_average_pooling": 4,
        }
        self.resnet = ResNet101(**self.arch_kwargs)


class TestResNet101_160x160(unittest.TestCase, TestAbstractResNetSplitting):
    def setUp(self) -> None:
        self.arch_kwargs = {
            "n_cls": 1000,
            "in_ch": 3,
            "input_resolution": (160, 160),
            "early_downsampling": True,
            "global_average_pooling": 5,
        }
        self.resnet = ResNet101(**self.arch_kwargs)


if __name__ == "__main__":
    unittest.main()
