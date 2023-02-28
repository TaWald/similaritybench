from __future__ import annotations

from pathlib import Path

import torch
from ke.arch.abstract_acti_extr import AbsActiExtrArch
from ke.arch.arch_utils import create_module
from ke.arch.arch_utils import serialize_architecture_info
from ke.arch.intra_modules.trained_adversarial_approximation import TrainedAdversarialApproximation
from ke.arch.ke_architectures.base_feature_arch import BaseFeatureArch
from ke.util.data_structs import ArchitectureInfo
from ke.util.data_structs import BaseArchitecture
from ke.util.find_architectures import get_base_arch
from torch import nn


class FeatureTransferUnuseableDownstreamArch(BaseFeatureArch):
    """
    Architecture where features of the new model are transferred to old architectures.
    These are forwarded to their output and used to calculate the CE loss.

    The transfer layer is optimized to perfectly minimize the CE loss while,
    start of the other architecture has to make sure it can't learn it properly.
    """

    def __init__(
        self,
        old_models: list[ArchitectureInfo],
        new_model: ArchitectureInfo,
        transfer_depth: int,
        transfer_kernel_width: int,
        gradient_reversal_scale: float,
    ):
        super(FeatureTransferUnuseableDownstreamArch, self).__init__()
        self.new_arch = get_base_arch(BaseArchitecture(new_model.arch_type_str))(**new_model.arch_kwargs)
        self.old_archs: nn.ModuleList[AbsActiExtrArch] = nn.ModuleList(
            [get_base_arch(BaseArchitecture(s.arch_type_str))(**s.arch_kwargs) for s in old_models]
        )

        if len(old_models[0].hooks) > 1:
            raise NotImplementedError

        # Create the partial old models where activations should be compared
        all_n_channels: list[int] = [om.hooks[0].n_channels for om in old_models]
        sa: AbsActiExtrArch
        self.all_partial_olds: nn.ModuleList = nn.ModuleList()
        self.all_partial_olds_linears: nn.ModuleList = nn.ModuleList()
        for sa, s in zip(self.old_archs, old_models):
            self.all_partial_olds.append(create_module(sa, s.checkpoint, s.hooks[-1], None))
            self.all_partial_olds_linears.append(sa.get_linear_layer(sa))

        # Create the partial modules of the new model.
        self.all_partial_new_parts: nn.ModuleList = nn.ModuleList()
        n_new_channels = new_model.hooks[0].n_channels

        self.all_partial_new_parts.append(create_module(self.new_arch, None, None, new_model.hooks[0]))
        self.all_partial_new_parts.append(create_module(self.new_arch, None, new_model.hooks[0], None))

        # In the adversarial setting the new model is trying to approximate the old ones.
        #   Hence we new is passed to the TrainedAdversarialApproximation.
        self.transfer_module = TrainedAdversarialApproximation(
            n_new_channels, all_n_channels, transfer_depth, transfer_kernel_width, gradient_reversal_scale
        )
        self.transfer_module_infos = [
            {
                "trained_approx_type": TrainedAdversarialApproximation.__name__,
                "source_channels": n_new_channels,
                "target_channels": all_n_channels,
                "transfer_depth": transfer_depth,
                "transfer_kernel": transfer_kernel_width,
                "gradient_reversal_scale": gradient_reversal_scale,
                "hook_architecture_index": old_models[0].hooks[0].architecture_index,
                "architecture_infos": [serialize_architecture_info(s) for s in old_models],
                "count": 0,
            }
        ]

        self.linear_layer: nn.Module = self.new_arch.get_linear_layer(self.new_arch)
        # Disregard the gradients of the source models
        self.set_trainable_gradients()

    def train(self, mode: bool = True):
        self.training = mode

        # New stuff
        self.new_arch.train()
        self.all_partial_new_parts.train()
        self.linear_layer.train()

        # Old stuff
        self.old_archs.eval()
        self.all_partial_olds.eval()
        self.all_partial_olds_linears.eval()

    def get_new_model(self):
        return self.new_arch

    def load_individual_state_dicts(
        self, new_ckpt: Path | str, approx_layer_ckpts: list[Path | str], old_arch_ckpts: list[Path | str]
    ):
        """Loads all checkpoints needed for proper reconstruction of original behavior, given the right values."""
        self.new_arch.load_state_dict(torch.load(new_ckpt))
        for trans_module, trans_ckpt_path in zip(self.all_transfer_modules, approx_layer_ckpts):
            trans_module.load_state_dict(torch.load(trans_ckpt_path))
        for old_arch, old_ckpt_path in zip(self.old_archs, old_arch_ckpts):
            old_arch.load_state_dict(torch.load(old_ckpt_path))

    def get_trainable_parameters(self):
        all_params = []
        for p in self.all_partial_new_parts.parameters():
            all_params.append(p)
        # Enable transfer parameters
        for p in self.transfer_module.parameters():
            all_params.append(p)
        for p in self.linear_layer.parameters():
            all_params.append(p)
        return all_params

    def set_trainable_gradients(self):
        """Set requires_grad of old models to true and of rest to True"""
        # Enable for old models
        for p in self.all_partial_olds.parameters():
            p.requires_grad = True
        for p in self.all_partial_olds_linears.parameters():
            p.requires_grad = True
        # Enable transfer params for new model to True
        for p in self.new_arch.parameters():
            p.requires_grad = True
        for p in self.all_partial_new_parts.parameters():  # Just to make sure. (Should be covered by new_arch)
            p.requires_grad = True
        for p in self.linear_layer.parameters():  # Just to make sure. (Should be covered by new_arch)
            p.requires_grad = True
        # Enable transfer parameters
        for p in self.transfer_module.parameters():
            p.requires_grad = True

    def get_state_dict(self) -> dict:
        return self.new_arch.state_dict()

    def get_approx_state_dict(self) -> list[tuple[dict, dict]]:
        return [(self.transfer_module_infos[0], self.transfer_module.state_dict())]

    def forward(self, x) -> tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor]]:
        """
        Does forward through the different architecture blocks.
        At the intersections (where the regulariazation is supposed to happen).
        The models are extracted and then
        :returns approximation/transfer, intermediate tbt features, output logits
        """

        # Flow in this forward:
        # Create a normal input for all the old architectures (current_old) and one for the new (cur_new)
        # Go through each block to the first "break" where transfer happens (for cur_par..., cur_trans...,)
        #

        cur_new: torch.Tensor = x

        new_inter = self.all_partial_new_parts[0](cur_new)
        new_final = self.all_partial_new_parts[-1](new_inter)
        new_out = self.linear_layer(new_final)
        # Transfer to new ones.
        new_transfer = self.transfer_module(new_inter)
        old_transferred_outs = []
        for new_tr, old_part, old_lin in zip(new_transfer, self.all_partial_olds, self.all_partial_olds_linears):
            old_transferred_outs.append(old_lin(old_part(new_tr)))

        old_original_outs = []
        with torch.no_grad():
            for o in self.old_archs:
                old_original_outs.append(o(x))

        return new_out, old_transferred_outs, old_original_outs
