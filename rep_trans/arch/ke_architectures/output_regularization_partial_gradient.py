from __future__ import annotations

import torch
from rep_trans.arch.abstract_acti_extr import AbsActiExtrArch
from rep_trans.arch.arch_utils import create_module
from rep_trans.arch.ke_architectures.base_feature_arch import BaseFeatureArch
from rep_trans.util.data_structs import ArchitectureInfo
from rep_trans.util.data_structs import BaseArchitecture
from rep_trans.util.find_architectures import get_base_arch
from torch import nn


class OutputRegularizerPartialGradientArch(BaseFeatureArch):
    """
    This class is supposed to be a wrapper around old architectures,
        which provide features of an intermediate layer. Also
    """

    def __init__(self, sources: list[ArchitectureInfo], tbt_model: ArchitectureInfo, hook_id: int):
        super(OutputRegularizerPartialGradientArch, self).__init__()
        self.tbt_arch = get_base_arch(BaseArchitecture(tbt_model.arch_type_str))(**tbt_model.arch_kwargs)
        hook = self.tbt_arch.hooks[hook_id]
        self.first_part = create_module(self.tbt_arch, None, first_hook=None, second_hook=hook)
        self.last_part = create_module(self.tbt_arch, None, hook, None)
        self.linear = self.tbt_arch.get_linear_layer(self.tbt_arch)

        self.old_archs: nn.ModuleList[AbsActiExtrArch] = nn.ModuleList(
            [get_base_arch(BaseArchitecture(s.arch_type_str))(**s.arch_kwargs) for s in sources]
        )
        for cnt, src in enumerate(sources):
            state_dict = torch.load(src.checkpoint)
            self.old_archs[cnt].load_state_dict(state_dict)

        # Disregard the gradients of the source models
        self.set_trainable_gradients()

    def train(self, mode: bool = True):
        self.training = mode

        self.tbt_arch.train()
        self.first_part.train()
        self.last_part.train()
        self.linear.train()
        self.old_archs.eval()

    def get_new_model(self):
        return self.tbt_arch

    def get_trainable_parameters(self):
        return self.tbt_arch.parameters()

    def set_trainable_gradients(self):
        """Set requires_grad of source models to false and of rest to True"""
        for src in self.old_archs.parameters():
            src.requires_grad = False

    def get_state_dict(self) -> dict:
        return self.tbt_arch.state_dict()

    def forward(self, x) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """
        Does forward through the different architecture blocks.
        At the intersections (where the regulariazation is supposed to happen).
        The models are extracted and then
        :returns approximation/transfer, intermediate tbt features, output logits
        """

        # Flow in this forward:
        # Create a normal input for all the source architectures (current) and one for the target (cur_true)
        # Go through each block to the first "break" where transfer happens (for cur_par..., cur_trans...,)
        #
        with torch.no_grad():
            other_outs = [s(x).detach() for s in self.old_archs]

        x = self.first_part(x)
        x = self.last_part(x)
        out = self.linear(x)

        return out, other_outs

    def get_alt_trainable_parameters(self):
        """
        For the alternative backwards pass one wants to **only update the very first part** of the architecture.
        This follows the notion that if one enforces dissimilarity at the end acccuracy suffers.
        However when having some differing early reps the later ones might still allow high quality performance!
        """
        return self.first_part.parameters()
