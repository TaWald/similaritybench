from __future__ import annotations

import torch
from ke.arch.abstract_acti_extr import AbsActiExtrArch
from ke.arch.lense_architecture import UNetLense
from ke.util.data_structs import ArchitectureInfo
from ke.util.data_structs import BaseArchitecture
from ke.util.find_architectures import get_base_arch
from torch import nn


class AdversarialLenseTrainingArch(nn.Module):
    """
    This class is supposed to be a wrapper around old architectures,
        which provide features of an intermediate layer. Also
    """

    def __init__(self, old_models: list[ArchitectureInfo], lense: UNetLense):
        super(AdversarialLenseTrainingArch, self).__init__()
        self.old_archs: nn.ModuleList[AbsActiExtrArch] = nn.ModuleList(
            [get_base_arch(BaseArchitecture(s.arch_type_str))(**s.arch_kwargs) for s in old_models]
        )
        self.lense = lense
        for cnt, src in enumerate(old_models):
            state_dict = torch.load(src.checkpoint)
            self.old_archs[cnt].load_state_dict(state_dict)

    def train(self, mode: bool = True):
        self.training = mode

        self.lense.train()
        self.old_archs.eval()

    def get_state_dict(self) -> dict:
        return self.lense.state_dict()

    def get_trainable_parameters(self):
        return self.lense.parameters()

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
        x_post_lense = self.lense(x)

        other_outs = [s(x_post_lense) for s in self.old_archs]

        return x_post_lense, other_outs

    def clean_forward(self, x) -> list[torch.Tensor]:
        other_outs = [s(x).detach() for s in self.old_archs]

        return other_outs
