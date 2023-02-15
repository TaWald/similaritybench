from __future__ import annotations

import torch
from rep_trans.arch.abstract_acti_extr import AbsActiExtrArch
from rep_trans.arch.ke_architectures.base_feature_arch import BaseFeatureArch
from rep_trans.util.data_structs import ArchitectureInfo
from rep_trans.util.data_structs import BaseArchitecture
from rep_trans.util.find_architectures import get_base_arch
from torch import nn


class OutputRegularizerArch(BaseFeatureArch):
    """
    This class is supposed to be a wrapper around old architectures,
        which provide features of an intermediate layer. Also
    """

    def __init__(
        self,
        sources: list[ArchitectureInfo],
        tbt_model: ArchitectureInfo,
    ):
        super(OutputRegularizerArch, self).__init__()
        self.new_arch = get_base_arch(BaseArchitecture(tbt_model.arch_type_str))(**tbt_model.arch_kwargs)
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

        self.new_arch.train()
        self.old_archs.eval()

    def get_new_model(self):
        return self.new_arch

    def get_trainable_parameters(self):
        return self.new_arch.parameters()

    def set_trainable_gradients(self):
        """Set requires_grad of source models to false and of rest to True"""
        for src in self.old_archs.parameters():
            src.requires_grad = False

    def get_state_dict(self) -> dict:
        return self.new_arch.state_dict()

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

        out = self.new_arch(x)

        with torch.no_grad():
            other_outs = [s(x).detach() for s in self.old_archs]

        return out, other_outs
