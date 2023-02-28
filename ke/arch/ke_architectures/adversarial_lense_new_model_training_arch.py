from __future__ import annotations

import torch
from ke.arch.abstract_acti_extr import AbsActiExtrArch
from ke.arch.ke_architectures.base_feature_arch import BaseFeatureArch
from ke.arch.lense_architecture import UNetLense
from ke.util.data_structs import ArchitectureInfo
from ke.util.data_structs import BaseArchitecture
from ke.util.find_architectures import get_base_arch
from torch import nn


class AdversarialLenseNewModelTrainingArch(BaseFeatureArch):
    """
    This class is supposed to be a wrapper around old architectures,
        which provide features of an intermediate layer. Also
    """

    def __init__(self, new_model: ArchitectureInfo, old_models: list[ArchitectureInfo], lense: UNetLense):
        super(AdversarialLenseNewModelTrainingArch, self).__init__()
        self.old_archs: nn.ModuleList[AbsActiExtrArch] = nn.ModuleList(
            [get_base_arch(BaseArchitecture(s.arch_type_str))(**s.arch_kwargs) for s in old_models]
        )
        self.new_arch = get_base_arch(BaseArchitecture(new_model.arch_type_str))(**new_model.arch_kwargs)
        self.lense = lense
        for cnt, src in enumerate(old_models):
            state_dict = torch.load(src.checkpoint)
            self.old_archs[cnt].load_state_dict(state_dict)

    def train(self, mode: bool = True):
        self.training = mode

        self.new_arch.train()
        self.lense.eval()
        self.old_archs.eval()

    def get_new_model(self):
        return self.new_arch

    def get_trainable_parameters(self):
        return self.new_arch.parameters()

    def eval_forward(self, x):
        with torch.no_grad():
            other_outs = [s(x).detach() for s in self.old_archs]
            new_out = self.new_arch(x)
        return new_out, other_outs

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
            x_post_lense = self.lense(x)
            # rand_noise = torch.rand(size=(x.shape[0], 1,1,1))
            # x_post_lense = x_post_lense * rand_noise + x * (1-rand_noise)
            other_outs = [s(x).detach() for s in self.old_archs]

        new_out = self.new_arch(x_post_lense)
        return new_out, other_outs
