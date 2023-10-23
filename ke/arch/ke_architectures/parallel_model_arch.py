from __future__ import annotations

import torch
from ke.arch.ke_architectures.feature_approximation import create_model_instances
from ke.util.data_structs import ArchitectureInfo
from torch import nn


class ParallelModelArch(nn.Module):
    """
    This class is supposed to be a wrapper around old architectures,
        which provide features of an intermediate layer. Also
    """

    def __init__(self, model_infos: list[ArchitectureInfo]):
        super(ParallelModelArch, self).__init__()
        # Instatiante the models (if ckpts are provided they can also be loaded.)
        self.all_models: nn.ModuleList = nn.ModuleList(create_model_instances(model_infos))

    def forward(self, x) -> torch.tensor:
        """
        Does forward through the different architecture blocks.
        At the intersections (where the regulariazation is supposed to happen).
        The models are extracted and then
        :returns approximation/transfer, intermediate tbt features, output logits
        """
        outs = torch.stack([m(x) for m in self.all_models], dim=0)

        return outs
