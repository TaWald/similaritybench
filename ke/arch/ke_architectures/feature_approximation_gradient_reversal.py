from __future__ import annotations

from pathlib import Path

import torch
from ke.arch.abstract_acti_extr import AbsActiExtrArch
from ke.arch.arch_utils import create_module
from ke.arch.arch_utils import serialize_architecture_info
from ke.arch.intra_modules.abs_trained_transfer_layer import AbstractTrainedApproximation
from ke.arch.intra_modules.trained_adversarial_approximation import TrainedAdversarialApproximation
from ke.arch.intra_modules.trained_approximation import TrainedApproximation
from ke.arch.ke_architectures.base_feature_arch import BaseFeatureArch
from ke.util.data_structs import ArchitectureInfo
from ke.util.data_structs import BaseArchitecture
from ke.util.find_architectures import get_base_arch
from torch import nn


class FAGradientReversalArch(BaseFeatureArch):
    """
    This class is supposed to be a wrapper around old architectures,
        which provide features of an intermediate layer. Also
    """

    def __init__(
        self,
        old_models: list[ArchitectureInfo],
        new_model: ArchitectureInfo,
        transfer_depth: int,
        transfer_kernel_width: int,
        gradient_reversal_scale: float,
    ):
        super(FAGradientReversalArch, self).__init__()
        self.new_arch = get_base_arch(BaseArchitecture(new_model.arch_type_str))(**new_model.arch_kwargs)
        self.old_archs: nn.ModuleList[AbsActiExtrArch] = nn.ModuleList(
            [get_base_arch(BaseArchitecture(s.arch_type_str))(**s.arch_kwargs) for s in old_models]
        )

        # Create the partial old models where activations should be compared
        all_partial_olds: nn.ModuleList[nn.ModuleList] = nn.ModuleList()
        all_n_channels: list[list[int]] = [[] for _ in range(len(old_models[0].hooks))]
        sa: AbsActiExtrArch
        self.all_partial_olds_linears: nn.ModuleList = nn.ModuleList()
        for sa, s in zip(self.old_archs, old_models):
            # In order for there to be 3 positions where features are extracted there need to be 4 hooks.
            # None pos to 1. Hook = 1st Transition
            # 1. Hook to 2. Hook = 2nd Transition
            # 2. Hook to 3. Hook = 3rd Transition
            # 3. Hook to the end = final output --> Need

            this_olds_modules = nn.ModuleList()
            for cnt, h in enumerate(s.hooks):
                all_n_channels[cnt].append(h.n_channels)
                if cnt == 0:
                    this_olds_modules.append(create_module(sa, s.checkpoint, None, s.hooks[cnt]))
                else:
                    this_olds_modules.append(create_module(sa, s.checkpoint, s.hooks[cnt - 1], s.hooks[cnt]))
            this_olds_modules.append(create_module(sa, s.checkpoint, s.hooks[-1], None))
            all_partial_olds.append(this_olds_modules)
            self.all_partial_olds_linears.append(sa.get_linear_layer(sa))
        self.all_partial_olds_t: nn.ModuleList[nn.ModuleList] = nn.ModuleList(
            map(nn.ModuleList, zip(*all_partial_olds))
        )

        # Create the partial modules of the new model.
        self.all_partial_new_parts: nn.ModuleList = nn.ModuleList()
        all_n_new_channels = []
        for cnt, h in enumerate(new_model.hooks):
            all_n_new_channels.append(h.n_channels)
            if cnt == 0:
                self.all_partial_new_parts.append(create_module(self.new_arch, None, None, new_model.hooks[cnt]))
            else:
                self.all_partial_new_parts.append(
                    create_module(self.new_arch, None, new_model.hooks[cnt - 1], new_model.hooks[cnt])
                )
        self.all_partial_new_parts.append(create_module(self.new_arch, None, new_model.hooks[-1], None))

        # In the adversarial setting the new model is trying to approximate the old ones.
        #   Hence we new is passed to the TrainedAdversarialApproximation.
        self.all_transfer_modules: nn.ModuleList[AbstractTrainedApproximation] = nn.ModuleList(
            [
                TrainedAdversarialApproximation(
                    new_model_n_ch, old_model_n_ch, transfer_depth, transfer_kernel_width, gradient_reversal_scale
                )
                for new_model_n_ch, old_model_n_ch in zip(all_n_new_channels, all_n_channels)
            ]
        )
        self.transfer_module_infos = [
            {
                "trained_approx_type": TrainedApproximation.__name__,
                "source_channels": new_n_ch,
                "target_channels": old_n_ch,
                "transfer_depth": transfer_depth,
                "transfer_kernel": transfer_kernel_width,
                "gradient_reversal_scale": gradient_reversal_scale,
                "hook_architecture_index": sh.architecture_index,
                "architecture_infos": [serialize_architecture_info(s) for s in old_models],
                "count": cnt,
            }
            for cnt, (new_n_ch, old_n_ch, sh) in enumerate(
                zip(all_n_new_channels, all_n_channels, old_models[0].hooks)
            )
        ]

        self.linear_layer: nn.Module = self.new_arch.get_linear_layer(self.new_arch)
        # Disregard the gradients of the source models
        self.set_trainable_gradients()

    def train(self, mode: bool = True):
        self.training = mode
        # New
        self.new_arch.train()
        self.all_partial_new_parts.train()
        self.linear_layer.train()
        # Transfer
        self.all_transfer_modules.train()
        # Old
        self.all_partial_olds_t.eval()
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
        for p in self.all_transfer_modules.parameters():
            all_params.append(p)
        for p in self.linear_layer.parameters():
            all_params.append(p)
        return all_params

    def set_trainable_gradients(self):
        """Set requires_grad of old models to false and of rest to True"""
        # Disable for old models
        for p in self.all_partial_olds_t.parameters():
            p.requires_grad = False
        # Enable transfer params for target model to True
        for p in self.all_partial_new_parts.parameters():
            p.requires_grad = True
        # Enable transfer parameters
        for p in self.all_transfer_modules.parameters():
            p.requires_grad = True

    def get_state_dict(self) -> dict:
        return self.new_arch.state_dict()

    def get_approx_state_dict(self) -> list[tuple[dict, dict]]:
        return [(inf, mod.state_dict()) for inf, mod in zip(self.transfer_module_infos, self.all_transfer_modules)]

    def forward(self, x) -> tuple[list[torch.Tensor], list[torch.Tensor], torch.Tensor, torch.Tensor]:
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

        list_approx_inter: list[torch.Tensor] = []  # One List at each trans position (optionally one for each model)
        list_old_inter: list[torch.Tensor] = []  # one at each transfer position

        current_old: list[torch.Tensor] = [x for _ in range(len(self.all_partial_olds_t[0]))]
        cur_new: torch.Tensor = x

        for cur_partial_old, cur_transfer, cur_partial_new in zip(
            self.all_partial_olds_t[:-1], self.all_transfer_modules, self.all_partial_new_parts[:-1]
        ):
            with torch.no_grad():
                current_old = [cpo.forward(cuo) for cuo, cpo in zip(current_old, cur_partial_old)]
            cur_new = cur_partial_new(cur_new)
            cur_approx: list[torch.Tensor] = cur_transfer(cur_new)  # N_Existing_Models x Batch x Ch ...
            list_old_inter.append(torch.stack(current_old, dim=0))  # N_Existing_Models x Batch x Ch
            list_approx_inter.append(torch.stack(cur_approx, dim=0))
        cur_new = self.all_partial_new_parts[-1](cur_new)
        with torch.no_grad():
            current_old = [pa(c) for c, pa in zip(current_old, self.all_partial_olds_t[-1])]

        new_out = self.linear_layer(cur_new)
        with torch.no_grad():
            old_outs = torch.stack(
                [lins(app_logit) for app_logit, lins in zip(current_old, self.all_partial_olds_linears)], dim=0
            )

        return list_approx_inter, list_old_inter, old_outs, new_out
