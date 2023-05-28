from __future__ import annotations

from pathlib import Path

import torch
from ke.arch.abstract_acti_extr import AbsActiExtrArch
from ke.arch.arch_utils import create_module
from ke.arch.arch_utils import serialize_architecture_info
from ke.arch.intra_modules.trained_adversarial_approximation import TrainedAdversarialApproximation
from ke.arch.intra_modules.trained_approximation import TrainedApproximation, NotAligned
from ke.arch.ke_architectures.base_feature_arch import BaseFeatureArch
from ke.util.data_structs import ArchitectureInfo
from ke.util.data_structs import BaseArchitecture
from ke.util.find_architectures import get_base_arch
from torch import nn


class FAArch(BaseFeatureArch):
    """
    This class is supposed to be a wrapper around old architectures,
        which provide features of an intermediate layer. Also
    """

    def __init__(
        self,
        old_model_info: list[ArchitectureInfo],
        new_model_info: ArchitectureInfo,
        aggregate_old_reps: bool,
        transfer_depth: int,
        transfer_kernel_width: int,
    ):
        super(FAArch, self).__init__()
        self.aggregate_old_reps = aggregate_old_reps
        self.new_arch = get_base_arch(BaseArchitecture(new_model_info.arch_type_str))(**new_model_info.arch_kwargs)
        self.old_archs: nn.ModuleList[AbsActiExtrArch] = nn.ModuleList(
            [get_base_arch(BaseArchitecture(s.arch_type_str))(**s.arch_kwargs) for s in old_model_info]
        )

        all_partial_old_models: nn.ModuleList[nn.ModuleList] = nn.ModuleList()
        all_n_channels: list[list[int]] = [[] for _ in range(len(old_model_info[0].hooks))]
        sa: AbsActiExtrArch
        self.all_partial_old_models_linears: nn.ModuleList = nn.ModuleList()
        for sa, s in zip(self.old_archs, old_model_info):
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
            all_partial_old_models.append(this_olds_modules)
            self.all_partial_old_models_linears.append(sa.get_linear_layer(sa))
        self.all_partial_old_models_t: nn.ModuleList[nn.ModuleList] = nn.ModuleList(
            map(nn.ModuleList, zip(*all_partial_old_models))
        )

        if self.aggregate_old_reps:
            all_n_channels = [[sum(l)] for l in all_n_channels]
        self.all_partial_new_modules: nn.ModuleList = nn.ModuleList()
        all_n_tbt_channels = []
        for cnt, h in enumerate(new_model_info.hooks):
            all_n_tbt_channels.append(h.n_channels)
            if cnt == 0:
                self.all_partial_new_modules.append(
                    create_module(self.new_arch, None, None, new_model_info.hooks[cnt])
                )
            else:
                self.all_partial_new_modules.append(
                    create_module(self.new_arch, None, new_model_info.hooks[cnt - 1], new_model_info.hooks[cnt])
                )
        self.all_partial_new_modules.append(create_module(self.new_arch, None, new_model_info.hooks[-1], None))

        self.all_transfer_modules: nn.ModuleList[TrainedApproximation|NotAligned]
        if transfer_depth == 0:
            self.all_transfer_modules = nn.ModuleList([ NotAligned() for _ in all_n_tbt_channels ])
        else:
            self.all_transfer_modules = nn.ModuleList(
                [
                    TrainedApproximation(src_n_ch, tgt_n_ch, transfer_depth, transfer_kernel_width)
                    for tgt_n_ch, src_n_ch in zip(all_n_tbt_channels, all_n_channels)
                ]
            )

        self.transfer_module_infos = [
            {
                "trained_approx_type": TrainedAdversarialApproximation.__name__,
                "source_channels": src_n_ch,
                "target_channels": tgt_n_ch,
                "transfer_depth": transfer_depth,
                "transfer_kernel": transfer_kernel_width,
                "hook_architecture_index": sh.architecture_index,
                "architecture_infos": [serialize_architecture_info(s) for s in old_model_info],
                "count": cnt,
            }
            for cnt, (tgt_n_ch, src_n_ch, sh) in enumerate(
                zip(all_n_tbt_channels, all_n_channels, old_model_info[0].hooks)
            )
        ]

        self.linear_layer: nn.Module = self.new_arch.get_linear_layer(self.new_arch)
        # Disregard the gradients of the source models
        self.set_trainable_gradients()

    def train(self, mode: bool = True):
        """
        Makes sure lightning doesn't put stuff to train mode which should remain static!
        """
        self.training = mode

        self.new_arch.train()
        self.all_partial_new_modules.train()
        self.linear_layer.train()
        self.all_transfer_modules.train()

        self.old_archs.eval()
        self.all_partial_old_models_t.eval()
        self.all_partial_old_models_linears.eval()

    def get_new_model(self):
        return self.new_arch

    def load_individual_state_dicts(
        self, tbt_ckpt: Path | str, approx_layer_ckpts: list[Path | str], source_arch_ckpts: list[Path | str]
    ):
        """Loads all checkpoints needed for proper reconstruction of original behavior, given the right values."""
        self.new_arch.load_state_dict(torch.load(tbt_ckpt))
        for trans_module, trans_ckpt_path in zip(self.all_transfer_modules, approx_layer_ckpts):
            trans_module.load_state_dict(torch.load(trans_ckpt_path))
        for src_arch, src_ckpt_path in zip(self.old_archs, source_arch_ckpts):
            src_arch.load_state_dict(torch.load(src_ckpt_path))

    def get_trainable_parameters(self):
        all_params = []
        for p in self.all_partial_new_modules.parameters():
            all_params.append(p)
        # Enable transfer parameters
        for p in self.all_transfer_modules.parameters():
            all_params.append(p)
        for p in self.linear_layer.parameters():
            all_params.append(p)
        return all_params

    def set_trainable_gradients(self):
        """Set requires_grad of source models to false and of rest to True"""
        # Disable for source models
        for p in self.all_partial_old_models_t.parameters():
            p.requires_grad = False
        # Enable transfer params for target model to True
        for p in self.all_partial_new_modules.parameters():
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
        # Create a normal input for all the source architectures (current) and one for the target (cur_true)
        # Go through each block to the first "break" where transfer happens (for cur_par..., cur_trans...,)
        #

        list_approx_inter: list[torch.Tensor] = []  # One List at each trans position (optionally one for each model)
        list_true_inter: list[torch.Tensor] = []  # one at each transfer position

        current: list[torch.Tensor] = [x for _ in range(len(self.all_partial_old_models_t[0]))]
        cur_true: torch.Tensor = x
        current_partials: nn.ModuleList

        for cur_partial_source, cur_transfer, cur_partial_tbt in zip(
            self.all_partial_old_models_t[:-1], self.all_transfer_modules, self.all_partial_new_modules[:-1]
        ):
            with torch.no_grad():
                current = [cp.forward(cur) for cur, cp in zip(current, cur_partial_source)]
                tmp_current = current
                if self.aggregate_old_reps:
                    tmp_current = [torch.concat(current, dim=1)]
            approx: torch.Tensor = torch.stack(cur_transfer(tmp_current), dim=0)  # N_Existing_Models x Batch x Ch ...
            cur_true = cur_partial_tbt(cur_true)
            list_true_inter.append(torch.unsqueeze(cur_true, dim=0))  # 1 x Batch x ... !
            list_approx_inter.append(approx)
        cur_true = self.all_partial_new_modules[-1](cur_true)
        approx_logits = [pa(c) for c, pa in zip(current, self.all_partial_old_models_t[-1])]

        out = self.linear_layer(cur_true)
        with torch.no_grad():
            approx_outs = torch.stack(
                [lins(app_logit) for app_logit, lins in zip(approx_logits, self.all_partial_old_models_linears)],
                dim=0,
            )

        return list_approx_inter, list_true_inter, approx_outs, out
