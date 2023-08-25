from __future__ import annotations

from pathlib import Path

import torch
from ke.arch.abstract_acti_extr import AbsActiExtrArch
from ke.arch.arch_utils import serialize_architecture_info
from ke.arch.intra_modules.trained_adversarial_approximation import TrainedAdversarialApproximation
from ke.arch.intra_modules.trained_approximation import NotAligned
from ke.arch.intra_modules.trained_approximation import TrainedApproximation
from ke.arch.ke_architectures.base_feature_arch import BaseFeatureArch
from ke.util.data_structs import ArchitectureInfo
from ke.util.data_structs import BaseArchitecture
from ke.util.find_architectures import get_base_arch
from torch import nn


def forward_storage_hook(hook_index: int, hook_storage: dict[int : torch.Tensor]):
    """Saves the hook outputs in the storage for later use."""

    def hook(module, input, output):
        hook_storage[hook_index] = output

    return hook


def create_transfer_v2(
    transfer_depth: int, transfer_kernel: int, target_channels: list[int], src_channels: list[list[int]]
):
    """Creates as many transfers as there are target channels"""

    all_transfer_modules: TrainedApproximation | NotAligned
    if transfer_depth == 0:
        all_transfer_modules = nn.ModuleList([NotAligned() for _ in target_channels])
    else:
        all_transfer_modules = nn.ModuleList(
            [
                TrainedApproximation(src_n_ch, tgt_n_ch, transfer_depth, transfer_kernel)
                for tgt_n_ch, src_n_ch in zip(target_channels, src_channels)
            ]
        )
    return all_transfer_modules


def initialize_arch_from_arch_info(arch_info: ArchitectureInfo, load_checkpoint: bool = False):
    """Initializes and loads the architecture from arch_info"""
    arch = get_base_arch(BaseArchitecture(arch_info.arch_type_str))(**arch_info.arch_kwargs)
    if load_checkpoint:
        arch.load_state_dict(torch.load(arch_info.checkpoint))
    return arch


class FAArchV2(BaseFeatureArch):
    """
    This class is supposed to be a wrapper around old architectures,
        which provide features of an intermediate layer. Also
    """

    def __init__(
        self,
        model_infos_old: list[ArchitectureInfo],
        model_info_new: ArchitectureInfo,
        aggregate_old_reps: bool,
        transfer_depth: int,
        transfer_kernel_width: int,
    ):
        super(FAArchV2, self).__init__()

        self.model_info_new = model_info_new
        self.model_infos_old = model_infos_old
        self.aggregate_reps = aggregate_old_reps
        self.transfer_depth = transfer_depth
        self.transfer_kernel_width = transfer_kernel_width

        # Create the new model & partial model
        self.arch_new: AbsActiExtrArch = initialize_arch_from_arch_info(model_info_new)
        all_new_channels: list[int] = [h.n_channels for h in model_info_new.hooks]
        self.intermediate_rep_storage_new: dict[int : torch.Tensor] = {}

        # Create the old models & partial models.
        self.archs_old: nn.ModuleList[str:AbsActiExtrArch] = nn.ModuleList(
            [initialize_arch_from_arch_info(oi, True) for oi in model_infos_old]
        )
        self.intermediate_rep_storage_old: dict[int : dict[int : torch.Tensor]] = {
            i: {} for i in range(len(self.archs_old))
        }

        # Contains the number of channels with indexes corresponding to the hook index
        all_n_channels: list[list[int]] = [[] for _ in model_infos_old[0].hooks]
        for oi in model_infos_old:
            for cnt, h in enumerate(oi.hooks):
                if cnt == 0:
                    all_n_channels[cnt] = [h.n_channels]
                else:
                    all_n_channels[cnt].append(h.n_channels)
        if self.aggregate_reps:
            all_n_channels = [[sum(v)] for v in all_n_channels]

        # Create the transfer module
        self.all_transfer_modules: nn.ModuleList[TrainedApproximation | NotAligned]
        self.all_transfer_modules = create_transfer_v2(
            transfer_depth, transfer_kernel_width, all_new_channels, all_n_channels
        )
        self.transfer_module_infos = [
            {
                "trained_approx_type": TrainedAdversarialApproximation.__name__,
                "source_channels": src_n_ch,
                "target_channels": tgt_n_ch,
                "transfer_depth": transfer_depth,
                "transfer_kernel": transfer_kernel_width,
                "hook_architecture_index": sh.architecture_index,
                "architecture_infos": [serialize_architecture_info(s) for s in model_infos_old],
                "count": cnt,
            }
            for cnt, (tgt_n_ch, src_n_ch, sh) in enumerate(
                zip(all_new_channels, all_n_channels, model_infos_old[0].hooks)
            )
        ]
        # Disregard the gradients of the source models
        self.set_require_grad()

        self.all_new_handles = self.register_new_model_hooks()
        self.all_old_handles = self.register_old_model_hooks()

    def register_new_model_hooks(self):
        """Registers the storage hooks in the new model"""
        all_handles = []
        named_modules_new = dict(self.arch_new.named_modules())
        for cnt, h in enumerate(self.model_info_new.hooks):
            try:
                module_of_interest = named_modules_new[".".join(h.keys)]
            except KeyError:
                try:
                    module_of_interest = named_modules_new[".".join(["module"] + h.keys)]
                except KeyError:
                    raise KeyError(f"Could not find module {h.keys} in new model.")
            all_handles.append(
                module_of_interest.register_forward_hook(forward_storage_hook(cnt, self.intermediate_rep_storage_new))
            )
        return all_handles

    def register_old_model_hooks(self):
        """Registers the storage hooks of the old models"""
        all_handles = []

        for cnt1, arch_old in enumerate(self.archs_old):
            named_modules_old = dict(arch_old.named_modules())
            for cnt2, h in enumerate(self.model_infos_old[cnt1].hooks):
                try:
                    module_of_interest = named_modules_old[".".join(h.keys)]
                except KeyError:
                    try:
                        module_of_interest = named_modules_old[".".join(["module"] + h.keys)]
                    except KeyError:
                        raise KeyError(f"Could not find module {h.keys} in new model.")
                all_handles.append(
                    module_of_interest.register_forward_hook(
                        forward_storage_hook(cnt2, self.intermediate_rep_storage_old[cnt1])
                    )
                )
        return all_handles

    def train(self, mode: bool = True):
        """
        Makes sure lightning doesn't put stuff to train mode which should remain static!
        """
        self.training = mode

        self.arch_new.train()
        self.all_transfer_modules.train()
        self.archs_old.eval()

    def get_new_model(self) -> AbsActiExtrArch:
        return self.arch_new

    def load_individual_state_dicts(
        self,
        tbt_ckpt: Path | str | None | dict,
        approx_layer_ckpts: list[Path | str | dict] | None,
        source_arch_ckpts: list[Path | str | dict] | None,
    ):
        """Loads all checkpoints needed for proper reconstruction of original behavior, given the right values."""
        if tbt_ckpt is not None:
            if isinstance(tbt_ckpt, (Path, str)):
                tbt_ckpt = torch.load(tbt_ckpt)
            self.new_arch.load_state_dict(tbt_ckpt)
        if approx_layer_ckpts is not None:
            for trans_module, trans_ckpt_path in zip(self.all_transfer_modules, approx_layer_ckpts):
                if isinstance(trans_ckpt_path, (Path, str)):
                    trans_ckpt_path = torch.load(trans_ckpt_path)
                trans_module.load_state_dict(trans_ckpt_path)
        if source_arch_ckpts is not None:
            for src_arch, src_ckpt_path in zip(self.old_archs, source_arch_ckpts):
                if isinstance(src_ckpt_path, (Path, str)):
                    src_ckpt_path = torch.load(src_ckpt_path)
                src_arch.load_state_dict(torch.load(src_ckpt_path))

    # This seems to be the issue? Adding multiple versions of trainable params?
    def get_trainable_parameters(self):
        all_params = []
        for p in self.arch_new.parameters():
            all_params.append(p)
        # Enable transfer parameters
        for p in self.all_transfer_modules.parameters():
            all_params.append(p)

        return all_params

    def set_require_grad(self):
        """Set requires_grad of source models to false and of rest to True"""
        # Disable for source models
        for p in self.archs_old.parameters():
            p.requires_grad = False

        for p in self.arch_new.parameters():
            p.requires_grad = True
        for p in self.all_transfer_modules.parameters():
            p.requires_grad = True

    def get_state_dict(self) -> dict:
        return self.new_arch.state_dict()

    def get_approx_state_dict(self) -> list[tuple[dict, dict]]:
        return [(inf, mod.state_dict()) for inf, mod in zip(self.transfer_module_infos, self.all_transfer_modules)]

    def simple_forward(self, x) -> torch.Tensor:
        """Does a simple forward of the partialed new model. Should be identical to self.new_model(x)"""
        return self.new_arch(x)

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

        # Forward pass and get intermediate reps of old model.
        old_outs = torch.stack([old_arch.forward(x) for old_arch in self.archs_old], dim=0)

        for idx, transfer in enumerate(self.all_transfer_modules):
            intermediate_actis_old = []
            for i in range(len(self.archs_old)):
                intermediate_actis_old.append(self.intermediate_rep_storage_old[i][idx])
            if self.aggregate_reps:
                intermediate_actis_old = [torch.concat(intermediate_actis_old, dim=1)]
            approx: torch.Tensor = torch.stack(
                transfer(intermediate_actis_old), dim=0
            )  # N_Existing_Models x Batch x Ch ...
            list_approx_inter.append(approx)

        # Forward pass & get intermediate reps of new model
        out_new = self.arch_new.forward(x)
        list_true_inter = [torch.unsqueeze(v, dim=0) for v in self.intermediate_rep_storage_new.values()]

        return list_approx_inter, list_true_inter, old_outs, out_new
