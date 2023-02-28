from __future__ import annotations

import torch
from ke.arch.ke_architectures.feature_approximation import FAArch
from torch import nn


class FASubtractionArch(FAArch):
    """
    This class is supposed to be a wrapper around old architectures,
        which provide features of an intermediate layer. Also
    """

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
            tmp_current = [torch.concat(current, dim=1)]
            # Subtraction only makes sense when all are
            approx: torch.Tensor = cur_transfer(tmp_current)[0]  # N_Existing_Models x Batch x Ch ...
            cur_true = cur_partial_tbt(cur_true)
            list_true_inter.append(torch.unsqueeze(cur_true, dim=0))  # 1 x Batch x Ch
            list_approx_inter.append(torch.unsqueeze(approx, dim=0))
            cur_true = cur_true - approx  # THIS IS THE CRUCIAL PART HERE! Removal of approximated features!
        cur_true = self.all_partial_new_modules[-1](cur_true)
        approx_logits = [pa(c) for c, pa in zip(current, self.all_partial_old_models_t[-1])]

        out = self.linear_layer(cur_true)
        approx_outs = torch.stack(
            [lins(app_logit) for app_logit, lins in zip(approx_logits, self.all_partial_old_models_linears)], dim=0
        )

        return list_approx_inter, list_true_inter, approx_outs, out
