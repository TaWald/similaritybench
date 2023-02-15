from __future__ import annotations

import pytorch_lightning as pl
import torch
from torch import nn
from torch.optim.lr_scheduler import _LRScheduler as LRScheduler  # noqa

from rep_trans.util import data_structs as ds
from rep_trans.util.data_structs import BasicTrainingInfo
from rep_trans.util.find_architectures import get_base_arch
from rep_trans.metrics.ke_metrics import multi_output_metrics, single_output_metrics


class EnsembleEvaluationLightningModule(pl.LightningModule):
    def __init__(
        self,
        infos: list[BasicTrainingInfo],
        arch_name: str,
        dataset_name: str
    ):
        super().__init__()
        architectures = [get_base_arch(ds.BaseArchitecture(info.architecture)) for info in infos]
        loaded_architectures = [arch.load_state_dict(info.path_ckpt) for arch, info in zip(architectures, infos)]

        self.infos = infos
        self.dataset_name =  dataset_name
        self.arch_name = arch_name
        self.models = nn.ModuleList(loaded_architectures)
        
        # For the final validation epoch we want to aggregate all activation maps and approximations
        # to calculate the metrics in a less noisy manner.
        self.outputs: torch.Tensor | None = None
        self.gts: torch.Tensor | None = None
        
        self.max_data_points = 3e8
        self.all_metrics: dict = {}

    def forward(self, x):
        return [m(x) for m in self.models]

    def on_validation_start(self) -> None:
        """
        Empty potential remaining results from before.
        """
        self.outputs = None
        self.gts = None

    def get_outputs(self) -> dict[str, torch.Tensor]:
        return {"outputs": self.outputs.detach().cpu().numpy(), "groundtruths": self.gts.detach().cpu().numpy()}

    def on_validation_end(self) -> None:
        for i in range(self.outputs.shape[0]):
            if i == 0:
                self.all_metrics[0] = single_output_metrics(new_output=self.outputs[i], groundtruth=self.gts)
            else:
                self.all_metrics[i] = multi_output_metrics(
                    new_output=self.outputs[i],
                    old_outputs=self.outputs[:i],
                    groundtruth=self.gts,
                    dataset=ds.Dataset(self.dataset_name),
                    architecture=ds.BaseArchitecture(self.arch_name))
                

    def save_validation_values(
        self,
        groundtruths: torch.Tensor | None,
        outputs: torch.Tensor | None,
    ):
        # Save Groundtruths:
        if groundtruths is not None:
            if self.gts is None:
                self.gts = groundtruths
            else:
                self.gts = torch.concatenate([self.gts, groundtruths], dim=0)

        # Aggregate new models outputs
        if outputs is not None:
            detached_output = outputs.detach()
            if self.outputs is None:
                self.outputs = detached_output
            else:
                self.outputs = torch.cat([self.outputs, detached_output], dim=0)

    def validation_step(self, batch, batch_idx):
        im, gt = batch
        outputs = torch.stack(self(im), dim=0)
        self.save_validation_values(groundtruths=gt, outputs=outputs)
        