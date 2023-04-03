from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict

import pytorch_lightning as pl
import torch
from ke.arch.arch_loading import load_model_from_info_file
from ke.metrics.ke_metrics import multi_output_metrics
from ke.metrics.ke_metrics import single_output_metrics
from ke.util.data_structs import FirstModelInfo
from ke.util.load_own_objects import load_temperature_from_info
from torch import nn
from torch.optim.lr_scheduler import _LRScheduler as LRScheduler  # noqa


class EvaluationLightningModule(pl.LightningModule):
    """
    Evaluation class that can evaluate one or multiple models at once.
    It does multiple forward passes through the single models which increases GPU util and decreases CPU load.
    Results can then be extracted from the dict containing model id and the metric dict.
    """

    def __init__(self, infos: list[FirstModelInfo], arch_name: str, dataset_name: str):
        super().__init__()
        architectures = [load_model_from_info_file(info, load_ckpt=True) for info in infos]

        self.infos = infos
        self.dataset_name = dataset_name
        self.arch_name = arch_name
        self.models = nn.ModuleList(architectures)
        self._calibrated = False
        self.calibration_temperatures: list[float] = [load_temperature_from_info(info) for info in infos]

        # For the final validation epoch we want to aggregate all activation maps and approximations
        # to calculate the metrics in a less noisy manner.
        self.outputs: torch.Tensor | None = None
        self.gts: torch.Tensor | None = None

        self.max_data_points = 3e8
        self.all_single_metrics: dict[int, dict] = {}
        self.all_ensemble_metrics: dict[int, dict] = {}
        self.all_calib_ensemble_metrics: dict[int, dict] = {}

    def forward(self, x):
        return [m(x) for m in self.models]

    @contextmanager
    def calibration_mode(self, is_calibrated):
        self._calibrated = is_calibrated
        try:
            yield
        finally:
            self._calibrated = False

    def on_validation_start(self) -> None:
        """
        Empty potential remaining results from before.
        """
        self.all_single_metrics = {}
        self.all_ensemble_metrics = {}
        self.outputs = None
        self.gts = None

    def get_outputs(self) -> dict[str, torch.Tensor]:
        return {"outputs": self.outputs.detach().cpu().numpy(), "groundtruths": self.gts.detach().cpu().numpy()}

    def calibrate_outputs(self) -> None:
        self.outputs = (
            self.outputs / torch.tensor(self.calibration_temperatures, device=self.outputs.device)[:, None, None]
        )

    def on_validation_end(self) -> None:
        for i in range(self.outputs.shape[0]):
            self.all_single_metrics[i] = asdict(
                single_output_metrics(new_output=self.outputs[i], groundtruth=self.gts)
            )
            if i > 0:
                self.all_ensemble_metrics[i] = asdict(
                    multi_output_metrics(
                        new_output=self.outputs[i],
                        old_outputs=self.outputs[:i],
                        groundtruth=self.gts,
                        dataset=self.dataset_name,
                        architecture=self.arch_name,
                    )
                )
        if self._calibrated:
            self.calibrate_outputs()
            for i in range(self.outputs.shape[0]):
                if i > 0:
                    self.all_calib_ensemble_metrics[i] = asdict(
                        multi_output_metrics(
                            new_output=self.outputs[i],
                            old_outputs=self.outputs[:i],
                            groundtruth=self.gts,
                            dataset=self.dataset_name,
                            architecture=self.arch_name,
                        )
                    )

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
                self.outputs = torch.cat([self.outputs, detached_output], dim=1)

    def validation_step(self, batch, batch_idx):
        im, gt = batch
        outputs = torch.stack(self(im), dim=0)
        self.save_validation_values(groundtruths=gt, outputs=outputs)
