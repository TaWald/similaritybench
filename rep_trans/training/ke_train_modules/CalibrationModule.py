from dataclasses import asdict

import numpy as np
import torch as t
from rep_trans.arch.abstract_acti_extr import AbsActiExtrArch
from rep_trans.metrics.ke_metrics import single_output_metrics
from torch import nn
from torch.optim import LBFGS
from torch.utils.data import DataLoader


class Calibrator:
    def __init__(self, model: AbsActiExtrArch):
        super().__init__()
        self.model: AbsActiExtrArch = model
        self.temperature: nn.Parameter = nn.Parameter(t.tensor((1,), device="cuda", dtype=t.float))
        self.ce = nn.CrossEntropyLoss()

    def get_temperatures(self) -> float:
        return float(self.temperature.cpu().detach())

    def collect_logits_and_labels(self, dataloader: DataLoader) -> tuple[t.Tensor, t.Tensor]:
        """
        Collects the logits and labels of all partial models of the EnsembleModule.
        :return List of logits tensors for each model and the corresponding labels.
        """
        with t.no_grad():
            all_labels = []
            all_logits = []
            self.model.cuda()
            self.model.eval()

            for im, label in dataloader:
                im = im.cuda()
                label = label.cuda()
                logits: t.Tensor = self.model(im)
                all_logits.append(logits)
                all_labels.append(label)

        logits = t.concatenate(all_logits, dim=0)
        labels = t.concatenate(all_labels, dim=0)
        return logits, labels

    def calibrate(self, dataloader: DataLoader):
        all_logits, all_labels = self.collect_logits_and_labels(dataloader)

        optim = LBFGS([self.temperature], lr=0.001, max_iter=1000, history_size=200)

        def eval():
            optim.zero_grad()
            loss = self.ce(all_logits.detach() / self.temperature, all_labels)
            loss.backward()
            return loss

        optim.step(eval)
        return

    def calculate_calibration_effect(self, dataloader: DataLoader) -> dict:
        """
        Calculates metrics for calibrated and uncalibrated models.

        :param dataloader: Dataloader (test or validation) for which to calculate the metrics
        """
        all_logits, all_labels = self.collect_logits_and_labels(dataloader)
        # Uncalibrated
        uncalibrated_logits = all_logits
        calibrated_logits = all_logits / self.temperature

        uncalibrated_metrics = single_output_metrics(uncalibrated_logits, all_labels)
        calibrated_metrics = single_output_metrics(calibrated_logits, all_labels)

        assert np.isclose(
            uncalibrated_metrics.accuracy, calibrated_metrics.accuracy
        ), "Accuracies should be identical before and after calib!"

        outputs = {
            "temperature": float(self.temperature.detach().cpu().numpy()),
            "calibrated": asdict(calibrated_metrics),
            "uncalibrated": asdict(uncalibrated_metrics),
        }

        return outputs
