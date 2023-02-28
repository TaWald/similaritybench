import torch
from ke.losses.representation_similarity_losses.ke_abs_loss import AbstractStaticLenseLoss
from ke.losses.representation_similarity_losses.ke_abs_loss import ReconstructionLossProto
from torch import nn


class KELenseOutputTrainLoss(nn.Module):
    def __init__(
        self,
        adversarial_loss: AbstractStaticLenseLoss,
        reconstruction_loss: ReconstructionLossProto,
        lense_adversarial_weight: float = 1.0,
        lense_reconstruction_weight: float = 1.0,
    ):
        super(KELenseOutputTrainLoss, self).__init__()
        self.softmax = nn.Softmax(dim=1)
        self.reco_loss = reconstruction_loss
        self.adv_loss = adversarial_loss

        self.lense_adv_weight: float = lense_adversarial_weight
        self.lense_reco_weight: float = lense_reconstruction_weight
        self.current_dis_weight = 0.0

        self.ce_loss = nn.CrossEntropyLoss()

    def forward(
        self,
        original_image: torch.Tensor,
        reconstructed_image: torch.Tensor,
        groundtruth: torch.Tensor,
        aug_predictions: list[torch.Tensor],
    ) -> dict:

        reco_loss = self.reco_loss.forward(original_image, reconstructed_image) * self.lense_reco_weight
        adv_loss = self.adv_loss.forward(aug_predictions, groundtruth)

        l_reco_weighted = reco_loss * self.lense_reco_weight
        l_adv_weighted = adv_loss * self.lense_adv_weight
        total_loss = l_reco_weighted + l_adv_weighted

        with torch.no_grad():
            for p in aug_predictions:
                torch.mean(torch.argmax(torch.softmax(p, dim=-1), dim=-1) == groundtruth, dtype=torch.float)
        aug_accs = torch.mean(
            groundtruth[None, ...] == torch.argmax(torch.stack(aug_predictions, dim=0), dim=2), dtype=torch.float
        )

        output = {
            "loss": total_loss,
            "loss_info": {
                "adversarial_weighted": l_adv_weighted.detach(),
                "reconstruction_weighted": l_reco_weighted.detach(),
            },
            "metrics": {
                "augmented_accuracy": aug_accs.detach(),
                "mse": adv_loss.detach(),
            },
        }

        return output

    def on_epoch_end(self, outputs: list[dict]) -> dict:
        """
        Creates better averages of returns for ke_foward.
        In the training loop it averages only losses since no metrics are calculated for that.
        In the validation loop it averages losses and the metrics
        """
        # Always calculate the loss stuff
        adv_weighted = torch.stack([o["loss_info"]["adversarial_weighted"] for o in outputs]).mean()
        rec_weighted = torch.stack([o["loss_info"]["reconstruction_weighted"] for o in outputs]).mean()
        mse = torch.stack([o["metrics"]["mse"] for o in outputs]).mean()
        loss_values = {
            "loss/adversarial": adv_weighted,
            "loss/reconstruction": rec_weighted,
            "metrics/mse": mse,
        }

        return loss_values
