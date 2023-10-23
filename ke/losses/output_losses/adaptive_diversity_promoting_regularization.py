import torch as t
from torch import nn


def cos_sim_ensemble_diversity(predicted: t.Tensor, prev_predicted: t.Tensor, groundtruth: t.Tensor):
    """
    Calculates the log of the Ensemble diversity which is the determinant of
    each sample's probabilities for the not-true class

    :param predicted: Softmax Probabilties of the currently being trained model DIMS: Batch x Classes
    :param prev_predicted: Softmax probabiltieis of the already trained models  DIMS: NPrev x Batch x Classes
    :param groundtruth: tensor of Integer values (not one hot!)
    """

    gt = t.zeros((groundtruth.shape[0], predicted.shape[1]), device=predicted.device, dtype=t.float32)
    gt[t.arange(groundtruth.shape[0]), groundtruth] = 1.0  # One hot of the true class now.
    false_class = (1.0 - gt).type(t.bool)  # Batch x N_Classes

    K = prev_predicted.shape[0]

    # K represents the total number of models in the ensemble
    with t.no_grad():
        # K x (Batch * N_Classes-1)
        false_probs = t.masked_select(prev_predicted.detach(), false_class[None, ...])
        negative_old_probs = t.reshape(false_probs, shape=(K, prev_predicted.shape[1], prev_predicted.shape[2] - 1))

    false_new_probs = t.masked_select(predicted, false_class)
    negative_new_probs = t.reshape(false_new_probs, shape=(predicted.shape[0], predicted.shape[1] - 1))

    cos_sim = t.sum(negative_new_probs[None, ...] * negative_old_probs, dim=-1)  # K x Batch
    return cos_sim


def one_hot_groundtruth(groundtruth: t.Tensor, n_classes: int, device: t.device, dtype=t.float32) -> t.Tensor:
    """Creates one-hot encoding of groundtruth"""
    gt = t.zeros((groundtruth.shape[0], n_classes), device=device, dtype=dtype)
    gt[t.arange(groundtruth.shape[0]), groundtruth] = 1.0  # One hot of the true class now.
    return gt


def filter_true_class_from_predictions(prediction: t.Tensor, false_prediction_mask: t.Tensor):
    """
    :param prediction: Predicted probabilities of shape [K x Batch x N_Classes]
    :param false_prediction_mask: Boolean mask indicating false prediction [Batch x N_Classes]
    :return: [K x Batch x (N_classes -1)]  The same prediction tensor just without the true class probability.
    """
    n_models, n_samples, n_classes = prediction.shape
    false_probs = t.masked_select(prediction, false_prediction_mask[None, ...])  # K x (Batch * N_Classes-1)
    negative_probs = t.reshape(false_probs, shape=(n_models, n_samples, n_classes - 1))
    return negative_probs


def nlog_ensemble_diversity(prediction: t.Tensor, groundtruth: t.Tensor, verbose: bool = False):
    """
    Calculates the log of the Ensemble diversity which is the determinant of
    each sample's probabilities for the not-true class

    :param prediction: Softmax Probabilties of all models DIMS: [K x Batch x Classes]
    :param groundtruth: tensor of Integer values (not one hot!) [Batch]
    """
    det_offset = 1e-6
    log_offset = 1e-20

    gt = one_hot_groundtruth(groundtruth, prediction.shape[2], device=prediction.device, dtype=prediction.dtype)
    false_class = (1.0 - gt).type(t.bool)  # Batch x N_Classes

    # K represents the total number of models in the ensemble
    negative_probs = filter_true_class_from_predictions(prediction, false_class)
    joint_prediction = t.transpose(negative_probs, dim0=1, dim1=0)  # Batch x K x N_Classes-1

    normed_predictions = joint_prediction / t.linalg.vector_norm(joint_prediction, ord=2, dim=-1, keepdim=True)
    mat = t.matmul(normed_predictions, t.transpose(normed_predictions, dim0=1, dim1=2))  # Batch x K x K

    # Prepare offset for later use
    mat_det_offset = (
        det_offset
        * t.eye(n=joint_prediction.shape[1], m=joint_prediction.shape[1], device=prediction.device)[None, ...]
    )  # 1 x K x K
    robust_mat = mat + mat_det_offset

    # Det represents Volume² is high when orthogonal --> promote high values.
    det = t.det(robust_mat)
    log_det = t.log(det + log_offset)
    # log_det = t.log(det)
    if not verbose:
        return log_det  # Batch x 1
    else:
        return log_det, robust_mat, normed_predictions, negative_probs


def ensemble_entropy(predictions: t.Tensor):
    """Calculates the entropy of the ensemble
    :param predictions: K x Batch x Classes with softmax probabilities
    """
    log_offset = 1e-20
    mean_ensemble_pred_probs = t.mean(predictions, dim=0)  # Batch x N_Classes
    # Want to maximize entropy to make as uncertain as possible (omitting - sign for entropy)
    ens_entropy = t.sum(
        -mean_ensemble_pred_probs * t.log(mean_ensemble_pred_probs + log_offset), dim=-1
    )  # Batch x N_Classes
    return ens_entropy


class AdaptiveDiversityPromotionV2(nn.Module):
    def __init__(self, weight_led: float = 2.0, weight_ee: float = 0.5, weight_ce: float = 1.0):
        super(AdaptiveDiversityPromotionV2, self).__init__()
        self.weight_led = weight_led
        self.weight_ee = weight_ee
        self.weight_ce = weight_ce
        self.calc_ee = True if weight_ee > 0 else False
        self.calc_led = True if weight_led > 0 else False
        self.ce = nn.CrossEntropyLoss()

    def forward(self, target: t.Tensor, logits: t.Tensor) -> t.Tensor:
        """
        Calculates the loss for the adaptive diversity promotion loss.

        :param outputs: list of model outputs
        :param target: ground truth
        :return: loss
        """
        all_logits = logits
        all_probs = t.softmax(all_logits, dim=-1)

        ce = t.mean(t.stack([self.weight_ce * self.ce(logit, target) for logit in logits]))
        ce_loss = ce * self.weight_ce
        total_loss = ce_loss
        if self.calc_led:
            led = t.mean(nlog_ensemble_diversity(all_probs, target))
            led_loss = led * self.weight_led
            total_loss -= led_loss
        if self.calc_ee:
            ee = t.mean(ensemble_entropy(all_probs))
            ee_loss = ee * self.weight_ee
            total_loss -= ee_loss

        return total_loss
