import torch as t
from ke.losses.output_losses.ke_abs_out_loss import KEAbstractOutputLoss


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

    gt = one_hot_groundtruth(groundtruth, prediction.shape[2], device=prediction.device, dtype=prediction.dtype)
    false_class = (1.0 - gt).type(t.bool)  # Batch x N_Classes

    # K represents the total number of models in the ensemble
    negative_probs = filter_true_class_from_predictions(prediction, false_class)
    joint_prediction = t.transpose(negative_probs, dim0=1, dim1=0)  # Batch x K x N_Classes-1

    normed_predictions = joint_prediction / t.linalg.vector_norm(joint_prediction, ord=2, dim=-1, keepdim=True)
    mat = t.matmul(normed_predictions, t.transpose(normed_predictions, dim0=1, dim1=2))  # Batch x K x K

    # Prepare offset for later use
    det_offset = (
        det_offset
        * t.eye(n=joint_prediction.shape[1], m=joint_prediction.shape[1], device=prediction.device)[None, ...]
    )  # 1 x K x K
    robust_mat = mat + det_offset

    # Det represents Volume² is high when orthogonal --> promote high values.
    log_det = t.logdet(robust_mat)
    if not verbose:
        return -log_det  # Batch x 1
    else:
        return -log_det, robust_mat, normed_predictions, negative_probs


def ensemble_entropy(predictions: t.Tensor):
    """Calculates the entropy of the ensemble
    :param predictions: K x Batch x Classes with softmax probabilities
    """
    log_offset = 1e-20
    mean_ensemble_pred_probs = t.mean(predictions, dim=0)  # Batch x N_Classes
    # Want to maximize entropy to make as uncertain as possible (omitting - sign for entropy)
    ens_entropy = mean_ensemble_pred_probs * t.log(mean_ensemble_pred_probs + log_offset)  # Batch x N_Classes
    return ens_entropy


class AdaptiveDiversityPromotion(KEAbstractOutputLoss):
    num_tasks = 2

    def __init__(self, n_classes: int, weight_led: float = 2.0, weight_ee: float = 0.5):
        super(AdaptiveDiversityPromotion, self).__init__(n_classes)
        self.weight_led = weight_led
        self.weight_ee = weight_ee

    def forward(
        self, logit_prediction: t.Tensor, groundtruth: t.Tensor, logit_other_predictions: list[t.Tensor]
    ) -> list[t.Tensor]:
        """
        Abstract method that is used for the forward pass

        :param logit_other_predictions: list[batch x Class Probs]
        :param groundtruth: Groundtruth (Batch) -- Value encodes class! --> One hot this.
        :param logit_prediction: Prediction of the to be trained model.
        """

        cur_prob_preds = t.softmax(logit_prediction, dim=-1)
        prev_prob_preds = t.softmax(t.stack(logit_other_predictions, dim=0), dim=-1)
        all_preds = t.concatenate([cur_prob_preds[None, ...], prev_prob_preds], dim=0)  # [K x Batch x N_Class]
        led = t.mean(nlog_ensemble_diversity(all_preds, groundtruth))
        ee = t.mean(ensemble_entropy(all_preds))

        return [led * self.weight_led, ee * self.weight_ee]
