import numpy as np
import torch


def binary_cohens_kappa(preds_a: torch.Tensor, preds_b: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """Calculates the error consistency between the model a and model b

    :param preds_a:  Values of the predicted class e.g. CIFAR10: [0 ... 9]
    :param preds_b: Values of the predicted class e.g. CIFAR10: [0 ... 9]
    :param gt:  NOT One-HOT! Instead pass argmax of the one hots!
    :return:
    """
    total_sample = preds_a.shape[0]

    observed_matches = preds_a == preds_b
    c_observed = torch.sum(observed_matches) / total_sample

    correct_preds_a = preds_a == gt
    correct_preds_b = preds_b == gt
    accuracy_a = torch.sum(correct_preds_a) / total_sample
    accuracy_b = torch.sum(correct_preds_b) / total_sample

    # Probability of both predicting the same = p_both correct and p_both_incorrect. (for the same sample)
    c_expected = (accuracy_a * accuracy_b) + ((1.0 - accuracy_a) * (1.0 - accuracy_b))

    kappa = (c_observed - c_expected) / (1.0 + 1e-4 - c_expected)

    return kappa


def class_wise_cohens_kappa(predictions_a: np.ndarray, predictions_b: np.ndarray, groundtruth: np.ndarray) -> float:
    """Calculates the error consistency between the model a and model b

    :param predictions_a:
    :param predictions_b:
    :param groundtruth:
    :return:
    """

    # confusion_matrix_a = confusion_matrix(y_true=groundtruth, y_pred=predictions_a)
    # confusion_matrix_b = confusion_matrix(y_true=groundtruth, y_pred=predictions_b)

    possible_classes = np.unique(groundtruth)
    total_result = []
    for gt_value in possible_classes:
        ids_of_that_class = np.argwhere(groundtruth == gt_value)
        num_gt_ids = len(ids_of_that_class)
        pred_a = predictions_a[ids_of_that_class]
        pred_b = predictions_b[ids_of_that_class]

        class_joint_probability = []

        for class_value in possible_classes:
            num_pred_a = np.sum(
                np.where(
                    np.equal(pred_a, class_value),
                    np.ones_like(pred_a),
                    np.zeros_like(pred_a),
                )
            )
            num_pred_b = np.sum(
                np.where(
                    np.equal(pred_b, class_value),
                    np.ones_like(pred_b),
                    np.zeros_like(pred_b),
                )
            )
            conditional_prob_a = num_pred_a / num_gt_ids
            conditional_prob_b = num_pred_b / num_gt_ids

            joint_probability = conditional_prob_a * conditional_prob_b
            class_joint_probability.append(joint_probability)
        classwise_expected_probability = sum(class_joint_probability)
        observed_probability = (
            np.sum(
                np.where(
                    np.equal(pred_a, pred_b),
                    np.ones_like(pred_a),
                    np.zeros_like(pred_b),
                )
            )
            / num_gt_ids
        )

        classwise_kappa = (observed_probability - classwise_expected_probability) / (
            (1.0 - classwise_expected_probability) + 1e-9
        )
        total_result.append(classwise_kappa)

    return float(np.mean(total_result))
