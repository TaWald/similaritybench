import pickle
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from rep_trans.losses.output_losses.adaptive_diversity_promoting_regularization import ensemble_entropy
from rep_trans.losses.output_losses.adaptive_diversity_promoting_regularization import (
    filter_true_class_from_predictions,
)
from rep_trans.losses.output_losses.adaptive_diversity_promoting_regularization import nlog_ensemble_diversity


def read_single_sample_file():
    res = Path(__file__).parent / "adp_original_behavior" / "single_sample.pckl"
    with open(res, "rb") as f:
        res_dict: dict = pickle.load(f, encoding="latin1")
    return res_dict


def read_file(n_model: int, n_cls: int, n_samples: int):
    res = (
        Path(__file__).parent
        / "adp_original_behavior"
        / f"data_NModels_{n_model}_NClasses_{n_cls}_NSamples{n_samples}.pckl"
    )

    with open(res, "rb") as f:
        res_dict: dict = pickle.load(f, encoding="latin1")
    return res_dict


class TestADPReimplementation(unittest.TestCase):
    """
    Values taken from feeding it to the original code.
    Used probabilities and one_hot_labels as inputs with respective passed number of classes and models.
    This is code where its loaded from file in order to not install old garbage cpu tensorflow and python2.7.

    """

    """
    Keys of Saved pickle files.

    - log_det
    - ensemble_entropy
    - cross_entropy
    - joint_loss
    - lamda
    - log_det_lamda
    - n_classes
    - n_models
    """

    """
    Keys of explicit file
    - log_det
    - ensemble_entropy
    - filtered_preds
    - normed_filtered_preds
    - lamda
    - log_det_lamda
    - n_classes
    - n_models
    - log_det_matrix
    """

    def test_filtering_works(self):
        """
        Test that the filtered values are actually the probabilities one expects.
        """
        ensemble_probs = np.reshape(np.array([[0.7, 0.2, 0.1], [0.2, 0.6, 0.2]]), (2, 1, 3))
        expected_negative_probs = ensemble_probs[..., 1:]
        labels = np.reshape(np.array([0]), newshape=(1,))
        res = nlog_ensemble_diversity(torch.from_numpy(ensemble_probs), torch.from_numpy(labels), verbose=True)
        _, _, _, negative_probs = res
        actual_negative_probs = negative_probs.numpy()
        self.assertTrue(np.all(actual_negative_probs == expected_negative_probs))

    def test_normalization_works_as_intended(self):
        """
        Test the normalization is as expected.
        """
        ensemble_probs = np.reshape(np.array([[0.7, 0.2, 0.1], [0.2, 0.6, 0.2]]), (2, 1, 3))
        expected_negative_probs = ensemble_probs[..., 1:]
        exp_neg_norm_probs = expected_negative_probs / np.sqrt(
            np.sum(expected_negative_probs**2, axis=-1, keepdims=True)
        )
        labels = np.array([0])
        res = nlog_ensemble_diversity(torch.from_numpy(ensemble_probs), torch.from_numpy(labels), verbose=True)
        _, _, normed_predictions, _ = res
        normed_predictions = np.transpose(normed_predictions.numpy(), axes=(1, 0, 2))
        same_values = np.all(normed_predictions == exp_neg_norm_probs)
        # Sqrt can be omitted for norm as sum should be 1 (if deviates deviation is not norm difference)
        all_normed = np.all(np.isclose(np.sum(normed_predictions**2, axis=-1), 1))
        self.assertTrue(same_values, "Normed vector values are not as expected")
        self.assertTrue(all_normed, "Euclidean norm does not add up to 1. as would be expected")

    def test_filtering_true_class_if_true_class_is_max_val(self):
        """
        Test the submethod `filter_true_class_from_predictions` used in `nlog_ensemble_diversity`.
        Therefore making sure this is not an issue. (Redundancy to `test_filtering_works`)
        """

        ensemble_logits = torch.from_numpy(np.random.rand(1, 100, 10))  # 1 Model 100 Samples 10 Classes
        ensemble_probs = F.softmax(ensemble_logits, dim=-1)  # Now is normed
        label = torch.argmax(ensemble_probs, dim=-1)

        false_mask = torch.ones_like(ensemble_probs)
        false_mask[:, np.arange(100), label] = 0.0
        false_mask = false_mask[0]  # Get rid of first dimension
        pos_vals = torch.sum(false_mask, dim=1).numpy()
        self.assertTrue(np.all(pos_vals == 9), "Expect all masks to have one class as true class.")
        false_mask = false_mask.to(torch.bool)

        filtered_probs = filter_true_class_from_predictions(ensemble_probs, false_mask)
        sample_sum = torch.sum(filtered_probs, dim=(0, 2))
        expected_sum = 1 - torch.max(ensemble_probs, dim=-1).values

        self.assertTrue(torch.all(torch.isclose(sample_sum, expected_sum)))

    def test_filtering_identical_to_original(self):
        """
        Assures the matrix before logdet calculation is identical
        Test if the normed probabilities of the non true classes are identical
        """
        res_dict: dict = read_single_sample_file()

        # Ensemble probs are list of [Batch x N_CLasses]
        ensemble_probs = torch.from_numpy(np.stack(res_dict["ensemble_probs"], axis=0))  # [K x Batch x N_Classes]
        orig_fltr_preds = np.transpose(res_dict["filtered_preds"], axes=(1, 0, 2))

        non_one_hot_labels = torch.from_numpy(res_dict["labels"]).to(torch.int64)
        (
            _,
            _,
            _,
            filtered_preds,
        ) = nlog_ensemble_diversity(prediction=ensemble_probs, groundtruth=non_one_hot_labels, verbose=True)
        filtered_preds = filtered_preds.numpy()
        prediction_close = np.isclose(orig_fltr_preds, filtered_preds)

        self.assertTrue(np.all(prediction_close))
        print("Whoop.")

    def test_norm_identical_to_original(self):
        """
        Assures the matrix before logdet calculation is identical
        Test if the normed probabilities of the non true classes are identical
        """
        res_dict: dict = read_single_sample_file()

        # Ensemble probs are list of [Batch x N_CLasses]
        ensemble_probs = torch.from_numpy(np.stack(res_dict["ensemble_probs"], axis=0))  # [K x Batch x N_Classes]
        orig_normed_fltr_preds = res_dict["normed_filtered_preds"]

        non_one_hot_labels = torch.from_numpy(res_dict["labels"]).to(torch.int64)
        (
            _,
            _,
            normed_filtered_preds,
            _,
        ) = nlog_ensemble_diversity(prediction=ensemble_probs, groundtruth=non_one_hot_labels, verbose=True)
        filtered_preds = normed_filtered_preds.numpy()
        prediction_close = np.isclose(orig_normed_fltr_preds, filtered_preds)

        self.assertTrue(np.all(prediction_close))

    def test_matrix_identical_to_original(self):
        """
        Assures the matrix before logdet calculation is identical
        Test if the normed probabilities of the non true classes are identical
        """
        res_dict: dict = read_single_sample_file()

        # Ensemble probs are list of [Batch x N_CLasses]
        ensemble_probs = torch.from_numpy(np.stack(res_dict["ensemble_probs"], axis=0))  # [K x Batch x N_Classes]
        orig_matrix = res_dict["log_det_matrix"]

        non_one_hot_labels = torch.from_numpy(res_dict["labels"]).to(torch.int64)
        (
            _,
            mat,
            _,
            _,
        ) = nlog_ensemble_diversity(prediction=ensemble_probs, groundtruth=non_one_hot_labels, verbose=True)
        mat = mat.numpy()
        prediction_close = np.isclose(orig_matrix, mat)

        self.assertTrue(np.all(prediction_close))

    def test_logdet_identical_to_original(self):
        """
        Assures the matrix before logdet calculation is identical
        Test if the normed probabilities of the non true classes are identical
        """
        res_dict: dict = read_single_sample_file()

        # Ensemble probs are list of [Batch x N_CLasses]
        ensemble_probs = torch.from_numpy(np.stack(res_dict["ensemble_probs"], axis=0))  # [K x Batch x N_Classes]
        non_one_hot_labels = torch.from_numpy(res_dict["labels"]).to(torch.int64)
        orig_log_det = res_dict["log_det"]

        (
            ld,
            _,
            _,
            _,
        ) = nlog_ensemble_diversity(prediction=ensemble_probs, groundtruth=non_one_hot_labels, verbose=True)
        prediction_close = np.isclose(orig_log_det, -ld.numpy())
        self.assertTrue(np.all(prediction_close))

    def test_ensemble_entropy(self):
        """
        Assures the matrix before logdet calculation is identical
        Test if the normed probabilities of the non true classes are identical
        """
        res_dict: dict = read_single_sample_file()

        # Ensemble probs are list of [Batch x N_CLasses]
        ensemble_probs = torch.from_numpy(np.stack(res_dict["ensemble_probs"], axis=0))  # [K x Batch x N_Classes]
        orig_ensemble_entropy = res_dict["ensemble_entropy"]

        ee = ensemble_entropy(predictions=ensemble_probs)
        ee_sum = -np.sum(ee.numpy())  # Sign is ommitted since it is integrated later on.
        prediction_close = np.isclose(orig_ensemble_entropy, ee_sum)
        self.assertTrue(np.all(prediction_close))

    def compare_values(self, res_dict: dict) -> None:
        ensemble_probs = torch.from_numpy(np.stack(res_dict["ensemble_probs"], axis=0))  # [K x Batch x N_Classes]
        orig_log_det = res_dict["log_det"]
        orig_ensemble_entropy = res_dict["ensemble_entropy"]

        non_one_hot_labels = torch.from_numpy(res_dict["labels"]).to(torch.int64)
        ld = nlog_ensemble_diversity(prediction=ensemble_probs, groundtruth=non_one_hot_labels, verbose=False)
        ee = ensemble_entropy(ensemble_probs)
        ee_sum = -np.sum(ee.numpy(), axis=-1)
        ld_neg = -ld.numpy()
        ld_close = np.isclose(orig_log_det, ld_neg)
        ee_close = np.isclose(orig_ensemble_entropy, ee_sum)

        self.assertTrue(np.all(ld_close))
        self.assertTrue(np.all(ee_close))

    def test_3_models_10_classes_2_samples(self):
        """
        Assures the matrix before logdet calculation is identical
        Test if the normed probabilities of the non true classes are identical
        """
        res_dict: dict = read_file(3, 10, 2)
        self.compare_values(res_dict)

    def test_5_models_10_classes_2_samples(self):
        """
        Assures the matrix before logdet calculation is identical
        Test if the normed probabilities of the non true classes are identical
        """
        res_dict: dict = read_file(5, 10, 2)
        self.compare_values(res_dict)

    def test_7_models_10_classes_2_samples(self):
        """
        Assures the matrix before logdet calculation is identical
        Test if the normed probabilities of the non true classes are identical
        """
        res_dict: dict = read_file(7, 10, 2)
        self.compare_values(res_dict)

    def test_3_models_100_classes_2_samples(self):
        """
        Assures the matrix before logdet calculation is identical
        Test if the normed probabilities of the non true classes are identical
        """
        res_dict: dict = read_file(3, 100, 2)
        self.compare_values(res_dict)

    def test_5_models_100_classes_2_samples(self):
        """
        Assures the matrix before logdet calculation is identical
        Test if the normed probabilities of the non true classes are identical
        """
        res_dict: dict = read_file(3, 100, 2)
        self.compare_values(res_dict)

    def test_7_models_100_classes_2_samples(self):
        """
        Assures the matrix before logdet calculation is identical
        Test if the normed probabilities of the non true classes are identical
        """
        res_dict: dict = read_file(3, 100, 2)
        self.compare_values(res_dict)


if __name__ == "__main__":
    unittest.main()
