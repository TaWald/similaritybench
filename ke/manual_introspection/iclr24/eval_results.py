from ke.manual_introspection.comparison_json_creator import compare_functional_same_seed_ensemble
from ke.manual_introspection.comparison_json_creator import compare_representations_same_seed_first_to_second
from ke.manual_introspection.iclr24.paths import ckpt_results
from ke.manual_introspection.iclr24.paths import json_results_path


def baseline_similarity_comparison():
    cifar10_resnet_34_unregularized = {
        "cifar10_ResNet34_first_models": {
            "dataset": "CIFAR10",
            "architecture": "ResNet34",
            "dis_loss": "None",
            "dis_loss_weight": 0.00,
            "ce_loss_weight": 1.00,
        },
    }

    compare_representations_same_seed_first_to_second(
        cifar10_resnet_34_unregularized,
        json_results_path=json_results_path / "iclr24",
        ckpt_result_path=ckpt_results,
        overwrite=False,
    )


def lambda_ablation_cifar10():
    """Compares how changing lambda effects the metrics (when only looking at a group of 2 models)"""
    lambda_ablations = {
        f"two_models__cifar10__ResNet34__{loss}_{j:.02f}__tp_{i}": {
            "dataset": "CIFAR10",
            "architecture": "ResNet34",
            "hooks": [i],
            "dis_loss": "ExpVar",
            "dis_loss_weight": j,
            "ce_loss_weight": 1.00,
        }
        for loss in ["ExpVar", "LinCKA", "L2Corr"]
        for j in [0.25, 1.00, 4.00]
        for i in [1, 3, 8, 13]
    }
    compare_functional_same_seed_ensemble(
        lambda_ablations, 2, json_results_path=json_results_path, ckpt_result_path=ckpt_results, overwrite=False
    )
    lambda_baseline = {
        f"two_models__cifar10__ResNet34__None_0.00__tp_1": {
            "dataset": "CIFAR10",
            "architecture": "ResNet34",
            "hooks": [1],
            "dis_loss": "None",
            "dis_loss_weight": 0.00,
            "ce_loss_weight": 1.00,
        }
    }
    compare_functional_same_seed_ensemble(
        lambda_baseline, 2, json_results_path=json_results_path, ckpt_result_path=ckpt_results, overwrite=False
    )


def cifar10_resnet_34_ensemble_early_middle_late_very_late_vs_baseline():
    """
    Evaluates the unregularized ensemble of 5 models of ResNet34 (trained on CIFAR10),
    to an ensemble of early, middle, late and very late regularization. (weight 1.0)
    """
    cifar10_resnet_34_ensemble = {
        f"ensemble__cifar10__ResNet34__ExpVar_{dlw:.02f}__tp_{i}": {
            "dataset": "CIFAR10",
            "architecture": "ResNet34",
            "hooks": [i],
            "dis_loss": "ExpVar",
            "dis_loss_weight": dlw,
            "ce_loss_weight": 1.00,
        }
        for i in [1, 3, 8, 13]
        for dlw in [0.25, 1.0, 4.0]
    }
    compare_functional_same_seed_ensemble(
        cifar10_resnet_34_ensemble,
        5,
        json_results_path=json_results_path,
        ckpt_result_path=ckpt_results,
        overwrite=True,
    )
    cifar10_resnet_34_ensemble_baseline = {
        f"ensemble__cifar10__ResNet34__baseline": {
            "dataset": "CIFAR10",
            "architecture": "ResNet34",
            "dis_loss": "None",
            "dis_loss_weight": 0.00,
            "ce_loss_weight": 1.00,
        }
    }
    compare_functional_same_seed_ensemble(
        cifar10_resnet_34_ensemble_baseline,
        5,
        json_results_path=json_results_path,
        ckpt_result_path=ckpt_results,
        overwrite=True,
    )


def main():
    cifar10_resnet_34_ensemble_early_middle_late_very_late_vs_baseline()


if __name__ == "__main__":
    main()
