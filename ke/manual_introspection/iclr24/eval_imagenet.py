from ke.manual_introspection.comparison_json_creator import (
    compare_functional_same_seed_ensemble,
)
from ke.manual_introspection.comparison_json_creator import compare_representations_same_seed_first_to_second
from ke.manual_introspection.iclr24.paths import ckpt_results
from ke.manual_introspection.iclr24.paths import json_results_path


def baseline_similarity_comparison():
    imagenet_resnet_34_unregularized = {
        "imagenet_ResNet34_first_models": {
            "dataset": "ImageNet",
            "architecture": "ResNet34",
            "dis_loss": "None",
            "dis_loss_weight": 0.00,
            "ce_loss_weight": 1.00,
        },
    }

    compare_representations_same_seed_first_to_second(
        imagenet_resnet_34_unregularized,
        json_results_path=json_results_path / "iclr24",
        ckpt_result_path=ckpt_results,
        overwrite=False,
    )


def imagenet_resnet_34_ensemble_early_middle_late_very_late_vs_baseline():
    """
    Evaluates the unregularized ensemble of 5 models of ResNet34 (trained on CIFAR10),
    to an ensemble of early, middle, late and very late regularization. (weight 1.0)
    """
    imagenet_resnet_34_ensemble = {
        f"ensemble__imagenet__ResNet34__ExpVar_{dlw:.02f}__tp_{i}": {
            "dataset": "ImageNet",
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
        imagenet_resnet_34_ensemble,
        5,
        json_results_path=json_results_path,
        ckpt_result_path=ckpt_results,
        overwrite=False,
    )
    imagenet_resnet_34_ensemble_baseline = {
        f"ensemble__cifar10__ResNet34__baseline": {
            "dataset": "CIFAR10",
            "architecture": "ResNet34",
            "dis_loss": "None",
            "dis_loss_weight": 0.00,
            "ce_loss_weight": 1.00,
            "hooks": [1],
        }
    }
    compare_functional_same_seed_ensemble(
        imagenet_resnet_34_ensemble_baseline,
        5,
        json_results_path=json_results_path,
        ckpt_result_path=ckpt_results,
        overwrite=True,
    )


def main():
    imagenet_resnet_34_ensemble_early_middle_late_very_late_vs_baseline()


if __name__ == "__main__":
    main()
