from simbench.manual_introspection.scis23.scis23_compare_representations_of_models import (
    create_same_seed_functional_ensemble_comparison,
)
from simbench.manual_introspection.scripts import grouped_model_results as grm


def main():
    print("Functional eval Baseline")
    create_same_seed_functional_ensemble_comparison(grm.baseline_cifar100_resnet101_layer_diff, True)
    print("Functional eval LinCKA")
    create_same_seed_functional_ensemble_comparison(grm.lincka_ensemble_cifar100_resnet101_layer_diff, True)


if __name__ == "__main__":
    main()
