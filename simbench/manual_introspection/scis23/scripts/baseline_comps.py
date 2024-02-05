from simbench.manual_introspection.scis23.scis23_compare_representations_of_models import create_baseline_comparisons
from simbench.manual_introspection.scripts import grouped_model_results as grm


def main():
    # create_baseline_comparisons(grm.baseline_cifar10)
    # create_baseline_comparisons(grm.baseline_cifar100)  # Needs a comparison model to grab the baselines
    # create_baseline_comparisons(grm.baseline_imagenet100)  # Needs a comparison model to create baselines
    create_baseline_comparisons(grm.baseline_imagenet1k)


if __name__ == "__main__":
    main()
