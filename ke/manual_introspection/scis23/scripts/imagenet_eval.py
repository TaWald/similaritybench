from ke.manual_introspection.scis23.scis23_compare_representations_of_models import (
    create_same_seed_functional_ensemble_comparison,
)
from ke.manual_introspection.scripts import grouped_model_results as grm


def main():
    # create_same_seed_comparisons(grm.baseline_5_ensembles_imagenet1k)
    # create_same_seed_comparisons(grm.five_ensembles_imagenet1k)
    create_same_seed_functional_ensemble_comparison(grm.baseline_5_ensembles_imagenet1k, False)
    # create_same_seed_functional_ensemble_comparison(grm.five_ensembles_imagenet1k, False)


if __name__ == "__main__":
    main()
