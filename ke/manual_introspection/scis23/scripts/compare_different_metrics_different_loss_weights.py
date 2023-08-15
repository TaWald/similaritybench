from ke.manual_introspection.scis23.scis23_compare_representations_of_models import (
    create_same_seed_functional_ensemble_comparison_first_two,
)
from ke.manual_introspection.scripts import grouped_model_results as grm


def main():
    create_same_seed_functional_ensemble_comparison_first_two(grm.different_metrics_scis23, True)
    print("Functional eval Baseline")
    create_same_seed_functional_ensemble_comparison_first_two(grm.scis_baseline_ensembles_first_two, True)


if __name__ == "__main__":
    main()
