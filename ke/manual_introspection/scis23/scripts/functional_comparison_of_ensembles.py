from ke.manual_introspection.scis23.scis23_compare_representations_of_models import (
    create_same_seed_functional_ensemble_comparison,
)
from ke.manual_introspection.scripts import grouped_model_results as grm


def main():
    create_same_seed_functional_ensemble_comparison(grm.l2corr_ensemble_layer_DIFF, True)
    print("Functional eval Baseline")
    create_same_seed_functional_ensemble_comparison(grm.scis_baseline_ensembles, True)
    print("Functional eval LinCKA")
    create_same_seed_functional_ensemble_comparison(grm.lincka_ensemble_layer_DIFF, True)
    print("Functional eval ExpVar")
    create_same_seed_functional_ensemble_comparison(grm.expvar_ensemble_layer_DIFF, True)


if __name__ == "__main__":
    main()
