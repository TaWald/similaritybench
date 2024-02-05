from simbench.manual_introspection.scis23.scis23_compare_representations_of_models import create_same_seed_comparisons
from simbench.manual_introspection.scripts import grouped_model_results as grm


def main():
    create_same_seed_comparisons(grm.layer_DIFF_tdepth_1_expvar_1, True)


if __name__ == "__main__":
    main()
