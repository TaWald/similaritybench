from ke.manual_introspection.scis23.scis23_compare_representations_of_models import create_same_seed_comparisons
from ke.manual_introspection.scripts import grouped_model_results as grm


def main():
    create_same_seed_comparisons(grm.layer_SCIS23_MULTI_tdepth_1_expvar_1, False)


if __name__ == "__main__":
    main()