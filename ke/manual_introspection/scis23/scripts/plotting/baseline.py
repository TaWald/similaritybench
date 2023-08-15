from argparse import ArgumentParser

from ke.manual_introspection.scis23.scripts.plot_cka_sim import plot_cka_sim
from ke.manual_introspection.scripts import grouped_model_results as grm

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--json_name", type=str, required=True, help="Name of the json file located in representation_comp_results."
    )

    for name in [f + ".json" for f in list(grm.baseline_cifar10.keys())]:
        plot_cka_sim(name)
