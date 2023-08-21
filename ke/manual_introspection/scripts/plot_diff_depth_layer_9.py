from argparse import ArgumentParser

from ke.manual_introspection.scripts.plot_multilayer import plot_cka_sim

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--json_name",
        type=str,
        required=True,
        help="Name of the json file located " "in representation_comp_results.",
    )

    for name in [
        "layer_9_tdepth_1_expvar_1.json",
        "layer_9_tdepth_3_expvar_1.json",
        "layer_9_tdepth_5_expvar_1.json",
        "layer_9_tdepth_7_expvar_1.json",
        "layer_9_tdepth_9_expvar_1.json",
    ]:
        plot_cka_sim(name)

    for name in [
        "layer_9_tdepth_1_expvar_1.json",
        "layer_9_tdepth_3_expvar_1.json",
        "layer_9_tdepth_5_expvar_1.json",
        "layer_9_tdepth_7_expvar_1.json",
        "layer_9_tdepth_9_expvar_1.json",
    ]:
        plot_cka_sim(name)
