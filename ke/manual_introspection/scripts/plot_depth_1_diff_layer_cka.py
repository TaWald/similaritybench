from argparse import ArgumentParser

from ke.manual_introspection.scripts.plot_cka_sim import plot_cka_sim

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--json_name', type=str, required=True, help="Name of the json file located in representation_comp_results.")

    for name in [
        'layer_1_tdepth_1_expvar_1.json',
        'layer_3_tdepth_1_expvar_1.json',
        'layer_5_tdepth_1_expvar_1.json',
        'layer_7_tdepth_1_expvar_1.json',
        'layer_9_tdepth_1_expvar_1.json',
        'layer_11_tdepth_1_expvar_1.json',
        'layer_13_tdepth_1_expvar_1.json',
        'layer_15_tdepth_1_expvar_1.json',]:
        plot_cka_sim(name)

