from argparse import ArgumentParser

from ke.manual_introspection.scripts.plot_multilayer import plot_cka_sim

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--json_name', type=str, required=True, help="Name of the json file located in representation_comp_results.")

    for name in [
        'non_reg_2_non_reg__layer_9_tdepth_1_expvar_1.json',
        'in_seed_non_reg_to_reg__layer_9_tdepth_1_expvar_1.json',
        'cross_seed_reg_2_reg__layer_9_tdepth_1_expvar_1.json',
        'cross_seed_reg_2_non_reg__layer_9_tdepth_1_expvar_1.json']:
        plot_cka_sim(name)
