import argparse

import numpy as np
from graph.config import DATASET_LIST
from graph.config import DEFAULT_SEEDS
from graph.config import GNN_LIST
from graph.config import LAYER_TEST_N_LAYERS
from graph.config import LAYER_TEST_NAME
from graph.config import MEASURE_DICT
from graph.config import MEASURE_DICT_FUNC_KEY
from graph.config import MEASURE_DICT_PREP_KEY
from graph.config import MEASURE_LIST

from ._base import GraphTest
from .graph_trainer import LayerTestTrainer


class LayerTest(GraphTest):

    def __init__(self, model_name: str, dataset_name: str, n_layers: int = LAYER_TEST_N_LAYERS):
        GraphTest.__init__(self, model_name=model_name, dataset_name=dataset_name, test_name=LAYER_TEST_NAME)

        self.n_layers = n_layers

    def test_measure(self, measure_name, measure=None, save_similarities=True):

        if measure is None:
            measure = MEASURE_DICT[measure_name][MEASURE_DICT_FUNC_KEY]
            prep_funcs = MEASURE_DICT[measure_name][MEASURE_DICT_PREP_KEY]
        else:
            prep_funcs = []

        similarities = dict()
        n_violations = 0
        n_comb_count = 0

        graph_trainer = LayerTestTrainer(self.model_name, self.dataset_name, self.seeds)
        rep_dict = graph_trainer.get_test_representations()

        for seed in self.seeds:

            curr_reps = rep_dict[seed]

            for f in prep_funcs:
                curr_reps = {i: f(R) for i, R in curr_reps.items()}

            similarities[seed] = np.zeros((self.n_layers, self.n_layers))
            for i in range(self.n_layers):

                for j in range(i + 1, self.n_layers):
                    similarities[seed][i, j] = measure(curr_reps[i], curr_reps[j], shape="nd")

            # TODO: wrap this kind of accuracy into separate function, and add rank correlation
            for i in range(self.n_layers):
                for j in range(i + 1, self.n_layers):

                    for k in range(i, j):
                        for l in range(k + 1, j + 1):
                            n_comb_count += 1
                            if similarities[seed][i, j] < similarities[seed][k, l]:
                                n_violations += 1

            print(similarities[seed])

        res_score = 1 - n_violations / n_comb_count

        print(f"The test score for the {measure_name} measure is {res_score}")

        if save_similarities:
            self._save_similarities(similarity_dict=similarities)

        self._save_results(measure_name, res_score)


def parse_args():
    """Parses arguments given to script

    Returns:
        dict-like object -- Dict-like object containing all given arguments
    """
    parser = argparse.ArgumentParser()

    # Test parameters
    parser.add_argument("-a", "--algorithms", nargs="+", type=str, choices=GNN_LIST, help="GNN methods to train")
    parser.add_argument(
        "-d",
        "--datasets",
        nargs="*",
        type=str,
        choices=DATASET_LIST,
        default=DATASET_LIST,
        help="Datasets used in evaluation.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=DEFAULT_SEEDS,
        help="Seeds to be used in experiments. By default, all seeds will be used.",
    )
    parser.add_argument(
        "--train_only", action="store_true", help="whether to only train models but not apply any measures."
    )
    parser.add_argument(
        "-m", "--measures", nargs="*", type=str, choices=MEASURE_LIST, default=MEASURE_DICT, help="Measure to apply."
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    for algorithm in args.algorithms:
        for dataset in args.datasets:

            test = LayerTest(algorithm, dataset)
            for m in args.measures:
                test.test_measure(measure_name=m)
