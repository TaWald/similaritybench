import argparse
from itertools import product

from graphs.config import DATASET_LIST
from graphs.config import GNN_LIST
from graphs.config import LAYER_TEST_NAME
from graphs.config import MEASURE_LIST
from graphs.config import NN_TESTS_LIST
from graphs.tests.graph_test import LayerTest

GNN_TEST_DICT = {LAYER_TEST_NAME: LayerTest}


def parse_args():
    """Parses arguments given to script

    Returns:
        dict-like object -- Dict-like object containing all given arguments
    """
    parser = argparse.ArgumentParser()

    # Test parameters
    parser.add_argument("-a", "--architectures", nargs="+", type=str, choices=GNN_LIST, help="GNN methods to train")
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
        "-m",
        "--measures",
        nargs="*",
        type=str,
        choices=MEASURE_LIST,
        default=MEASURE_LIST,
        help="Measures to test.",
    )
    parser.add_argument(
        "-t",
        "--test",
        nargs=1,
        type=str,
        choices=NN_TESTS_LIST,
        default=LAYER_TEST_NAME,
        help="Tests to run.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    for model_name, dataset_name in product(args.architectures, args.datasets):

        test = GNN_TEST_DICT[args.test](model_name=model_name, dataset_name=dataset_name)
        test.test_measures(args.measures)