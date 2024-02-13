import argparse
from itertools import product

from graph.config import DATASET_LIST
from graph.config import GNN_LIST
from graph.config import LAYER_TEST_NAME
from graph.config import MEASURE_DICT
from graph.config import MEASURE_LIST
from graph.config import NN_TESTS_LIST
from graph.tests.graph_test import LayerTest

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
        help="Datasets used in evaluation.",
    )
    parser.add_argument(
        "-m",
        "--measures",
        nargs="*",
        type=str,
        choices=MEASURE_LIST,
        default=MEASURE_LIST,
        help="Datasets used in evaluation.",
    )
    parser.add_argument(
        "-t",
        "--test",
        nargs=1,
        type=str,
        choices=NN_TESTS_LIST,
        default=LAYER_TEST_NAME,
        help="Datasets used in evaluation.",
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

    for model_name, dataset_name in product(args.architectures, args.datasets):

        test = GNN_TEST_DICT[args.test](model_name=model_name, dataset_name=dataset_name)
        for m in args.measures:
            test.test_measure(measure_name=m)
