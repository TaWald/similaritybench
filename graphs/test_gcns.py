import argparse
import copy
import itertools
import json
from pathlib import Path
from time import localtime
from time import strftime

import numpy as np
import pandas as pd
import torch
from graphs.config import DATASET_LIST
from graphs.config import GNN_DICT
from graphs.config import GNN_LIST
from graphs.config import GNN_PARAMS_DICT
from graphs.config import OPTIMIZER_PARAMS_DICT
from graphs.config import SPLIT_IDX_TEST_KEY
from graphs.config import TORCH_STATE_DICT_FILE_NAME_SEED
from graphs.config import TRAIN_LOG_FILE_NAME_SEED
from graphs.gnn import get_representations
from graphs.gnn import train_model
from graphs.graph_trainer import GraphTrainer
from repsim.benchmark.paths import BASE_PATH
from repsim.benchmark.types_globals import GRAPH_ARCHITECTURE_TYPE
from repsim.benchmark.types_globals import GRAPH_DATASET_TRAINED_ON
from repsim.benchmark.types_globals import GRAPH_EXPERIMENT_SEED

SEEDS = [1, 2, 3, 4, 5]

RES_DIR = Path(BASE_PATH, "graph_tests")


class GNNTester:

    def __init__(
        self,
        architecture_type: GRAPH_ARCHITECTURE_TYPE,
        dataset_name: GRAPH_DATASET_TRAINED_ON,
        seed: GRAPH_EXPERIMENT_SEED,
        model_name: str,
        device: int | str = 0,
    ):

        self.architecture_type = architecture_type
        self.seed = seed
        self.dataset_name: GRAPH_DATASET_TRAINED_ON = dataset_name
        self.data, self.n_classes, self.split_idx = GraphTrainer.get_data(self.dataset_name)
        self.model_name = model_name

        if isinstance(device, str):
            self.device = torch.device(device)
        else:
            dev_str = f"cuda:{device}" if torch.cuda.is_available() else "cpu"
            self.device = torch.device(dev_str)

        self.gnn_params = copy.deepcopy(GNN_PARAMS_DICT[self.architecture_type][self.dataset_name])
        self.gnn_params["in_channels"] = self.data.num_features
        self.gnn_params["out_channels"] = self.n_classes
        self.optimizer_params = OPTIMIZER_PARAMS_DICT[self.architecture_type][self.dataset_name]

        model_dataset_path = RES_DIR / self.dataset_name
        Path(model_dataset_path).mkdir(parents=True, exist_ok=True)

        arch_path = model_dataset_path / self.architecture_type
        Path(arch_path).mkdir(parents=True, exist_ok=True)

        self.model_path = arch_path / self.model_name
        Path(self.model_path).mkdir(parents=True, exist_ok=True)

        with open(self.model_path / "gnn_params.json", "w") as fp:
            json.dump(self.gnn_params, fp)

        with open(self.model_path / "optimizer_params.json", "w") as fp:
            json.dump(self.optimizer_params, fp)

    def _load_model(self):

        model = GNN_DICT[self.architecture_type](**self.gnn_params)
        model_file = self.model_path / TORCH_STATE_DICT_FILE_NAME_SEED(self.seed)

        print(model_file)
        if not model_file.is_file():
            raise FileNotFoundError(f"Model File for seed {self.seed} does not exist")

        model.load_state_dict(torch.load(model_file, map_location=self.device))

        return model

    def train_test_model(self):

        print(f"Train {self.architecture_type} on {self.dataset_name}")

        model = GNN_DICT[self.architecture_type](**self.gnn_params)
        save_path = self.model_path / TORCH_STATE_DICT_FILE_NAME_SEED(self.seed)

        train_results, test_acc = train_model(
            model=model,
            data=self.data,
            split_idx=self.split_idx,
            device=self.device,
            seed=self.seed,
            optimizer_params=self.optimizer_params,
            save_path=save_path,
            b_test=True,
        )
        print(train_results[-1])

        df_log = pd.DataFrame(
            train_results, columns=["Epoch", "Loss", "Training_Accuracy", "Validation_Accuracy", "Test_Accuracy"]
        )
        df_log.to_csv(
            self.model_path / TRAIN_LOG_FILE_NAME_SEED(self.seed),
            index=False,
        )

        return test_acc

    def get_test_representations(self):

        model = self._load_model()

        reps = get_representations(
            model=model,
            data=self.data,
            device=self.device,
            test_idx=self.split_idx[SPLIT_IDX_TEST_KEY],
            layer_ids=list(range(self.gnn_params["num_layers"] - 1)),
        )

        return reps


def parse_args():
    """Parses arguments given to script

    Returns:
        dict-like object -- Dict-like object containing all given arguments
    """
    parser = argparse.ArgumentParser()

    # Test parameters
    parser.add_argument(
        "-a", "--architectures", type=str, nargs="+", default=GNN_LIST, choices=GNN_LIST, help="GNN type to benchmark"
    )
    parser.add_argument(
        "-d",
        "--datasets",
        type=str,
        nargs="+",
        default=DATASET_LIST,
        choices=DATASET_LIST,
        help="Dataset to benchmark",
    )
    parser.add_argument("-n", "--n_layers", type=int, default=3, help="Number of Layers to benchmark")

    parser.add_argument("--device", type=int, default=0, help="Number of Layers to benchmark")

    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()
    for architecture, dataset in itertools.product(args.architectures, args.datasets):

        test_accs = []
        model_str = f"model_{strftime('%Y-%m-%d_%H-%M-%S', localtime())}"
        for s in SEEDS:
            print("seed is", s)
            trainer = GNNTester(architecture_type=architecture, dataset_name=dataset, seed=s, model_name=model_str)
            test_accuracy = trainer.train_test_model()
            test_accs.append(test_accuracy)
            plain_reps = trainer.get_test_representations()

            all_single_layer_reps = []
            for layer_id, rep in plain_reps.items():
                all_single_layer_reps.append(rep)

        print(f"Mean Test Accuracy for {architecture} on {dataset} is:")
        print(np.mean(test_accs))
