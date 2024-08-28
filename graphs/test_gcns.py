import argparse
import copy
import itertools
import json
from pathlib import Path
from time import localtime
from time import strftime

import graphs.pgnn.train as pgnn
import numpy as np
import pandas as pd
import torch
from graphs import gnn
from graphs.config import DATASET_LIST
from graphs.config import GNN_DICT
from graphs.config import GNN_LIST
from graphs.config import GNN_PARAMS_DICT
from graphs.config import GNN_PARAMS_N_LAYERS_KEY
from graphs.config import OPTIMIZER_PARAMS_DECAY_KEY
from graphs.config import OPTIMIZER_PARAMS_DICT
from graphs.config import OPTIMIZER_PARAMS_LR_KEY
from graphs.config import PGNN_PARAMS_ANCHOR_DIM_KEY
from graphs.config import PGNN_PARAMS_ANCHOR_NUM_KEY
from graphs.config import SPLIT_IDX_TEST_KEY
from graphs.config import TORCH_STATE_DICT_FILE_NAME_SEED
from graphs.config import TRAIN_LOG_FILE_NAME_SEED
from graphs.gnn import train_model
from graphs.graph_trainer import GraphTrainer
from graphs.tools import precompute_dist_data
from graphs.tools import preselect_anchor
from repsim.benchmark.paths import BASE_PATH
from repsim.benchmark.types_globals import EXPERIMENT_SEED
from repsim.benchmark.types_globals import GRAPH_ARCHITECTURE_TYPE
from repsim.benchmark.types_globals import GRAPH_DATASET_TRAINED_ON
from repsim.benchmark.types_globals import PGNN_MODEL_NAME
from torch_geometric.utils import to_edge_index

SEEDS = [1, 2, 3, 4, 5]

RES_DIR = Path(BASE_PATH, "graph_tests")


class GNNTester:

    def __init__(
        self,
        architecture_type: GRAPH_ARCHITECTURE_TYPE,
        dataset_name: GRAPH_DATASET_TRAINED_ON,
        seed: EXPERIMENT_SEED,
        model_name: str,
        n_layers: int = None,
        lr: float = None,
        decay: float = None,
        device: int | str = 0,
    ):

        self.architecture_type = architecture_type
        self.seed = seed
        self.dataset_name: GRAPH_DATASET_TRAINED_ON = dataset_name
        self.data, self.n_classes, self.split_idx = GraphTrainer.get_data(self.dataset_name)
        self.model_name = model_name
        self.edge_index = to_edge_index(self.data.adj_t)[0]

        if isinstance(device, str):
            self.device = torch.device(device)
        else:
            dev_str = f"cuda:{device}" if torch.cuda.is_available() else "cpu"
            self.device = torch.device(dev_str)

        self.gnn_params = copy.deepcopy(GNN_PARAMS_DICT[self.architecture_type][self.dataset_name])
        self.gnn_params["in_channels"] = self.data.num_features
        self.gnn_params["out_channels"] = self.n_classes

        if self.architecture_type == PGNN_MODEL_NAME:
            dists = precompute_dist_data(self.edge_index.numpy(), self.data.num_nodes, approximate=0)
            self.data.dists = torch.from_numpy(dists).float()

            anchor_dim = preselect_anchor(
                self.data,
                layer_num=self.gnn_params[GNN_PARAMS_N_LAYERS_KEY],
                anchor_num=self.gnn_params[PGNN_PARAMS_ANCHOR_NUM_KEY],
            )

            self.gnn_params[PGNN_PARAMS_ANCHOR_DIM_KEY] = anchor_dim

        if n_layers is not None:
            self.gnn_params[GNN_PARAMS_N_LAYERS_KEY] = n_layers

        self.optimizer_params = OPTIMIZER_PARAMS_DICT[self.architecture_type][self.dataset_name]

        if decay is not None:
            self.optimizer_params[OPTIMIZER_PARAMS_DECAY_KEY] = decay

        if lr is not None:
            self.optimizer_params[OPTIMIZER_PARAMS_LR_KEY] = lr

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

        save_path = self.model_path / TORCH_STATE_DICT_FILE_NAME_SEED(self.seed)
        model = GNN_DICT[self.architecture_type](**self.gnn_params)

        if self.architecture_type == PGNN_MODEL_NAME:

            train_results, test_acc = pgnn.train_model(
                model=model,
                data=self.data,
                split_idx=self.split_idx,
                device=self.device,
                seed=self.seed,
                optimizer_params=self.optimizer_params,
                save_path=save_path,
                b_test=True,
            )
        else:
            train_results, test_acc = train_model(
                model=model,
                data=self.data,
                edge_index=self.edge_index,
                split_idx=self.split_idx,
                device=self.device,
                seed=self.seed,
                optimizer_params=self.optimizer_params,
                p_drop_edge=0.0,
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

        if self.architecture_type == PGNN_MODEL_NAME:
            return pgnn.get_representations(
                model=model,
                data=self.data,
                device=self.device,
                test_idx=self.split_idx[SPLIT_IDX_TEST_KEY],
                layer_ids=list(range(self.gnn_params["num_layers"] - 1)),
            )

        return gnn.get_representations(
            model=model,
            data=self.data,
            device=self.device,
            test_idx=self.split_idx[SPLIT_IDX_TEST_KEY],
            layer_ids=list(range(self.gnn_params["num_layers"] - 1)),
        )


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
    parser.add_argument("--n_layers", type=int, default=3, help="Number of Layers to benchmark")

    parser.add_argument("--device", type=int, default=0, help="Number of Layers to benchmark")

    parser.add_argument("--decays", type=float, nargs="*", default=None)

    parser.add_argument("--learning_rates", type=float, nargs="*", default=None)

    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()
    for architecture, dataset in itertools.product(args.architectures, args.datasets):

        decays = [None] if args.decays is None else args.decays
        lrs = [None] if args.learning_rates is None else args.learning_rates

        for decay, lr in itertools.product(decays, lrs):
            test_accs = []
            model_str = f"model_dec-{decay}_{strftime('%Y-%m-%d_%H-%M-%S', localtime())}"
            for s in SEEDS:
                print("seed is", s)
                trainer = GNNTester(
                    architecture_type=architecture,
                    dataset_name=dataset,
                    seed=s,
                    model_name=model_str,
                    n_layers=args.n_layers,
                    decay=decay,
                    lr=lr,
                )
                test_accuracy = trainer.train_test_model()
                test_accs.append(test_accuracy)
                plain_reps = trainer.get_test_representations()

                all_single_layer_reps = []
                for layer_id, rep in plain_reps.items():
                    all_single_layer_reps.append(rep)

            print(f"Mean Test Accuracy for {architecture} on {dataset} with decay {decay} is:")
            print(np.mean(test_accs))
