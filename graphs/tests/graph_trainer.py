import argparse
import os
from abc import ABC
from abc import abstractmethod
from itertools import product
from pathlib import Path
from typing import get_args
from typing import List

import pandas as pd
import torch
from graphs.config import DATASET_LIST
from graphs.config import GAT_PARAMS_DEFAULT_ATT_DROPOUT
from graphs.config import GAT_PARAMS_DEFAULT_N_HEADS
from graphs.config import GNN_DICT
from graphs.config import GNN_LIST
from graphs.config import GNN_PARAMS_DEFAULT_DIMENSION
from graphs.config import GNN_PARAMS_DEFAULT_DROPOUT
from graphs.config import GNN_PARAMS_DEFAULT_LR
from graphs.config import GNN_PARAMS_DEFAULT_N_EPOCHS
from graphs.config import GNN_PARAMS_DEFAULT_N_LAYERS
from graphs.config import GNN_PARAMS_DEFAULT_NORM
from graphs.config import LAYER_EXPERIMENT_N_LAYERS
from graphs.config import TORCH_STATE_DICT_FILE_NAME_SEED
from graphs.config import TRAIN_LOG_FILE_NAME_SEED
from graphs.gnn import get_representations
from graphs.gnn import train_model
from graphs.tests.tools import shuffle_labels
from ogb.nodeproppred import PygNodePropPredDataset
from repsim.benchmark.paths import GRAPHS_DATA_PATH
from repsim.benchmark.paths import GRAPHS_MODEL_PATH
from repsim.benchmark.types_globals import BENCHMARK_EXPERIMENTS_LIST
from repsim.benchmark.types_globals import EXPERIMENT_DICT
from repsim.benchmark.types_globals import EXPERIMENT_IDENTIFIER
from repsim.benchmark.types_globals import GRAPH_ARCHITECTURE_TYPE
from repsim.benchmark.types_globals import GRAPH_DATASET_TRAINED_ON
from repsim.benchmark.types_globals import GRAPH_EXPERIMENT_SEED
from repsim.benchmark.types_globals import LABEL_EXPERIMENT_NAME
from repsim.benchmark.types_globals import LAYER_EXPERIMENT_NAME
from repsim.benchmark.types_globals import SETTING_IDENTIFIER
from repsim.benchmark.types_globals import SHORTCUT_EXPERIMENT_NAME
from repsim.benchmark.types_globals import SHORTCUT_EXPERIMENT_SEED
from repsim.benchmark.types_globals import STANDARD_SETTING
from torch_geometric import transforms as t


class GraphTrainer(ABC):

    def __init__(
        self,
        architecture_type: GRAPH_ARCHITECTURE_TYPE,
        dataset_name: GRAPH_DATASET_TRAINED_ON,
        test_name: EXPERIMENT_IDENTIFIER,
        seed: GRAPH_EXPERIMENT_SEED,
        device: int | str = 0,
    ):

        self.test_name = test_name
        self.settings = EXPERIMENT_DICT[test_name]
        self.architecture_type = architecture_type
        self.seed = seed
        self.dataset_name: str = dataset_name
        self.data, self.n_classes, self.split_idx = self._get_data()
        self.models = dict()

        if isinstance(device, str):
            self.device = torch.device(device)
        else:
            dev_str = f"cuda:{device}" if torch.cuda.is_available() else "cpu"
            self.device = torch.device(dev_str)

        self.gnn_params, self.optimizer_params = self._get_gnn_params()

        model_dataset_path = GRAPHS_MODEL_PATH / self.dataset_name
        Path(model_dataset_path).mkdir(parents=True, exist_ok=True)

        self.models_path = model_dataset_path / self.architecture_type
        Path(self.models_path).mkdir(parents=True, exist_ok=True)

        self.setting_paths = dict()
        for setting in self.settings:
            setting_path = self.models_path / setting
            Path(setting_path).mkdir(parents=True, exist_ok=True)
            self.setting_paths[setting] = setting_path

    # TODO: set up way to read in params which may be determined by graphgym
    @abstractmethod
    def _get_gnn_params(self):
        pass

    def _check_pretrained(self, settings):
        missing_settings = []
        for setting in settings:
            if not Path(self.setting_paths[setting], TORCH_STATE_DICT_FILE_NAME_SEED(self.seed)).exists():
                missing_settings.append(setting)

        return missing_settings

    def _load_model(self, setting):
        model = GNN_DICT[self.architecture_type](**self.gnn_params)
        model_file = self.setting_paths[setting] / TORCH_STATE_DICT_FILE_NAME_SEED(self.seed)

        print(model_file)
        if not model_file.is_file():
            raise FileNotFoundError(f"Model File for seed {self.seed} does not exist")

        model.load_state_dict(torch.load(model_file, map_location=self.device))

        return model

    def _get_data(self):
        # TODO: This is assuming an OGB dataset, consider multiple cases if non-obg data is used

        pyg_dataset = PygNodePropPredDataset(
            name=str(self.dataset_name),
            transform=t.Compose([t.ToUndirected(), t.ToSparseTensor()]),
            root=GRAPHS_DATA_PATH,
        )

        return pyg_dataset[0], pyg_dataset.num_classes, pyg_dataset.get_idx_split()

    def _log_train_results(self, train_results, setting):
        df_train = pd.DataFrame(train_results, columns=["Epoch", "Loss", "Training_Accuracy", "Validation_Accuracy"])
        df_train.to_csv(
            self.setting_paths[setting] / TRAIN_LOG_FILE_NAME_SEED(self.seed),
            index=False,
        )

    def train_models(self, settings: List[SETTING_IDENTIFIER] = None, retrain: bool = False):

        if settings is None:
            settings = self.settings
        else:
            for setting in settings:
                assert setting in self.settings, f"Setting {setting} is invalid, valid settings are {self.settings}"

        if not retrain:
            settings = self._check_pretrained(settings)

        for setting in settings:
            self._train_model(setting)

    @abstractmethod
    def _get_setting_data(self, setting: SETTING_IDENTIFIER):
        pass

    def _train_model(self, setting, log_results: bool = True):

        print(f"Train {self.architecture_type} on {self.dataset_name} in {setting} setting.")

        setting_data = self._get_setting_data(setting)

        model = GNN_DICT[self.architecture_type](**self.gnn_params)
        save_path = self.setting_paths[setting] / TORCH_STATE_DICT_FILE_NAME_SEED(self.seed)
        train_results = train_model(
            model, setting_data, self.split_idx, self.device, self.seed, self.optimizer_params, save_path
        )

        if log_results:
            self._log_train_results(train_results, setting)

    def get_test_representations(self, setting: SETTING_IDENTIFIER):

        model = self._load_model(setting)
        setting_data = self._get_setting_data(setting)

        reps = get_representations(
            model=model,
            data=setting_data,
            test_idx=self.split_idx["test"],
            layer_ids=list(range(self.gnn_params["num_layers"] - 1)),
        )

        return reps


class LayerTestTrainer(GraphTrainer):

    def __init__(
        self,
        architecture_type: GRAPH_ARCHITECTURE_TYPE,
        dataset_name: GRAPH_DATASET_TRAINED_ON,
        seed: GRAPH_EXPERIMENT_SEED,
        n_layers: int = None,
    ):
        self.n_layers = LAYER_EXPERIMENT_N_LAYERS if n_layers is None else n_layers

        GraphTrainer.__init__(
            self,
            architecture_type=architecture_type,
            dataset_name=dataset_name,
            seed=seed,
            test_name=LAYER_EXPERIMENT_NAME,
        )

    def _get_gnn_params(self):

        gnn_params = {
            "num_layers": self.n_layers,
            "in_channels": self.data.num_features,
            "hidden_channels": GNN_PARAMS_DEFAULT_DIMENSION,
            "dropout": GNN_PARAMS_DEFAULT_DROPOUT,
            "out_channels": self.n_classes,
            "norm": GNN_PARAMS_DEFAULT_NORM,
        }
        optimizer_params = {"epochs": GNN_PARAMS_DEFAULT_N_EPOCHS, "lr": GNN_PARAMS_DEFAULT_LR}

        if self.architecture_type == "GAT":
            gnn_params["heads"] = GAT_PARAMS_DEFAULT_N_HEADS
            gnn_params["dropout"] = GAT_PARAMS_DEFAULT_ATT_DROPOUT
            gnn_params["hidden_channels"] *= gnn_params["heads"]

        return gnn_params, optimizer_params

    def _get_setting_data(self, setting: SETTING_IDENTIFIER):
        return self.data.clone()


class LabelTestTrainer(GraphTrainer):

    def __init__(
        self,
        architecture_type: GRAPH_ARCHITECTURE_TYPE,
        dataset_name: GRAPH_DATASET_TRAINED_ON,
        seed: GRAPH_EXPERIMENT_SEED,
        n_layers: int = None,
    ):
        self.n_layers = LAYER_EXPERIMENT_N_LAYERS if n_layers is None else n_layers
        GraphTrainer.__init__(
            self,
            architecture_type=architecture_type,
            dataset_name=dataset_name,
            seed=seed,
            test_name=LABEL_EXPERIMENT_NAME,
        )

    def _get_gnn_params(self):

        gnn_params = {
            "num_layers": GNN_PARAMS_DEFAULT_N_LAYERS,
            "in_channels": self.data.num_features,
            "hidden_channels": GNN_PARAMS_DEFAULT_DIMENSION,
            "dropout": GNN_PARAMS_DEFAULT_DROPOUT,
            "out_channels": self.n_classes,
            "norm": GNN_PARAMS_DEFAULT_NORM,
        }

        if self.architecture_type == "GAT":
            gnn_params["heads"] = GAT_PARAMS_DEFAULT_N_HEADS
            gnn_params["dropout"] = GAT_PARAMS_DEFAULT_ATT_DROPOUT
            gnn_params["hidden_channels"] *= gnn_params["heads"]

        optimizer_params = {"epochs": GNN_PARAMS_DEFAULT_N_EPOCHS, "lr": GNN_PARAMS_DEFAULT_LR}

        return gnn_params, optimizer_params

    def _get_setting_data(self, setting: SETTING_IDENTIFIER):

        setting_data = self.data.clone()

        if setting != STANDARD_SETTING:
            old_labels = self.data.y.detach().clone()
            shuffle_frac = int(setting.split("_")[-1]) / 100.0
            setting_data.y = shuffle_labels(old_labels, frac=shuffle_frac, seed=self.seed)

        return setting_data


class ShortCutTestTrainer(GraphTrainer):
    def __init__(
        self,
        architecture_type: GRAPH_ARCHITECTURE_TYPE,
        dataset_name: GRAPH_DATASET_TRAINED_ON,
        seed: GRAPH_EXPERIMENT_SEED,
        n_layers: int = None,
    ):
        self.n_layers = LAYER_EXPERIMENT_N_LAYERS if n_layers is None else n_layers
        GraphTrainer.__init__(
            self,
            architecture_type=architecture_type,
            dataset_name=dataset_name,
            seed=seed,
            test_name=SHORTCUT_EXPERIMENT_NAME,
        )

    def _get_gnn_params(self):

        gnn_params = {
            "num_layers": GNN_PARAMS_DEFAULT_N_LAYERS,
            "in_channels": self.data.num_features + 1,
            "hidden_channels": GNN_PARAMS_DEFAULT_DIMENSION,
            "dropout": GNN_PARAMS_DEFAULT_DROPOUT,
            "out_channels": self.n_classes,
            "norm": GNN_PARAMS_DEFAULT_NORM,
        }

        if self.architecture_type == "GAT":
            gnn_params["heads"] = GAT_PARAMS_DEFAULT_N_HEADS
            gnn_params["dropout"] = GAT_PARAMS_DEFAULT_ATT_DROPOUT
            gnn_params["hidden_channels"] *= gnn_params["heads"]

        optimizer_params = {"epochs": GNN_PARAMS_DEFAULT_N_EPOCHS, "lr": GNN_PARAMS_DEFAULT_LR}

        return gnn_params, optimizer_params

    def _get_setting_data(self, setting: SETTING_IDENTIFIER):

        setting_data = self.data.clone()

        train_idx, val_idx, test_idx = self.split_idx["train"], self.split_idx["valid"], self.split_idx["test"]

        old_labels = self.data.y.detach().clone()
        y_feature = self.data.y.detach().clone()
        shuffle_frac = 1.0 - int(setting.split("_")[-1]) / 100.0

        y_feature[train_idx] = shuffle_labels(old_labels[train_idx], frac=shuffle_frac, seed=self.seed)
        y_feature[val_idx] = shuffle_labels(old_labels[val_idx], frac=shuffle_frac, seed=self.seed)
        y_feature[test_idx] = shuffle_labels(old_labels[test_idx], frac=1, seed=SHORTCUT_EXPERIMENT_SEED)

        setting_data.x = torch.cat(tensors=(self.data.x.cpu().detach(), y_feature), dim=1)

        return setting_data


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
        "-t",
        "--test",
        type=str,
        choices=BENCHMARK_EXPERIMENTS_LIST,
        default=LAYER_EXPERIMENT_NAME,
        help="Tests to run.",
    )
    parser.add_argument(
        "-s",
        "--seeds",
        nargs="*",
        type=int,
        choices=list(get_args(GRAPH_EXPERIMENT_SEED)),
        default=list(get_args(GRAPH_EXPERIMENT_SEED)),
        help="Tests to run.",
    )
    parser.add_argument(
        "--settings",
        nargs="*",
        type=str,
        choices=list(get_args(SETTING_IDENTIFIER)),
        default=None,
        help="Tests to run.",
    )
    parser.add_argument(
        "--retrain",
        action="store_true",
        help="Whether to retrain existing models.",
    )
    return parser.parse_args()


GNN_TRAINER_DICT = {
    LAYER_EXPERIMENT_NAME: LayerTestTrainer,
    LABEL_EXPERIMENT_NAME: LabelTestTrainer,
    SHORTCUT_EXPERIMENT_NAME: ShortCutTestTrainer,
}

if __name__ == "__main__":
    args = parse_args()

    for s in args.seeds:
        for architecture, dataset in product(args.architectures, args.datasets):
            trainer = GNN_TRAINER_DICT[args.test](architecture_type=architecture, dataset_name=dataset, seed=s)
            trainer.train_models(settings=args.settings, retrain=args.retrain)
