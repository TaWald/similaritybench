import argparse
import os
from abc import ABC
from abc import abstractmethod
from itertools import product
from typing import get_args
from typing import List

import pandas as pd
import torch
from graphs.config import DATA_DIR
from graphs.config import DATASET_LIST
from graphs.config import GNN_DICT
from graphs.config import GNN_LIST
from graphs.config import GNN_PARAMS_DEFAULT_DIMENSION
from graphs.config import GNN_PARAMS_DEFAULT_DROPOUT
from graphs.config import GNN_PARAMS_DEFAULT_LR
from graphs.config import GNN_PARAMS_DEFAULT_N_EPOCHS
from graphs.config import GNN_PARAMS_DEFAULT_N_LAYERS
from graphs.config import LAYER_TEST_N_LAYERS
from graphs.config import LAYER_TEST_NAME
from graphs.config import MODEL_DIR
from graphs.config import NN_TESTS_LIST
from graphs.config import TORCH_STATE_DICT_FILE_NAME_SETTING_SEED
from graphs.gnn import get_representations
from graphs.gnn import train_model
from graphs.tests.tools import shuffle_labels
from ogb.nodeproppred import PygNodePropPredDataset
from repsim.benchmark.types_globals import EXPERIMENT_DICT
from repsim.benchmark.types_globals import EXPERIMENT_IDENTIFIER
from repsim.benchmark.types_globals import GRAPH_ARCHITECTURE_TYPE
from repsim.benchmark.types_globals import GRAPH_DATASET_TRAINED_ON
from repsim.benchmark.types_globals import GRAPH_EXPERIMENT_SEED
from repsim.benchmark.types_globals import LABEL_TEST_NAME
from repsim.benchmark.types_globals import SETTING_IDENTIFIER
from repsim.benchmark.types_globals import STANDARD_SETTING
from torch_geometric import transforms as t


class GraphTrainer(ABC):

    def __init__(
        self,
        architecture_type: GRAPH_ARCHITECTURE_TYPE,
        dataset_name: GRAPH_DATASET_TRAINED_ON,
        test_name: EXPERIMENT_IDENTIFIER,
        seed: GRAPH_EXPERIMENT_SEED,
    ):

        self.test_name = test_name
        self.settings = EXPERIMENT_DICT[test_name]
        self.architecture_type = architecture_type
        self.seed = seed
        self.dataset_name: str = dataset_name
        self.data, self.n_classes, self.split_idx = self._get_data()
        self.models = dict()
        self.settings = EXPERIMENT_DICT[test_name]

        self.gnn_params, self.optimizer_params = self._get_gnn_params()

        model_test_path = os.path.join(MODEL_DIR, self.test_name)
        if not os.path.isdir(model_test_path):
            os.mkdir(model_test_path)

        model_dataset_path = os.path.join(model_test_path, self.dataset_name)
        if not os.path.isdir(model_dataset_path):
            os.mkdir(model_dataset_path)

        self.models_path = os.path.join(model_dataset_path, self.architecture_type)
        if not os.path.isdir(self.models_path):
            os.mkdir(self.models_path)

    # TODO: set up way to read in params which may be determined by graphgym
    @abstractmethod
    def _get_gnn_params(self):
        pass

    def _check_pretrained(self, settings):
        missing_settings = []
        for setting in settings:
            if not os.path.exists(
                os.path.join(self.models_path, TORCH_STATE_DICT_FILE_NAME_SETTING_SEED(setting, self.seed))
            ):
                missing_settings.append(setting)

        return missing_settings

    def _load_model(self, setting):
        model = GNN_DICT[self.architecture_type](**self.gnn_params)
        model_file = os.path.join(self.models_path, TORCH_STATE_DICT_FILE_NAME_SETTING_SEED(setting, self.seed))

        print(model_file)
        if not os.path.isfile(model_file):
            raise FileNotFoundError(f"Model File for seed {self.seed} does not exist")

        model.load_state_dict(torch.load(model_file))

        return model

    def _get_data(self):
        # TODO: This is assuming an OGB dataset, consider multiple cases if non-obg data is used

        pyg_dataset = PygNodePropPredDataset(
            name=str(self.dataset_name), transform=t.Compose([t.ToUndirected(), t.ToSparseTensor()]), root=DATA_DIR
        )

        return pyg_dataset[0], pyg_dataset.num_classes, pyg_dataset.get_idx_split()

    def train_models(self, settings: List[SETTING_IDENTIFIER] = None, retrain: bool = False):

        if settings is None:
            settings = self.settings
        else:
            for setting in settings:
                assert setting in self.settings, f"Setting {setting} is invalid, valid settings are {self.settings}"

        if not retrain:
            settings = self._check_pretrained(settings)

        for setting in settings:
            self.models[setting] = self._train_model(setting)

    @abstractmethod
    def _train_model(self, setting: SETTING_IDENTIFIER):
        pass

    @abstractmethod
    def get_test_representations(self, setting: SETTING_IDENTIFIER):
        pass


class LayerTestTrainer(GraphTrainer):

    def __init__(
        self,
        architecture_type: GRAPH_ARCHITECTURE_TYPE,
        dataset_name: GRAPH_DATASET_TRAINED_ON,
        seed: GRAPH_EXPERIMENT_SEED,
        n_layers: int = None,
    ):
        self.n_layers = LAYER_TEST_N_LAYERS if n_layers is None else n_layers

        GraphTrainer.__init__(
            self, architecture_type=architecture_type, dataset_name=dataset_name, seed=seed, test_name=LAYER_TEST_NAME
        )

    def _get_gnn_params(self):

        gnn_params = {
            "num_layers": self.n_layers,
            "in_channels": self.data.num_features,
            "hidden_channels": GNN_PARAMS_DEFAULT_DIMENSION,
            "dropout": GNN_PARAMS_DEFAULT_DROPOUT,
            "out_channels": self.n_classes,
            # "norm": GNN_PARAMS_DEFAULT_NORM,
        }
        optimizer_params = {"epochs": GNN_PARAMS_DEFAULT_N_EPOCHS, "lr": GNN_PARAMS_DEFAULT_LR}

        return gnn_params, optimizer_params

    def _train_model(self, setting: SETTING_IDENTIFIER):

        model = GNN_DICT[self.architecture_type](**self.gnn_params)
        save_path = os.path.join(self.models_path, TORCH_STATE_DICT_FILE_NAME_SETTING_SEED(setting, self.seed))
        train_results = train_model(model, self.data, self.split_idx, self.seed, self.optimizer_params, save_path)
        print(train_results[-1])

        # TODO: outsource function to create results frames, create global functions/vars for colnames and result
        #  file name
        df_train = pd.DataFrame(train_results, columns=["Epoch", "Loss", "Training_Accuracy", "Validation_Accuracy"])
        df_train.to_csv(os.path.join(self.models_path, f"train_results_{setting}_s{self.seed}.csv"), index=False)

        return model

    def get_test_representations(self, setting: SETTING_IDENTIFIER):

        model = self._load_model(setting)
        reps = get_representations(
            model=model,
            data=self.data,
            test_idx=self.split_idx["test"],
            layer_ids=list(range(self.gnn_params["num_layers"])),
        )

        return reps


class LabelTestTrainer(GraphTrainer):

    def __init__(
        self,
        architecture_type: GRAPH_ARCHITECTURE_TYPE,
        dataset_name: GRAPH_DATASET_TRAINED_ON,
        seed: GRAPH_EXPERIMENT_SEED,
        n_layers: int = None,
    ):
        self.n_layers = LAYER_TEST_N_LAYERS if n_layers is None else n_layers
        GraphTrainer.__init__(
            self, architecture_type=architecture_type, dataset_name=dataset_name, seed=seed, test_name=LABEL_TEST_NAME
        )

    def _get_gnn_params(self):

        gnn_params = {
            "num_layers": GNN_PARAMS_DEFAULT_N_LAYERS,
            "in_channels": self.data.num_features,
            "hidden_channels": GNN_PARAMS_DEFAULT_DIMENSION,
            "dropout": GNN_PARAMS_DEFAULT_DROPOUT,
            "out_channels": self.n_classes,
            # "norm": GNN_PARAMS_DEFAULT_NORM,
        }
        optimizer_params = {"epochs": GNN_PARAMS_DEFAULT_N_EPOCHS, "lr": GNN_PARAMS_DEFAULT_LR}

        return gnn_params, optimizer_params

    def _train_model(self, setting):

        print(f"Train {self.architecture_type} on {self.dataset_name} in {setting} setting.")

        setting_data = self.data.clone()

        if setting != STANDARD_SETTING:
            old_labels = self.data.y.detach().clone()
            shuffle_frac = int(setting.split("_")[-1]) / 100.0
            setting_data.y = shuffle_labels(old_labels, frac=shuffle_frac)

        model = GNN_DICT[self.architecture_type](**self.gnn_params)
        save_path = os.path.join(self.models_path, TORCH_STATE_DICT_FILE_NAME_SETTING_SEED(setting, self.seed))
        train_results = train_model(model, setting_data, self.split_idx, self.seed, self.optimizer_params, save_path)

        df_train = pd.DataFrame(train_results, columns=["Epoch", "Loss", "Training_Accuracy", "Validation_Accuracy"])
        df_train.to_csv(os.path.join(self.models_path, f"train_results_s{self.seed}.csv"), index=False)

    def get_test_representations(self, setting: SETTING_IDENTIFIER):

        model = self._load_model(setting)
        setting_data = self.data.clone()

        if setting != STANDARD_SETTING:
            old_labels = self.data.y.detach().clone()
            shuffle_frac = int(setting.split("_")[-1]) / 100.0
            setting_data.y = shuffle_labels(old_labels, frac=shuffle_frac)

        reps = get_representations(
            model=model,
            data=setting_data,
            test_idx=self.split_idx["test"],
            layer_ids=list(range(self.gnn_params["num_layers"])),
        )

        return reps


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
        choices=NN_TESTS_LIST,
        default=LAYER_TEST_NAME,
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


GNN_TRAINER_DICT = {LAYER_TEST_NAME: LayerTestTrainer, LABEL_TEST_NAME: LabelTestTrainer}

if __name__ == "__main__":
    args = parse_args()

    for s in args.seeds:
        for architecture, dataset in product(args.architectures, args.datasets):
            trainer = GNN_TRAINER_DICT[args.test](architecture_type=architecture, dataset_name=dataset, seed=s)
            trainer.train_models(settings=args.settings, retrain=args.retrain)