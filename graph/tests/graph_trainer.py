import os
from abc import ABC
from abc import abstractmethod

import pandas as pd
import torch
from graph.config import DATA_DIR
from graph.config import GNN_DICT
from graph.config import GNN_PARAMS_DEFAULT_DIMENSION
from graph.config import GNN_PARAMS_DEFAULT_DROPOUT
from graph.config import GNN_PARAMS_DEFAULT_LR
from graph.config import GNN_PARAMS_DEFAULT_N_EPOCHS
from graph.config import LAYER_TEST_N_LAYERS
from graph.config import LAYER_TEST_NAME
from graph.config import MODEL_DIR
from graph.config import TORCH_STATE_DICT_FILE_NAME_AT_SEED
from graph.tests.gnn_helpers import get_representations
from graph.tests.gnn_helpers import train_model
from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric import transforms as t


class GraphTrainer(ABC):

    def __init__(self, model_name, dataset_name: str, test_name: str, seeds):
        self.model_name = model_name
        self.seeds = seeds
        self.dataset_name: str = dataset_name
        self.data, self.split_idx = self._preprocess_data()
        self.models = None
        self.test_name = test_name

        self.gnn_params, self.optimizer_params = self._get_gnn_params()

        model_test_path = os.path.join(MODEL_DIR, self.test_name)
        if not os.path.isdir(model_test_path):
            os.mkdir(model_test_path)

        model_dataset_path = os.path.join(model_test_path, self.dataset_name)
        if not os.path.isdir(model_dataset_path):
            os.mkdir(model_dataset_path)

        self.models_path = os.path.join(model_dataset_path, self.model_name)
        if not os.path.isdir(self.models_path):
            os.mkdir(self.models_path)

    # TODO: set up way to read in params which may be determined by graphgym
    @abstractmethod
    def _get_gnn_params(self):
        pass

    @abstractmethod
    def _preprocess_data(self):
        pass

    def _check_pretrained(self):
        missing_seeds = []
        for seed in self.seeds:
            if not os.path.exists(os.path.join(self.models_path, TORCH_STATE_DICT_FILE_NAME_AT_SEED(seed))):
                missing_seeds.append(seed)

        return missing_seeds

    def _load_models(self):
        models = dict()
        for seed in self.seeds:
            models[seed] = GNN_DICT[self.model_name](**self.gnn_params)
            model_file = os.path.join(self.models_path, TORCH_STATE_DICT_FILE_NAME_AT_SEED(seed))

            if not os.path.isfile(model_file):
                return FileNotFoundError(f"Model File for seed {seed} does not exist")

            models[seed].load_state_dict(torch.load(model_file))

        return models

    @abstractmethod
    def train_models(self):
        pass

    @abstractmethod
    def get_test_representations(self):
        pass


class LayerTestTrainer(GraphTrainer):

    def __init__(self, model_name, dataset_name, seeds):
        GraphTrainer.__init__(
            self, model_name=model_name, dataset_name=dataset_name, seeds=seeds, test_name=LAYER_TEST_NAME
        )

    def _preprocess_data(self):
        # TODO: This is assuming an OGB dataset, consider multiple cases if non-obg data is used
        dataset = PygNodePropPredDataset(
            name=self.dataset_name, transform=t.Compose([t.ToUndirected(), t.ToSparseTensor()]), root=DATA_DIR
        )

        return dataset[0], dataset.get_idx_split()

    def _get_gnn_params(self):

        gnn_params = {
            "num_layers": LAYER_TEST_N_LAYERS,
            "in_channels": self.data.num_features,
            "hidden_channels": GNN_PARAMS_DEFAULT_DIMENSION,
            "dropout": GNN_PARAMS_DEFAULT_DROPOUT,
        }
        optimizer_params = {"epochs": GNN_PARAMS_DEFAULT_N_EPOCHS, "lr": GNN_PARAMS_DEFAULT_LR}

        return gnn_params, optimizer_params

    def train_models(self, training_seeds=None):

        if training_seeds is None:
            training_seeds = self.seeds

        for seed in training_seeds:
            model = GNN_DICT[self.model_name](**self.gnn_params)
            save_path = os.path.join(self.models_path, TORCH_STATE_DICT_FILE_NAME_AT_SEED(seed))
            train_results = train_model(model, self.data, self.split_idx, seed, self.optimizer_params, save_path)
            print(train_results[-1])

            df_train = pd.DataFrame(
                train_results, columns=["Epoch", "Loss", "Training_Accuracy", "Validation_Accuracy"]
            )
            df_train.to_csv(os.path.join(self.models_path, f"train_results_s{seed}.csv"), index=False)

    def get_test_representations(self):

        models = self._load_models()
        reps_dict = dict()

        for seed in self.seeds:
            reps_dict[seed] = get_representations(
                models[seed], self.data, self.split_idx["test"], self.gnn_params["num_layers"]
            )

        return reps_dict
