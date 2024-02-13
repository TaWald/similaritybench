import json
import os
import pickle
from abc import ABC
from abc import abstractmethod
from typing import List

import numpy as np
from graph.config import DEFAULT_SEEDS
from graph.config import LAYER_TEST_N_LAYERS
from graph.config import LAYER_TEST_NAME
from graph.config import MEASURE_DICT
from graph.config import MEASURE_DICT_FUNC_KEY
from graph.config import MEASURE_DICT_PREP_KEY
from graph.config import RES_DIR
from graph.config import SIMILARITIES_FILE_NAME
from graph.config import TEST_RESULTS_JSON_NAME
from graph.tests.graph_trainer import LayerTestTrainer

# from graph.config import *


class GraphTest(ABC):

    def __init__(self, model_name: str, dataset_name: str, test_name: str, seeds: List[int] = DEFAULT_SEEDS):
        self.model_name = model_name
        self.seeds = seeds
        self.dataset_name = dataset_name
        self.test_name = test_name

        test_results_path = os.path.join(RES_DIR, self.test_name)
        if not os.path.isdir(test_results_path):
            os.mkdir(test_results_path)

        # TODO: write separate naming convention function in config
        self.results_file_path = os.path.join(test_results_path, TEST_RESULTS_JSON_NAME)

        similarities_dataset_path = os.path.join(test_results_path, self.dataset_name)
        if not os.path.isdir(similarities_dataset_path):
            os.mkdir(similarities_dataset_path)

        self.similarities_path = os.path.join(similarities_dataset_path, self.model_name)
        if not os.path.isdir(self.similarities_path):
            os.mkdir(self.similarities_path)

    def _save_results(self, measure_name, score):

        if not os.path.isfile(self.results_file_path):
            results_dict = dict({self.dataset_name: {self.model_name: {measure_name: score}}})
            print(results_dict)
            with open(self.results_file_path, "w") as f:
                json.dump(results_dict, f, indent=4)

        else:
            with open(self.results_file_path, "r") as f:
                results_dict = json.load(f)

            if self.dataset_name not in results_dict.keys():
                results_dict[self.dataset_name] = {self.model_name: {measure_name: score}}
            elif self.model_name not in results_dict[self.dataset_name].keys():
                results_dict[self.dataset_name][self.model_name] = {measure_name: score}
            else:
                results_dict[self.dataset_name][self.model_name][measure_name]: score

            with open(self.results_file_path, "w") as f:
                json.dump(results_dict, f, indent=4)

    def _save_similarities(self, similarity_dict):
        with open(os.path.join(self.similarities_path, SIMILARITIES_FILE_NAME), "wb") as f:
            pickle.dump(similarity_dict, f)

    @abstractmethod
    def test_measure(self, measure):
        pass


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
