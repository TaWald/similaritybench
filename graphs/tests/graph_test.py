import json
import os
import pickle
from abc import ABC
from abc import abstractmethod
from typing import List

import numpy as np
from graphs.config import DEFAULT_SEEDS
from graphs.config import LAYER_TEST_N_LAYERS
from graphs.config import LAYER_TEST_NAME
from graphs.config import MEASURE_DICT
from graphs.config import MEASURE_DICT_FUNC_KEY
from graphs.config import RES_DIR
from graphs.config import SIMILARITIES_FILE_NAME
from graphs.config import TEST_RESULTS_JSON_NAME
from graphs.tests.graph_trainer import LayerTestTrainer


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
    def _get_representations(self):
        pass

    @abstractmethod
    def _build_similarity_matrices(self, rep_dict, measure):
        pass

    @abstractmethod
    def _get_performance_score(self, similarities):
        pass

    def test_measures(self, measures, save_similarities=True):

        rep_dict = self._get_representations()

        for measure_name in measures:
            similarities = self._build_similarity_matrices(
                rep_dict, MEASURE_DICT[measure_name][MEASURE_DICT_FUNC_KEY]
            )

            res_score = self._get_performance_score(similarities)

            print(f"The test score for the {measure_name} measure is {res_score}")

            if save_similarities:
                self._save_similarities(similarity_dict=similarities)

            self._save_results(measure_name, res_score)


class LayerTest(GraphTest):

    def __init__(self, model_name: str, dataset_name: str, n_layers: int = LAYER_TEST_N_LAYERS):
        GraphTest.__init__(self, model_name=model_name, dataset_name=dataset_name, test_name=LAYER_TEST_NAME)

        self.n_layers = n_layers

    def _get_representations(self):
        graph_trainer = LayerTestTrainer(self.model_name, self.dataset_name, self.seeds)
        return graph_trainer.get_test_representations()

    def _build_similarity_matrices(self, rep_dict, measure):

        similarities = dict()

        for seed in self.seeds:

            curr_reps = rep_dict[seed]
            print(curr_reps[0].shape)
            # this builds the similarity matrix for given seed/model ID
            similarities[seed] = np.zeros((self.n_layers, self.n_layers))
            for i in range(self.n_layers):
                for j in range(i + 1, self.n_layers):
                    similarities[seed][i, j] = measure(curr_reps[i], curr_reps[j], shape="nd")
            print(similarities[seed])

        return similarities

    def _get_performance_score(self, similarities):

        n_violations = 0
        n_comb_count = 0

        # this computes the "accuracy" score based on the given matrix, or rather counts the violations
        # TODO: wrap this kind of accuracy into separate function, and add rank correlation

        for seed in self.seeds:
            for i in range(self.n_layers):
                for j in range(i + 1, self.n_layers):

                    for k in range(i, j):
                        for l in range(k + 1, j + 1):
                            n_comb_count += 1
                            if similarities[seed][i, j] < similarities[seed][k, l]:
                                n_violations += 1

        return 1 - n_violations / n_comb_count
