import json
import pickle

from graph.config import *
from abc import ABC, abstractmethod
from typing import List


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
