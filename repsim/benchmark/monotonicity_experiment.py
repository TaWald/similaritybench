from repsim.benchmark.abstract_experiment import AbstractExperiment
from repsim.benchmark.registry import TrainedModel
from repsim.measures.utils import SimilarityMeasure


class MonotonicityExperiment(AbstractExperiment):
    def __init__(
        self,
        grouped_models: list[TrainedModel],
        measures: list[SimilarityMeasure],
        representation_dataset: str,
        storage_path: str | None = None,
        meta_data: dict | None = None,
        **kwargs,
    ) -> None:
        """Collect all the models and datasets to be used in the experiment"""
        self.meta_data: dict = meta_data
        self.groups_of_models: tuple[list[TrainedModel]] = grouped_models
        self.measures = measures
        self.representation_dataset = representation_dataset
        self.kwargs = kwargs
        self.storage_path = storage_path

    def eval(self) -> list[dict]:
        """Evaluate the results of the experiment"""
        raise NotImplementedError

    def run(self) -> None:
        """Run the experiment. Results can be accessed afterwards via the .results attribute"""
        raise NotImplementedError
