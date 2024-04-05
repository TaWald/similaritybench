from abc import abstractmethod

from repsim.measures.utils import SimilarityMeasure


class AbstractExperiment:
    def __init__(
        self,
        measures: list[SimilarityMeasure],
        representation_dataset: str,
        storage_path: str | None = None,
        threads: int = 1,
        cache: bool = False,
        only_extract_reps: bool = False,
    ) -> None:
        """
        Needs the measures to be employed, the dataset to be used and the path where to store the results.
        """
        self.measures: list[SimilarityMeasure] = measures
        self.representation_dataset: str = representation_dataset
        self.storage_path: str = storage_path
        self.cache = cache
        self.threads = threads
        self.only_extract_reps: bool = only_extract_reps

    @abstractmethod
    def eval(self) -> None:
        """Evaluate the results of the experiment"""
        pass

    @abstractmethod
    def run(self) -> None:
        """Run the experiment. Results can be accessed afterwards via the experiment storer."""
