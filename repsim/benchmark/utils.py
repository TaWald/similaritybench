import jsonlines
import numpy as np
import numpy.typing as npt
from loguru import logger
from repsim.benchmark.paths import EXPERIMENT_RESULTS_PATH
from repsim.benchmark.registry import TrainedModelRep


class TwoGroupExperiment:
    def __init__(
        self,
        models_group_a: list[TrainedModelRep] | None,
        models_group_b: list[TrainedModelRep] | None,
        measures: list[callable] | None,
    ) -> None:
        """
        Experiment where the goal is that the each member of group A is more similar to each other than to any member of group B.
        The user has to make sure each set of models is compatible with the measures used.

        :param models_group_a: List of trained models that are part of group A
        :param models_group_b: List of trained models that are part of group B
        :param measures: List of measures that should be used to compare the representations
        """
        self.models_a = models_group_a
        self.models_b = models_group_b
        self.measures = measures

    def _assure_everythings_ready(self) -> None:
        """Test everything is present and ready to run."""
        assert self.models_a is not None and len(self.models_a) > 0
        assert self.models_b is not None and len(self.models_b) > 0
        assert self.measures is not None and len(self.measures) > 0
        return

    def _measure_acc(in_group: list[float], cross_group: list[float]) -> float:
        """
        Measure the accuracy of the separation of the groups.
        """
        in_group_mean = np.mean(in_group)
        cross_group_mean = np.mean(cross_group)
        return (in_group_mean - cross_group_mean) / (in_group_mean + cross_group_mean)

    def run(self) -> dict[str, dict]:
        """
        Run the experiment
        """
        self._assure_everythings_ready()
        N_a = len(self.models_a)
        N_b = len(self.models_b)

        all_models = self.models_a + self.models_b
        results = {}
        for measure in self.measures:
            vals = np.full((len(all_models), len(all_models)), fill_value=np.nan, dtype=np.float32)
            for i, first in enumerate(all_models):
                for j, second in enumerate(all_models):
                    # All metrics should be symmetric
                    vals[i, j] = measure(first, second)
                    vals[j, i] = vals[i, j]
            # In group values are redundant. Only want the upper triangle of the matrix
            in_group_a = vals[:N_a, :N_a][np.triu_indices(N_a, k=1)]
            in_group_b = vals[-N_b:, -N_b:][np.triu_indices(N_b, k=1)]
            cross_group = vals[:N_a, N_a:].flatten()  # All cross group values are useful.

            # ----------------- Measure how well the groups are separated ---------------- #
            acc_a = self._measure_acc(in_group_a, cross_group)
            acc_b = self._measure_acc(in_group_b, cross_group)
            results[measure.__name__] = {
                "group_a": self.models_a,
                "group_b": self.models_b,
                "values": vals,
                "in_group_a": in_group_a,
                "in_group_b": in_group_b,
                "cross_group": cross_group,
                "acc": (acc_a + acc_b) / 2,
            }
        return results


class Result:
    def __init__(self, identifier: str) -> None:
        self.basedir = EXPERIMENT_RESULTS_PATH / identifier
        self.data = {"numpy_vals": [], "jsonable": []}

    def save(self) -> None:
        self.basedir.mkdir(exist_ok=True, parents=True)
        logger.info(f"Writing results to {self.basedir}")

        with jsonlines.open(self.basedir / "results.jsonl", mode="w") as writer:
            writer.write_all(self.data["jsonable"])

        # arr_0 will correspond to the first row in the jsonl file
        np.savez(self.basedir / "numpy_vals.npz", *self.data["numpy_vals"])

    def load(self) -> None:
        raise NotImplementedError

    def add(self, numpy_vals: npt.NDArray, **others) -> None:
        self.data["numpy_vals"].append(numpy_vals)
        self.data["jsonable"].append({str(key): str(val) for key, val in others.items()})
