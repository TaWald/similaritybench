import time
from itertools import combinations
from typing import Callable

import numpy as np
from loguru import logger
from registry import ALL_TRAINED_MODELS
from repsim.benchmark.registry import TrainedModel
from repsim.benchmark.utils import Result
from repsim.measures import distance_correlation
from repsim.measures.cca import pwcca
from repsim.measures.cca import svcca
from repsim.measures.cka import centered_kernel_alignment
from repsim.measures.eigenspace_overlap import eigenspace_overlap_score
from repsim.measures.gulp import gulp
from repsim.measures.procrustes import orthogonal_procrustes
from repsim.measures.procrustes import permutation_procrustes
from repsim.measures.rsa import representational_similarity_analysis
from scipy.stats import spearmanr
from vision.util.file_io import save_json


def flatten(xss):
    return [x for xs in xss for x in xs]


class OrdinalGroupSeparationExperiment:
    def __init__(
        self,
        experiment_identifier: str,
        models: list[TrainedModel],
        group_splitting_func: Callable,
        measures: list[Callable],
        representation_dataset: str,
        **kwargs,
    ) -> None:
        """Collect all the models and datasets to be used in the experiment"""
        self.groups_of_models: tuple[list[TrainedModel]] = group_splitting_func(
            models
        )  # Expects lists of models ordered by expected ordinality
        self.measures = measures
        self.representation_dataset = representation_dataset
        self.kwargs = kwargs
        self.results = Result(experiment_identifier)

        logger.add(self.results.basedir / "{time}.log")  # Not sure where this needs to go...

    def run(self) -> None:
        """Run the experiment. Results can be accessed afterwards via the .results attribute"""
        measure_index_json = {cnt: k.__class__.__name__ for cnt, k in enumerate(self.measures)}
        save_json(
            measure_index_json,
            self.results.basedir / "measure_index.json",
        )
        flat_models = flatten(self.groups_of_models)
        all_sims = np.full(
            (len(flat_models), len(flat_models), len(self.measures)),
            fill_value=np.nan,
            dtype=np.float32,
        )
        # ToDo: Pass "SingleLayerRepresentation" objects to the metrics instead of the raw representations
        # ToDo: Issues during development
        #  - Choice of shortcut (how many different groups, how many seeds)
        #  - Saving and loading of representations <--
        #   Maybe saving intermediate things would make sense?
        #  - Saving and loading of results!

        #   Would allow for more flexibility in dynamic saving and loading of representations?
        for cnt_a, model_a in enumerate(flat_models):
            model_reps_a = model_a.get_representation(self.representation_dataset, **self.kwargs)
            reps_a = model_reps_a.representations[-1].representation  # Only use the last layer
            shape_a = model_reps_a.representations[-1].shape

            for cnt_b, model_b in enumerate(flat_models):
                if cnt_a > cnt_b:
                    continue
                model_reps_b = model_b.get_representation(self.representation_dataset, **self.kwargs)
                reps_b = model_reps_b.representations[-1].representation  # Only use the last layer

                for cnt_m, measure in enumerate(self.measures):
                    start_time = time.perf_counter()
                    try:
                        sim = measure(reps_a, reps_b, shape_a)
                        all_sims[cnt_a, cnt_b, cnt_m] = sim
                        all_sims[cnt_b, cnt_a, cnt_m] = sim
                        logger.info(
                            f"'{measure.__name__}' comparison for '{str(model_a)}' and '{str(model_b)}' completed in {time.perf_counter() - start_time:.1f} seconds."
                        )
                    except Exception as e:
                        logger.error(
                            f"'{measure.__name__}' comparison for '{str(model_a)}' and '{str(model_b)}' failed."
                        )
                        logger.error(e)

        np.save(self.results.basedir / "all_sims.npy", all_sims)

        return all_sims


"""
SimCKA for shortcut of seed 0 only.
array([[1.        , 0.88063586, 0.8452915 , 0.78969693, 0.29663917],
       [0.88063586, 1.        , 0.874399  , 0.83381474, 0.3271054 ],
       [0.8452915 , 0.874399  , 1.        , 0.87857467, 0.37550303],
       [0.78969693, 0.83381474, 0.87857467, 1.        , 0.44088718],
       [0.29663917, 0.3271054 , 0.37550303, 0.44088718, 1.        ]],
      dtype=float32)

array([[1.        , 0.902106  , 0.88063586, 0.8806002 , 0.8452915 , 0.8495685 , 0.78969693, 0.79049397, 0.29663917, 0.29089352],
       [0.902106  , 1.        , 0.88480216, 0.884095  , 0.84894454, 0.85240364, 0.7934904 , 0.79633653, 0.29962784, 0.2929612 ],
       [0.88063586, 0.88480216, 1.        , 0.9024318 , 0.874399  , 0.87824595, 0.83381474, 0.8330886 , 0.3271054 , 0.320484  ],
       [0.8806002 , 0.884095  , 0.9024318 , 1.        , 0.87760025, 0.88188475, 0.8338022 , 0.833005  , 0.32659015, 0.31945845],
       [0.8452915 , 0.84894454, 0.874399  , 0.87760025, 1.        , 0.9094963 , 0.87857467, 0.8773353 , 0.37550303, 0.36777332],
       [0.8495685 , 0.85240364, 0.87824595, 0.88188475, 0.9094963 , 1.        , 0.8756632 , 0.87609726, 0.37522557, 0.36792675],
       [0.78969693, 0.7934904 , 0.83381474, 0.8338022 , 0.87857467, 0.8756632 , 1.        , 0.9111501 , 0.44088718, 0.43294594],
       [0.79049397, 0.79633653, 0.8330886 , 0.833005  , 0.8773353 , 0.87609726, 0.9111501 , 1.        , 0.4439877 , 0.4365481 ],
       [0.29663917, 0.29962784, 0.3271054 , 0.32659015, 0.37550303, 0.37522557, 0.44088718, 0.4439877 , 1.        , 0.94804287],
       [0.29089352, 0.2929612 , 0.320484  , 0.31945845, 0.36777332, 0.36792675, 0.43294594, 0.4365481 , 0.94804287, 1.        ]],
      dtype=float32)


"""

if __name__ == "__main__":

    # ToDo: Issues during development
    # - Choice of shortcut (how many different groups, how many seeds)
    # - Saving and loading of representations <--  Saving intermediate representations would make sense?
    # - Saving and loading of results!
    #

    subset_of_vision_models = [
        m
        for m in ALL_TRAINED_MODELS
        if (m.domain == "VISION")
        and (m.architecture == "ResNet18")
        and (
            m.train_dataset
            in [
                "ColorDot_100_CIFAR10DataModule",
                "ColorDot_75_CIFAR10DataModule",
                "ColorDot_50_CIFAR10DataModule",
                "ColorDot_25_CIFAR10DataModule",
                "ColorDot_0_CIFAR10DataModule",
            ]
        )
        and (m.additional_kwargs["seed_id"] <= 1)
    ]
    subset_of_nlp_models = [
        m
        for m in ALL_TRAINED_MODELS
        if (m.domain == "NLP") and (m.architecture == "BERT") and (m.train_dataset == "SST2")
    ]
    subset_of_graph_models = []

    # experiment = SameLayerExperiment(subset_of_vision_models, [centered_kernel_alignment], "CIFAR10")
    def vision_group_split_func(models: list[TrainedModel]) -> tuple[list[TrainedModel]]:
        """Split the models into groups based on the dataset they were trained on."""
        model_train_ds = sorted(list(set([m.train_dataset for m in models])), key=lambda x: int(x.split("_")[1]))
        model_groups = []
        for tr_ds in model_train_ds:
            group = [m for m in models if m.train_dataset == tr_ds]
            model_groups.append(group)
        return tuple(model_groups)

    experiment = OrdinalGroupSeparationExperiment(
        experiment_identifier="Vision_ColorDot_Separation",
        models=subset_of_vision_models,
        group_splitting_func=vision_group_split_func,
        measures=[
            centered_kernel_alignment,  # 2.4 seconds
            # orthogonal_procrustes,  # 77.2 seconds
            # permutation_procrustes,  # 16.2 seconds
            # eigenspace_overlap_score,  # 245 seconds for one comp! -- 4 minutes
            # gulp,  # failed
            # svcca,  #  157.5/129.7 seconds
            # pwcca,  # failed?
            # representational_similarity_analysis, # 94.5 seconds
            # distance_correlation,  # 75.2
        ],
        representation_dataset="ColorDot_0_CIFAR10DataModule",
    )
    result = experiment.run()
    print(result)
    print(0)
