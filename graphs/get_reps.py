from graphs.tests.graph_trainer import LabelTestTrainer
from graphs.tests.graph_trainer import LayerTestTrainer
from graphs.tests.graph_trainer import ShortCutTestTrainer
from repsim.benchmark.types_globals import BENCHMARK_EXPERIMENTS_LIST
from repsim.benchmark.types_globals import EXPERIMENT_DICT
from repsim.benchmark.types_globals import GRAPH_ARCHITECTURE_TYPE
from repsim.benchmark.types_globals import GRAPH_DATASET_TRAINED_ON
from repsim.benchmark.types_globals import GRAPH_EXPERIMENT_SEED
from repsim.benchmark.types_globals import LABEL_EXPERIMENT_NAME
from repsim.benchmark.types_globals import LAYER_EXPERIMENT_NAME
from repsim.benchmark.types_globals import SETTING_IDENTIFIER
from repsim.benchmark.types_globals import SHORTCUT_EXPERIMENT_NAME
from repsim.measures.utils import ND_SHAPE
from repsim.utils import ModelRepresentations
from repsim.utils import SingleLayerRepresentation


GRAPH_EXPERIMENT_TRAINER_DICT = {
    LABEL_EXPERIMENT_NAME: LabelTestTrainer,
    LAYER_EXPERIMENT_NAME: LayerTestTrainer,
    SHORTCUT_EXPERIMENT_NAME: ShortCutTestTrainer,
}


def get_graph_representations(
    architecture_name: GRAPH_ARCHITECTURE_TYPE,
    train_dataset: GRAPH_DATASET_TRAINED_ON,
    seed: GRAPH_EXPERIMENT_SEED,
    setting_identifier: SETTING_IDENTIFIER,
    representation_dataset: GRAPH_DATASET_TRAINED_ON,
) -> ModelRepresentations:
    """
    Finds the representations for a given model and dataset.
    :param architecture_name: The name of the architecture.
    :param seed_id: The id of the model.
    :param train_dataset: The name of the dataset.
    :param experiment_identifier:
    :param setting_identifier:
    :param representation_dataset:
    """

    experiment_identifier = ""

    for exp in BENCHMARK_EXPERIMENTS_LIST:
        if setting_identifier in EXPERIMENT_DICT[exp]:
            experiment_identifier = exp
            break

    graph_trainer = GRAPH_EXPERIMENT_TRAINER_DICT[experiment_identifier](
        architecture_type=architecture_name, dataset_name=train_dataset, seed=seed
    )
    plain_reps = graph_trainer.get_test_representations(setting=setting_identifier)

    all_single_layer_reps = []
    for layer_id, rep in plain_reps.items():
        all_single_layer_reps.append(SingleLayerRepresentation(representation=rep, shape=ND_SHAPE, layer_id=layer_id))

    model_reps = ModelRepresentations(
        setting_identifier=setting_identifier,
        architecture_name=architecture_name,
        seed_id=seed,
        train_dataset=train_dataset,
        representation_dataset=representation_dataset,
        representations=tuple(all_single_layer_reps),
    )
    return model_reps
