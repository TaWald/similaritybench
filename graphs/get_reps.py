from graphs.tests.graph_trainer import EXPERIMENT_IDENTIFIER
from graphs.tests.graph_trainer import GRAPH_ARCHITECTURE_TYPE
from graphs.tests.graph_trainer import GRAPH_DATASET_TRAINED_ON
from graphs.tests.graph_trainer import GRAPH_EXPERIMENT_SEED
from graphs.tests.graph_trainer import LabelTestTrainer
from graphs.tests.graph_trainer import LayerTestTrainer
from graphs.tests.graph_trainer import SETTING_IDENTIFIER
from repsim.benchmark.types_globals import LABEL_TEST_NAME
from repsim.benchmark.types_globals import LAYER_TEST_NAME
from repsim.measures.utils import ND_SHAPE
from repsim.utils import ModelRepresentations
from repsim.utils import SingleLayerRepresentation


GRAPH_TEST_TRAINER_DICT = {LABEL_TEST_NAME: LabelTestTrainer, LAYER_TEST_NAME: LayerTestTrainer}


def get_graph_representations(
    architecture_name: GRAPH_ARCHITECTURE_TYPE,
    train_dataset: GRAPH_DATASET_TRAINED_ON,
    seed_id: GRAPH_EXPERIMENT_SEED,
    experiment_identifier: EXPERIMENT_IDENTIFIER,
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

    graph_trainer = GRAPH_TEST_TRAINER_DICT[experiment_identifier](
        architecture_type=architecture_name, dataset_name=train_dataset, seed=seed_id
    )
    plain_reps = graph_trainer.get_test_representations(setting=setting_identifier)

    all_single_layer_reps = []
    for layer_id, rep in plain_reps.items():
        all_single_layer_reps.append(SingleLayerRepresentation(representation=rep, shape=ND_SHAPE, layer_id=layer_id))

    model_reps = ModelRepresentations(
        setting_identifier=setting_identifier,
        architecture_name=architecture_name,
        seed_id=seed_id,
        train_dataset=train_dataset,
        representation_dataset=representation_dataset,
        representations=tuple(all_single_layer_reps),
    )
    return model_reps
