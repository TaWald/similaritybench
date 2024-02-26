from graphs.tests.graph_trainer import GraphTrainer
from repsim.utils import ModelRepresentations


def get_graph_representations(
    architecture_name: str,
    train_dataset: str,
    seed_id: int,
    experiment_identifier: str | None,
    setting_identifier: str | None,
    representation_dataset: str,
) -> ModelRepresentations:
    """
    Finds the representations for a given model and dataset.
    :param architecture_name: The name of the architecture.
    :param seed_id: The id of the model.
    :param dataset: The name of the dataset.
    """

    graph_trainer = GraphTrainer(
        setting_identifier=setting_identifier, model_name=architecture_name, dataset_name=train_dataset, seed=seed_id
    )
    all_single_layer_reps = graph_trainer.get_test_representations()

    model_reps = ModelRepresentations(
        setting_identifier=setting_identifier,
        architecture_name=architecture_name,
        seed_id=seed_id,
        train_dataset=train_dataset,
        representation_dataset=representation_dataset,
        representations=tuple(all_single_layer_reps),
    )
    return model_reps
