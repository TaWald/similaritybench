from collections.abc import Sequence

from repsim.benchmark.registry import TrainedModel


def _group_models(
    models: list[TrainedModel],
    differentiation_keys: list[str] | None = None,
) -> list[list[TrainedModel]]:
    """
    Group models based on the differentiation_keys.
    """
    if differentiation_keys is None:
        return [models]

    # We find groups that share the same values for the keys chosen to split by.
    # This is done like so:
    # We create a dictionary of list of trained models.
    # The keys of the dict are tuples of the differentation_keys values.
    # By doing so models having the same differentation key values will be grouped together.
    group_dict = {}
    for model in models:
        key = tuple(getattr(model, key) for key in differentiation_keys)
        if key not in group_dict:
            group_dict[key] = []
        group_dict[key].append(model)

    grouped_models = list(group_dict.values())
    return grouped_models


def _filter_models(
    models: list[TrainedModel],
    filter_key_vals: dict[str, str | list[str]] | None,
) -> list[TrainedModel]:
    """
    Filter models based on the filter_key_vals.
    """
    if filter_key_vals is None:
        return models

    filtered_models = []

    for model in models:
        matches = True
        for key, val in filter_key_vals.items():
            m_val = getattr(model, key)
            if isinstance(val, Sequence):
                if m_val not in val:
                    matches = False
            else:
                if m_val != val:
                    matches = False
        if matches:
            filtered_models.append(model)
    return filtered_models


def get_grouped_trained_models(
    models: list[TrainedModel],
    filter_key_vals: dict[str, str | list[str]] | None,
    differentiation_keys: list[str] | None = None,
) -> list[Sequence[TrainedModel]]:
    """
    Get a dictionary of models grouped by the values of the filter_key_vals.
    """
    # --------------- Remove unwanted Trained Models from the list --------------- #
    if filter_key_vals is None:
        pass
    else:
        models = _filter_models(models, filter_key_vals)
    # ---------- Group them into groups that get compared to each other ---------- #
    if differentiation_keys is None:
        return [models]  # If no differentiation keys are provided, return all models as a single group.
    else:
        return _group_models(models, differentiation_keys)
