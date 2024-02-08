import os

from repsim.utils import ModelRepresentations
from vision.util import data_structs as ds
from vision.util import find_architectures as fa
from vision.util import default_params as dp
from paths import get_experiments_path


def get_possible_model_ids(architecture_name: str, dataset: str) -> list[int]:
    """
    Get all model ids for a given architecture and dataset.
    """
    architecture: ds.BaseArchitecture = ds.BaseArchitecture(architecture_name)
    dataset: ds.Dataset = ds.Dataset(dataset)

    arch_params = dp.get_default_arch_params(dataset)
    p: ds.Params = dp.get_default_parameters(architecture.value, dataset)

    
    experiments_path = os.join(get_experiments_path(), "vision")
    


    return [int(m.model_id) for m in ds.get_model_representations(architecture_name, dataset)]


def get_vision_representations(architecture_name: ds.BaseArchitecture, model_id: int, dataset: ds.Dataset) -> ModelRepresentations:
    """
    Finds the representations for a given model and dataset.
    :param architecture_name: The name of the architecture.
    :param model_id: The id of the model.
    :param dataset: The name of the dataset.
    """
    
    return ds.get_model_representations(architecture_name, dataset, model_id)
