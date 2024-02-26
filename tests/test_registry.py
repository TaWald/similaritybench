import pytest
from repsim.benchmark.registry import all_trained_nlp_models
from repsim.benchmark.registry import ModelRepresentations


@pytest.mark.gpu
@pytest.mark.slow
def test_nlp_model_representation_extraction():
    model_cfg = all_trained_nlp_models()[0]
    reps = model_cfg.get_representation(
        representation_dataset=None,
        dataset_path="sst2",
        dataset_config=None,
        dataset_split="test",
        device="cuda:0",
        token_pos=None,
    )
    assert isinstance(reps, ModelRepresentations)
