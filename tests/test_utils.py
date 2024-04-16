import pytest
import torch
from repsim.benchmark.registry import all_trained_nlp_models
from repsim.utils import ModelRepresentations
from repsim.utils import Prediction


@pytest.mark.gpu
@pytest.mark.slow
def test_nlp_model_representation_extraction():
    base_model = [m for m in all_trained_nlp_models() if m.identifier == "Normal"][0]
    reps = base_model.get_representation("sst2")
    assert isinstance(reps, ModelRepresentations)


@pytest.mark.gpu
@pytest.mark.slow
def test_nlp_shortcut_model_representation_extraction():
    model = [m for m in all_trained_nlp_models() if m.identifier == "Shortcut_0"][0]
    reps = model.get_representation("sst2_sc_rate0")
    assert isinstance(reps, ModelRepresentations)


@pytest.mark.gpu
@pytest.mark.slow
def test_nlp_model_output_extraction():
    base_model = [m for m in all_trained_nlp_models() if m.identifier == "Normal"][0]
    out = base_model.get_output("sst2")
    assert isinstance(out, Prediction)
    assert isinstance(out.output, torch.Tensor)
