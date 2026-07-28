import pytest

torch = pytest.importorskip("torch")

from ml_lab.config import ConfigurationError, parse_config
from ml_lab.generator import build_model, validate_forward_pass


def test_build_and_validate_model():
    config = parse_config(
        {
            "model": {"name": "tiny"},
            "input": {"shape": [4]},
            "layers": [
                {"type": "Linear", "params": {"in_features": 4, "out_features": 3}},
                {"type": "ReLU"},
                {"type": "Linear", "params": {"in_features": 3, "out_features": 2}},
            ],
        }
    )
    model = build_model(config)
    assert validate_forward_pass(model, config.input_shape) == (1, 2)


def test_rejects_unlisted_layer():
    config = parse_config(
        {"model": {"name": "bad"}, "input": {"shape": [1]}, "layers": [{"type": "Exec"}]}
    )
    with pytest.raises(ConfigurationError, match="unsupported layer"):
        build_model(config)
