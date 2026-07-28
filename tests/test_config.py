import pytest

from ml_lab.config import ConfigurationError, parse_config


def valid_data():
    return {
        "model": {"name": "classifier"},
        "input": {"shape": [4]},
        "layers": [{"type": "Linear", "params": {"in_features": 4, "out_features": 2}}],
    }


def test_parse_config():
    config = parse_config(valid_data())
    assert config.name == "classifier"
    assert config.input_shape == (4,)
    assert config.layers[0].type == "Linear"


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda data: data["input"].update(shape=[0, 2]), "positive integers"),
        (lambda data: data.update(extra=True), "unknown field"),
        (lambda data: data.update(layers=[]), "non-empty list"),
    ],
)
def test_invalid_config(change, message):
    data = valid_data()
    change(data)
    with pytest.raises(ConfigurationError, match=message):
        parse_config(data)
