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
    assert config.layers[0].id == "layer_0"
    assert config.layers[0].inputs == ("input",)
    assert config.outputs == ("layer_0",)


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


def graph_data():
    return {
        "model": {"name": "graph"},
        "inputs": {"x": {"shape": [4]}},
        "layers": [
            {
                "id": "output",
                "type": "Linear",
                "inputs": ["hidden"],
                "params": {"in_features": 3, "out_features": 2},
            },
            {
                "id": "hidden",
                "type": "Linear",
                "inputs": ["x"],
                "params": {"in_features": 4, "out_features": 3},
            },
        ],
        "outputs": ["output"],
    }


def test_graph_nodes_are_topologically_sorted():
    config = parse_config(graph_data())
    assert list(config.inputs) == ["x"]
    assert [layer.id for layer in config.layers] == ["hidden", "output"]


def test_graph_rejects_unknown_reference():
    data = graph_data()
    data["layers"][1]["inputs"] = ["missing"]
    with pytest.raises(ConfigurationError, match="unknown reference 'missing'"):
        parse_config(data)


def test_graph_rejects_duplicate_ids():
    data = graph_data()
    data["layers"][1]["id"] = "output"
    with pytest.raises(ConfigurationError, match="duplicate id 'output'"):
        parse_config(data)


def test_graph_rejects_cycles():
    data = graph_data()
    data["layers"][0]["inputs"] = ["hidden"]
    data["layers"][1]["inputs"] = ["output"]
    with pytest.raises(ConfigurationError, match="contains a cycle"):
        parse_config(data)


def test_graph_rejects_nodes_unused_by_outputs():
    data = graph_data()
    data["layers"].append({"id": "unused", "type": "ReLU", "inputs": ["x"]})
    with pytest.raises(ConfigurationError, match="do not contribute.*unused"):
        parse_config(data)
