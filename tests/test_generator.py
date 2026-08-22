import pytest

torch = pytest.importorskip("torch")

from ml_lab.config import ConfigurationError, parse_config
from ml_lab.generator import build_model, validate_architecture, validate_forward_pass


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


def test_architecture_validation_returns_per_layer_shape_trace():
    config = parse_config(
        {
            "model": {"name": "traceable"},
            "input": {"shape": [4]},
            "layers": [
                {"type": "Linear", "params": {"in_features": 4, "out_features": 3}},
                {"type": "ReLU"},
                {"type": "Linear", "params": {"in_features": 3, "out_features": 2}},
            ],
        }
    )

    result = validate_architecture(config)

    assert result.input_shape == (1, 4)
    assert result.output_shape == (1, 2)
    assert [(layer.input_shape, layer.output_shape) for layer in result.layers] == [
        ((1, 4), (1, 3)),
        ((1, 3), (1, 3)),
        ((1, 3), (1, 2)),
    ]


def test_architecture_validation_identifies_incompatible_layer():
    config = parse_config(
        {
            "model": {"name": "broken"},
            "input": {"shape": [4]},
            "layers": [
                {"type": "Linear", "params": {"in_features": 4, "out_features": 3}},
                {"type": "Linear", "params": {"in_features": 5, "out_features": 2}},
            ],
        }
    )

    with pytest.raises(
        ConfigurationError,
        match=r"layers\[1\] 'layer_1' \(Linear\).*\(1, 3\)",
    ):
        validate_architecture(config)


def residual_config():
    return parse_config(
        {
            "model": {"name": "residual"},
            "inputs": {"image": {"shape": [3, 8, 8]}},
            "layers": [
                {
                    "id": "stem",
                    "type": "Conv2d",
                    "inputs": ["image"],
                    "params": {
                        "in_channels": 3,
                        "out_channels": 4,
                        "kernel_size": 1,
                    },
                },
                {"id": "branch", "type": "ReLU", "inputs": ["stem"]},
                {"id": "residual", "type": "Add", "inputs": ["stem", "branch"]},
            ],
            "outputs": ["residual"],
        }
    )


def test_graph_model_executes_residual_branch():
    config = residual_config()
    model = build_model(config)
    output = model(torch.ones(2, 3, 8, 8))
    assert output.shape == (2, 4, 8, 8)

    validation = validate_architecture(config)
    add_trace = validation.layers[-1]
    assert add_trace.id == "residual"
    assert add_trace.input_shapes == ((1, 4, 8, 8), (1, 4, 8, 8))
    assert validation.output_shapes == {"residual": (1, 4, 8, 8)}


def test_concatenate_combines_branches():
    config = parse_config(
        {
            "model": {"name": "concat"},
            "inputs": {"x": {"shape": [4]}},
            "layers": [
                {"id": "left", "type": "Identity", "inputs": ["x"]},
                {"id": "right", "type": "Identity", "inputs": ["x"]},
                {
                    "id": "combined",
                    "type": "Concatenate",
                    "inputs": ["left", "right"],
                    "params": {"dim": 1},
                },
            ],
            "outputs": ["combined"],
        }
    )
    result = validate_architecture(config)
    assert result.output_shape == (1, 8)


def test_add_rejects_different_shapes():
    config = parse_config(
        {
            "model": {"name": "bad_add"},
            "inputs": {"x": {"shape": [4]}, "y": {"shape": [3]}},
            "layers": [{"id": "sum", "type": "Add", "inputs": ["x", "y"]}],
            "outputs": ["sum"],
        }
    )
    with pytest.raises(ConfigurationError, match="Add requires identical input shapes"):
        validate_architecture(config)


def test_multiple_inputs_and_outputs():
    config = parse_config(
        {
            "model": {"name": "multi"},
            "inputs": {"left": {"shape": [2]}, "right": {"shape": [2]}},
            "layers": [
                {"id": "left_relu", "type": "ReLU", "inputs": ["left"]},
                {"id": "right_relu", "type": "ReLU", "inputs": ["right"]},
            ],
            "outputs": ["left_relu", "right_relu"],
        }
    )
    model = build_model(config)
    outputs = model(left=torch.ones(1, 2), right=torch.ones(1, 2))
    assert len(outputs) == 2
    assert all(output.shape == (1, 2) for output in outputs)
