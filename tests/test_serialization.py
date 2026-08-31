from ml_lab import (
    EditableGraph,
    EditableLayer,
    config_to_dict,
    dump_config,
    load_config,
    save_config,
)
from ml_lab.config import ConfigurationError, parse_config


def legacy_data():
    return {
        "model": {"name": "classifier"},
        "input": {"shape": [4]},
        "layers": [
            {"type": "Linear", "params": {"in_features": 4, "out_features": 2}},
            {"type": "ReLU"},
        ],
    }


def test_legacy_config_serializes_to_canonical_versioned_graph():
    config = parse_config(legacy_data())
    serialized = config_to_dict(config)

    assert serialized["format_version"] == 1
    assert "input" not in serialized
    assert serialized["inputs"] == {"input": {"shape": [4]}}
    assert serialized["layers"][0]["id"] == "layer_0"
    assert serialized["layers"][1]["inputs"] == ["layer_0"]
    assert serialized["outputs"] == ["layer_1"]


def test_yaml_round_trip_preserves_config(tmp_path):
    original = parse_config(legacy_data())
    destination = tmp_path / "architecture.yaml"

    save_config(original, destination)
    reloaded = load_config(destination)

    assert reloaded == original
    assert destination.read_text(encoding="utf-8").startswith("format_version: 1\n")


def test_dump_config_is_stable():
    config = parse_config(legacy_data())
    assert dump_config(config) == dump_config(parse_config(config_to_dict(config)))


def test_rejects_unsupported_format_version():
    data = legacy_data()
    data["format_version"] = 2
    try:
        parse_config(data)
    except ConfigurationError as exc:
        assert exc.issue.code == "unsupported_format_version"
        assert exc.issue.path == ("format_version",)
    else:
        raise AssertionError("expected unsupported format version")


def test_editable_graph_can_be_temporarily_invalid():
    editable = EditableGraph.from_config(parse_config(legacy_data()))
    editable.layers.pop(0)

    issues = editable.validate()

    assert len(issues) == 1
    assert issues[0].code == "unknown_reference"
    assert issues[0].node_id == "layer_1"
    assert issues[0].path == ("layers", 0, "inputs", 0)


def test_editable_graph_reports_shape_issue():
    editable = EditableGraph(
        name="bad_shapes",
        inputs={"x": [3]},
        layers=[
            EditableLayer(
                id="output",
                type="Linear",
                inputs=["x"],
                params={"in_features": 4, "out_features": 2},
            )
        ],
        outputs=["output"],
    )

    issues = editable.validate(check_shapes=True)

    assert len(issues) == 1
    assert issues[0].code == "incompatible_shapes"
    assert issues[0].node_id == "output"
