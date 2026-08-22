"""YAML loading, normalization, and graph validation."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
import re
from typing import Any


class ConfigurationError(ValueError):
    """Raised when an architecture file is malformed."""


_ID_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_-]*$")


@dataclass(frozen=True)
class LayerConfig:
    """One node in an architecture graph.

    The name is retained for API compatibility; graph operations such as Add
    and Concatenate are represented by the same object as PyTorch layers.
    """

    type: str
    params: dict[str, Any] = field(default_factory=dict)
    id: str = ""
    inputs: tuple[str, ...] = ()


@dataclass(frozen=True)
class ArchitectureConfig:
    name: str
    inputs: dict[str, tuple[int, ...]]
    layers: tuple[LayerConfig, ...]
    outputs: tuple[str, ...]

    @property
    def input_shape(self) -> tuple[int, ...]:
        """Return the shape for a single-input architecture."""
        if len(self.inputs) != 1:
            raise ConfigurationError("input_shape is only available for single-input architectures")
        return next(iter(self.inputs.values()))


def _require_keys(value: dict[str, Any], allowed: set[str], location: str) -> None:
    unknown = set(value) - allowed
    if unknown:
        names = ", ".join(sorted(unknown))
        raise ConfigurationError(f"{location}: unknown field(s): {names}")


def _validate_id(value: Any, location: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigurationError(f"{location} must be a non-empty string")
    if not _ID_PATTERN.fullmatch(value):
        raise ConfigurationError(
            f"{location} must start with a letter or underscore and contain only "
            "letters, numbers, underscores, or hyphens"
        )
    return value


def _parse_shape(value: Any, location: str) -> tuple[int, ...]:
    if (
        not isinstance(value, list)
        or not value
        or any(
            not isinstance(size, int) or isinstance(size, bool) or size <= 0
            for size in value
        )
    ):
        raise ConfigurationError(f"{location} must be a non-empty list of positive integers")
    return tuple(value)


def _parse_inputs(data: dict[str, Any]) -> tuple[dict[str, tuple[int, ...]], bool]:
    has_legacy_input = "input" in data
    has_graph_inputs = "inputs" in data
    if has_legacy_input == has_graph_inputs:
        raise ConfigurationError("document must contain exactly one of input or inputs")

    if has_legacy_input:
        input_config = data["input"]
        if not isinstance(input_config, dict):
            raise ConfigurationError("input must be a mapping")
        _require_keys(input_config, {"shape"}, "input")
        return {"input": _parse_shape(input_config.get("shape"), "input.shape")}, True

    inputs_data = data["inputs"]
    if not isinstance(inputs_data, dict) or not inputs_data:
        raise ConfigurationError("inputs must be a non-empty mapping")
    parsed: dict[str, tuple[int, ...]] = {}
    for raw_name, input_config in inputs_data.items():
        name = _validate_id(raw_name, "input name")
        location = f"inputs.{name}"
        if not isinstance(input_config, dict):
            raise ConfigurationError(f"{location} must be a mapping")
        _require_keys(input_config, {"shape"}, location)
        parsed[name] = _parse_shape(input_config.get("shape"), f"{location}.shape")
    return parsed, False


def _topological_sort(
    layers: list[LayerConfig], input_names: set[str]
) -> tuple[LayerConfig, ...]:
    """Validate references and return nodes in dependency order."""
    by_id: dict[str, LayerConfig] = {}
    original_index: dict[str, int] = {}
    for index, layer in enumerate(layers):
        if layer.id in input_names:
            raise ConfigurationError(
                f"layers[{index}].id: {layer.id!r} conflicts with a model input"
            )
        if layer.id in by_id:
            raise ConfigurationError(f"layers[{index}].id: duplicate id {layer.id!r}")
        by_id[layer.id] = layer
        original_index[layer.id] = index

    known = input_names | set(by_id)
    for index, layer in enumerate(layers):
        for input_index, reference in enumerate(layer.inputs):
            if reference not in known:
                raise ConfigurationError(
                    f"layers[{index}].inputs[{input_index}]: unknown reference {reference!r}"
                )

    indegree = {layer.id: 0 for layer in layers}
    dependants: dict[str, list[str]] = {layer.id: [] for layer in layers}
    for layer in layers:
        for reference in layer.inputs:
            if reference in by_id:
                indegree[layer.id] += 1
                dependants[reference].append(layer.id)

    ready = deque(layer.id for layer in layers if indegree[layer.id] == 0)
    ordered: list[LayerConfig] = []
    while ready:
        node_id = ready.popleft()
        ordered.append(by_id[node_id])
        for dependant in dependants[node_id]:
            indegree[dependant] -= 1
            if indegree[dependant] == 0:
                ready.append(dependant)

    if len(ordered) != len(layers):
        cycle_ids = [node_id for node_id, degree in indegree.items() if degree > 0]
        cycle_ids.sort(key=original_index.__getitem__)
        raise ConfigurationError(f"architecture graph contains a cycle involving: {', '.join(cycle_ids)}")
    return tuple(ordered)


def _validate_outputs_and_usage(
    layers: tuple[LayerConfig, ...], input_names: set[str], outputs: tuple[str, ...]
) -> None:
    by_id = {layer.id: layer for layer in layers}
    known = input_names | set(by_id)
    for index, output in enumerate(outputs):
        if output not in known:
            raise ConfigurationError(f"outputs[{index}]: unknown reference {output!r}")

    required = set(outputs)
    pending = list(outputs)
    while pending:
        reference = pending.pop()
        layer = by_id.get(reference)
        if layer is None:
            continue
        for dependency in layer.inputs:
            if dependency not in required:
                required.add(dependency)
                pending.append(dependency)

    unused = [layer.id for layer in layers if layer.id not in required]
    if unused:
        raise ConfigurationError(
            "architecture contains node(s) that do not contribute to an output: "
            + ", ".join(unused)
        )


def parse_config(data: Any) -> ArchitectureConfig:
    """Validate decoded YAML and normalize it into an architecture graph."""
    if not isinstance(data, dict):
        raise ConfigurationError("document must be a mapping")
    _require_keys(data, {"model", "input", "inputs", "layers", "outputs"}, "document")

    model = data.get("model")
    if not isinstance(model, dict):
        raise ConfigurationError("model must be a mapping")
    _require_keys(model, {"name"}, "model")
    name = model.get("name")
    if not isinstance(name, str) or not name.strip():
        raise ConfigurationError("model.name must be a non-empty string")

    inputs, legacy_format = _parse_inputs(data)
    layers_data = data.get("layers")
    if not isinstance(layers_data, list) or not layers_data:
        raise ConfigurationError("layers must be a non-empty list")

    layers: list[LayerConfig] = []
    previous_id: str | None = None
    for index, layer in enumerate(layers_data):
        location = f"layers[{index}]"
        if not isinstance(layer, dict):
            raise ConfigurationError(f"{location} must be a mapping")
        _require_keys(layer, {"id", "type", "inputs", "params"}, location)

        layer_type = layer.get("type")
        if not isinstance(layer_type, str) or not layer_type.strip():
            raise ConfigurationError(f"{location}.type must be a non-empty string")

        if "id" in layer:
            node_id = _validate_id(layer["id"], f"{location}.id")
        elif legacy_format:
            node_id = f"layer_{index}"
        else:
            raise ConfigurationError(f"{location}.id is required when using named inputs")

        if "inputs" in layer:
            references = layer["inputs"]
            if not isinstance(references, list) or not references:
                raise ConfigurationError(f"{location}.inputs must be a non-empty list")
            node_inputs = tuple(
                _validate_id(reference, f"{location}.inputs[{input_index}]")
                for input_index, reference in enumerate(references)
            )
        elif legacy_format:
            node_inputs = (previous_id or "input",)
        else:
            raise ConfigurationError(f"{location}.inputs is required when using named inputs")

        params = layer.get("params", {})
        if not isinstance(params, dict):
            raise ConfigurationError(f"{location}.params must be a mapping")
        layers.append(LayerConfig(type=layer_type, params=params, id=node_id, inputs=node_inputs))
        previous_id = node_id

    raw_outputs = data.get("outputs")
    if raw_outputs is None:
        outputs = (layers[-1].id,)
    else:
        if not isinstance(raw_outputs, list) or not raw_outputs:
            raise ConfigurationError("outputs must be a non-empty list")
        outputs = tuple(
            _validate_id(output, f"outputs[{index}]")
            for index, output in enumerate(raw_outputs)
        )
        if len(set(outputs)) != len(outputs):
            raise ConfigurationError("outputs must not contain duplicate references")

    ordered_layers = _topological_sort(layers, set(inputs))
    _validate_outputs_and_usage(ordered_layers, set(inputs), outputs)
    return ArchitectureConfig(
        name=name,
        inputs=inputs,
        layers=ordered_layers,
        outputs=outputs,
    )


def load_config(path: str | Path) -> ArchitectureConfig:
    """Load and validate an architecture YAML file."""
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - depends on installation state
        raise RuntimeError("PyYAML is required; install the project with `pip install -e .`") from exc

    config_path = Path(path)
    try:
        with config_path.open("r", encoding="utf-8") as stream:
            data = yaml.safe_load(stream)
    except OSError as exc:
        raise ConfigurationError(f"could not read {config_path}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise ConfigurationError(f"invalid YAML in {config_path}: {exc}") from exc
    return parse_config(data)
