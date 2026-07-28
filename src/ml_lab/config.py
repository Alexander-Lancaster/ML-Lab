"""YAML loading and validation for architecture definitions."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


class ConfigurationError(ValueError):
    """Raised when an architecture file is malformed."""


@dataclass(frozen=True)
class LayerConfig:
    type: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ArchitectureConfig:
    name: str
    input_shape: tuple[int, ...]
    layers: tuple[LayerConfig, ...]


def _require_keys(value: dict[str, Any], allowed: set[str], location: str) -> None:
    unknown = set(value) - allowed
    if unknown:
        names = ", ".join(sorted(unknown))
        raise ConfigurationError(f"{location}: unknown field(s): {names}")


def parse_config(data: Any) -> ArchitectureConfig:
    """Validate decoded YAML and return an immutable architecture definition."""
    if not isinstance(data, dict):
        raise ConfigurationError("document must be a mapping")
    _require_keys(data, {"model", "input", "layers"}, "document")

    model = data.get("model")
    if not isinstance(model, dict):
        raise ConfigurationError("model must be a mapping")
    _require_keys(model, {"name"}, "model")
    name = model.get("name")
    if not isinstance(name, str) or not name.strip():
        raise ConfigurationError("model.name must be a non-empty string")

    input_config = data.get("input")
    if not isinstance(input_config, dict):
        raise ConfigurationError("input must be a mapping")
    _require_keys(input_config, {"shape"}, "input")
    shape = input_config.get("shape")
    if (
        not isinstance(shape, list)
        or not shape
        or any(not isinstance(size, int) or isinstance(size, bool) or size <= 0 for size in shape)
    ):
        raise ConfigurationError("input.shape must be a non-empty list of positive integers")

    layers_data = data.get("layers")
    if not isinstance(layers_data, list) or not layers_data:
        raise ConfigurationError("layers must be a non-empty list")

    layers: list[LayerConfig] = []
    for index, layer in enumerate(layers_data):
        location = f"layers[{index}]"
        if not isinstance(layer, dict):
            raise ConfigurationError(f"{location} must be a mapping")
        _require_keys(layer, {"type", "params"}, location)
        layer_type = layer.get("type")
        if not isinstance(layer_type, str) or not layer_type.strip():
            raise ConfigurationError(f"{location}.type must be a non-empty string")
        params = layer.get("params", {})
        if not isinstance(params, dict):
            raise ConfigurationError(f"{location}.params must be a mapping")
        layers.append(LayerConfig(type=layer_type, params=params))

    return ArchitectureConfig(name=name, input_shape=tuple(shape), layers=tuple(layers))


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
