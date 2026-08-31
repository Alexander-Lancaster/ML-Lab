"""Canonical YAML serialization for validated architectures."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .config import ArchitectureConfig


def _yaml_value(value: Any) -> Any:
    """Convert nested values to objects emitted safely and portably by YAML."""
    if isinstance(value, dict):
        return {key: _yaml_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_yaml_value(item) for item in value]
    return value


def config_to_dict(config: ArchitectureConfig) -> dict[str, Any]:
    """Return the canonical, versioned mapping for an architecture."""
    layers = []
    for layer in config.layers:
        serialized: dict[str, Any] = {
            "id": layer.id,
            "type": layer.type,
            "inputs": list(layer.inputs),
        }
        if layer.params:
            serialized["params"] = _yaml_value(layer.params)
        layers.append(serialized)

    return {
        "format_version": config.format_version,
        "model": {"name": config.name},
        "inputs": {
            name: {"shape": list(shape)}
            for name, shape in config.inputs.items()
        },
        "layers": layers,
        "outputs": list(config.outputs),
    }


def dump_config(config: ArchitectureConfig) -> str:
    """Serialize an architecture as canonical YAML text."""
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - depends on installation state
        raise RuntimeError("PyYAML is required; install the project with `pip install -e .`") from exc
    return yaml.safe_dump(
        config_to_dict(config),
        sort_keys=False,
        default_flow_style=False,
        allow_unicode=True,
    )


def save_config(config: ArchitectureConfig, path: str | Path) -> None:
    """Write an architecture to a UTF-8 YAML file."""
    config_path = Path(path)
    try:
        config_path.write_text(dump_config(config), encoding="utf-8")
    except OSError as exc:
        raise OSError(f"could not write {config_path}: {exc}") from exc
