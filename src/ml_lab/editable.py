"""Mutable graph state for an interactive editor."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

from .config import (
    CURRENT_FORMAT_VERSION,
    ArchitectureConfig,
    ConfigurationError,
    ValidationIssue,
    parse_config,
)


@dataclass
class EditableLayer:
    """A mutable node that may temporarily contain invalid values."""

    id: str
    type: str
    inputs: list[str] = field(default_factory=list)
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class EditableGraph:
    """Mutable GUI-facing graph state, including temporarily invalid graphs."""

    name: str
    inputs: dict[str, list[int]] = field(default_factory=dict)
    layers: list[EditableLayer] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)
    format_version: int = CURRENT_FORMAT_VERSION

    @classmethod
    def from_config(cls, config: ArchitectureConfig) -> EditableGraph:
        return cls(
            name=config.name,
            inputs={name: list(shape) for name, shape in config.inputs.items()},
            layers=[
                EditableLayer(
                    id=layer.id,
                    type=layer.type,
                    inputs=list(layer.inputs),
                    params=deepcopy(layer.params),
                )
                for layer in config.layers
            ],
            outputs=list(config.outputs),
            format_version=config.format_version,
        )

    def to_dict(self) -> dict[str, Any]:
        layers = []
        for layer in self.layers:
            serialized: dict[str, Any] = {
                "id": layer.id,
                "type": layer.type,
                "inputs": list(layer.inputs),
            }
            if layer.params:
                serialized["params"] = deepcopy(layer.params)
            layers.append(serialized)
        return {
            "format_version": self.format_version,
            "model": {"name": self.name},
            "inputs": {
                name: {"shape": list(shape)}
                for name, shape in self.inputs.items()
            },
            "layers": layers,
            "outputs": list(self.outputs),
        }

    def to_config(self) -> ArchitectureConfig:
        """Validate the current state and return its immutable representation."""
        return parse_config(self.to_dict())

    def validate(self, *, check_shapes: bool = False) -> tuple[ValidationIssue, ...]:
        """Return structured issues without preventing further graph editing."""
        try:
            config = self.to_config()
            if check_shapes:
                from .generator import validate_architecture

                validate_architecture(config)
        except ConfigurationError as exc:
            return (exc.issue,)
        return ()
