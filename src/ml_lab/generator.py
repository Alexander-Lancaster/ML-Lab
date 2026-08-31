"""PyTorch graph-model generation and tensor-shape validation."""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any

from .config import ArchitectureConfig, ConfigurationError

try:  # Keep YAML-only use capable of producing a helpful dependency error.
    import torch
    from torch import nn
except ImportError:  # pragma: no cover - depends on installation state
    torch = None
    nn = None


@dataclass(frozen=True)
class LayerValidation:
    """The tensor shapes observed at one graph node during validation."""

    index: int
    id: str
    type: str
    input_shapes: tuple[tuple[int, ...], ...]
    output_shape: tuple[int, ...]

    @property
    def input_shape(self) -> tuple[int, ...]:
        """Compatibility shortcut for a node with exactly one input."""
        if len(self.input_shapes) != 1:
            raise ConfigurationError("input_shape is only available for single-input nodes")
        return self.input_shapes[0]


@dataclass(frozen=True)
class ArchitectureValidation:
    """Successful shape trace for an entire architecture graph."""

    input_shapes: dict[str, tuple[int, ...]]
    output_shapes: dict[str, tuple[int, ...]]
    layers: tuple[LayerValidation, ...]

    @property
    def input_shape(self) -> tuple[int, ...]:
        if len(self.input_shapes) != 1:
            raise ConfigurationError("input_shape is only available for single-input architectures")
        return next(iter(self.input_shapes.values()))

    @property
    def output_shape(self) -> tuple[int, ...]:
        if len(self.output_shapes) != 1:
            raise ConfigurationError("output_shape is only available for single-output architectures")
        return next(iter(self.output_shapes.values()))


if nn is not None:

    class Add(nn.Module):
        """Add two or more tensors with identical shapes."""

        def forward(self, *values):
            if len(values) < 2:
                raise ValueError("Add requires at least two inputs")
            shapes = [tuple(value.shape) for value in values]
            if any(shape != shapes[0] for shape in shapes[1:]):
                raise ValueError(f"Add requires identical input shapes, received {shapes}")
            result = values[0]
            for value in values[1:]:
                result = result + value
            return result


    class Concatenate(nn.Module):
        """Concatenate two or more tensors along a configured dimension."""

        def __init__(self, dim: int = 1):
            super().__init__()
            self.dim = dim

        def forward(self, *values):
            if len(values) < 2:
                raise ValueError("Concatenate requires at least two inputs")
            ranks = [value.dim() for value in values]
            if any(rank != ranks[0] for rank in ranks[1:]):
                raise ValueError(f"Concatenate requires equal tensor ranks, received {ranks}")
            dim = self.dim if self.dim >= 0 else ranks[0] + self.dim
            if dim < 0 or dim >= ranks[0]:
                raise ValueError(f"Concatenate dimension {self.dim} is out of range for rank {ranks[0]}")
            shapes = [tuple(value.shape) for value in values]
            for axis in range(ranks[0]):
                if axis != dim and len({shape[axis] for shape in shapes}) != 1:
                    raise ValueError(
                        f"Concatenate input shapes must match outside dimension {self.dim}, "
                        f"received {shapes}"
                    )
            return torch.cat(values, dim=self.dim)


    class GraphModel(nn.Module):
        """Execute modules according to named graph dependencies."""

        def __init__(self, config: ArchitectureConfig, modules: dict[str, nn.Module]):
            super().__init__()
            self.architecture = config
            self.architecture_name = config.name
            self.input_names = tuple(config.inputs)
            self.output_names = config.outputs
            self.nodes = nn.ModuleDict(modules)

        def _bind_inputs(self, args, kwargs):
            if args and kwargs:
                raise ValueError("provide model inputs either positionally or by name, not both")
            if kwargs:
                missing = set(self.input_names) - set(kwargs)
                unknown = set(kwargs) - set(self.input_names)
                if missing or unknown:
                    details = []
                    if missing:
                        details.append("missing: " + ", ".join(sorted(missing)))
                    if unknown:
                        details.append("unknown: " + ", ".join(sorted(unknown)))
                    raise ValueError("invalid model inputs (" + "; ".join(details) + ")")
                return dict(kwargs)
            if len(args) != len(self.input_names):
                raise ValueError(
                    f"expected {len(self.input_names)} model input(s), received {len(args)}"
                )
            return dict(zip(self.input_names, args))

        def forward(self, *args, **kwargs):
            values = self._bind_inputs(args, kwargs)
            for layer in self.architecture.layers:
                node_inputs = [values[reference] for reference in layer.inputs]
                values[layer.id] = self.nodes[layer.id](*node_inputs)
            outputs = tuple(values[name] for name in self.output_names)
            return outputs[0] if len(outputs) == 1 else outputs

else:  # pragma: no cover - only used to make the missing dependency explicit

    class GraphModel:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("PyTorch is required; install the project with `pip install -e .`")


def _require_torch() -> None:
    if torch is None or nn is None:
        raise RuntimeError("PyTorch is required; install the project with `pip install -e .`")


def _layer_registry() -> dict[str, type]:
    _require_torch()
    # Explicit allow-list: YAML cannot instantiate arbitrary Python objects.
    names = (
        "AdaptiveAvgPool1d", "AdaptiveAvgPool2d", "AdaptiveAvgPool3d",
        "AvgPool1d", "AvgPool2d", "AvgPool3d", "BatchNorm1d", "BatchNorm2d",
        "BatchNorm3d", "Conv1d", "Conv2d", "Conv3d", "Dropout", "Dropout2d",
        "Dropout3d", "ELU", "Flatten", "GELU", "Identity", "LayerNorm",
        "LazyLinear", "LeakyReLU", "Linear", "LogSoftmax", "MaxPool1d",
        "MaxPool2d", "MaxPool3d", "ReLU", "Sigmoid", "Softmax", "Tanh",
    )
    registry = {name: getattr(nn, name) for name in names}
    registry.update({"Add": Add, "Concatenate": Concatenate})
    return registry


def build_model(config: ArchitectureConfig) -> GraphModel:
    """Convert an architecture graph into an executable PyTorch model."""
    registry = _layer_registry()
    modules: dict[str, nn.Module] = {}
    for index, layer in enumerate(config.layers):
        constructor = registry.get(layer.type)
        if constructor is None:
            supported = ", ".join(sorted(registry))
            raise ConfigurationError(
                f"layers[{index}].type: unsupported layer {layer.type!r}; supported: {supported}",
                code="unsupported_layer",
                path=("layers", index, "type"),
                node_id=layer.id,
            )
        try:
            modules[layer.id] = constructor(**layer.params)
        except (TypeError, ValueError) as exc:
            try:
                signature = inspect.signature(constructor)
                hint = f"; expected {layer.type}{signature}"
            except (TypeError, ValueError):
                hint = ""
            raise ConfigurationError(
                f"layers[{index}].params: {exc}{hint}",
                code="invalid_layer_params",
                path=("layers", index, "params"),
                node_id=layer.id,
            ) from exc
    return GraphModel(config, modules)


def validate_forward_pass(model: Any, input_shape: tuple[int, ...]) -> tuple[int, ...]:
    """Backward-compatible validation helper for single-input/output models."""
    _require_torch()
    try:
        with torch.no_grad():
            output = model(torch.zeros((1, *input_shape)))
    except Exception as exc:
        raise ConfigurationError(f"model failed its validation forward pass: {exc}") from exc
    if not isinstance(output, torch.Tensor):
        raise ConfigurationError("model has multiple outputs; use validate_architecture instead")
    return tuple(output.shape)


def validate_architecture(config: ArchitectureConfig) -> ArchitectureValidation:
    """Build and shape-check every node in an architecture graph."""
    _require_torch()
    model = build_model(config)
    model.eval()
    values = {
        name: torch.zeros((1, *shape))
        for name, shape in config.inputs.items()
    }
    input_shapes = {name: tuple(value.shape) for name, value in values.items()}
    trace: list[LayerValidation] = []

    with torch.no_grad():
        for index, layer in enumerate(config.layers):
            node_inputs = [values[reference] for reference in layer.inputs]
            node_input_shapes = tuple(tuple(value.shape) for value in node_inputs)
            try:
                output = model.nodes[layer.id](*node_inputs)
            except Exception as exc:
                raise ConfigurationError(
                    f"layers[{index}] {layer.id!r} ({layer.type}) cannot process input "
                    f"shape(s) {node_input_shapes}: {exc}",
                    code="incompatible_shapes",
                    path=("layers", index, "inputs"),
                    node_id=layer.id,
                ) from exc
            if not isinstance(output, torch.Tensor):
                raise ConfigurationError(
                    f"layers[{index}] {layer.id!r} ({layer.type}) returned "
                    f"{type(output).__name__}, expected a tensor",
                    code="invalid_layer_output",
                    path=("layers", index),
                    node_id=layer.id,
                )
            output_shape = tuple(output.shape)
            values[layer.id] = output
            trace.append(
                LayerValidation(
                    index=index,
                    id=layer.id,
                    type=layer.type,
                    input_shapes=node_input_shapes,
                    output_shape=output_shape,
                )
            )

    return ArchitectureValidation(
        input_shapes=input_shapes,
        output_shapes={name: tuple(values[name].shape) for name in config.outputs},
        layers=tuple(trace),
    )
