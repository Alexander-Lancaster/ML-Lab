"""PyTorch model generation."""

from __future__ import annotations

import inspect
from typing import Any

from .config import ArchitectureConfig, ConfigurationError


def _layer_registry(nn: Any) -> dict[str, type]:
    # Explicit allow-list: YAML cannot instantiate arbitrary Python objects.
    names = (
        "AdaptiveAvgPool1d", "AdaptiveAvgPool2d", "AdaptiveAvgPool3d",
        "AvgPool1d", "AvgPool2d", "AvgPool3d", "BatchNorm1d", "BatchNorm2d",
        "BatchNorm3d", "Conv1d", "Conv2d", "Conv3d", "Dropout", "Dropout2d",
        "Dropout3d", "ELU", "Flatten", "GELU", "Identity", "LayerNorm",
        "LazyLinear", "LeakyReLU", "Linear", "LogSoftmax", "MaxPool1d",
        "MaxPool2d", "MaxPool3d", "ReLU", "Sigmoid", "Softmax", "Tanh",
    )
    return {name: getattr(nn, name) for name in names}


def build_model(config: ArchitectureConfig):
    """Convert an architecture definition into a ``torch.nn.Sequential`` model."""
    try:
        from torch import nn
    except ImportError as exc:  # pragma: no cover - depends on installation state
        raise RuntimeError("PyTorch is required; install the project with `pip install -e .`") from exc

    registry = _layer_registry(nn)
    modules = []
    for index, layer in enumerate(config.layers):
        constructor = registry.get(layer.type)
        if constructor is None:
            supported = ", ".join(sorted(registry))
            raise ConfigurationError(
                f"layers[{index}].type: unsupported layer {layer.type!r}; supported: {supported}"
            )
        try:
            modules.append(constructor(**layer.params))
        except (TypeError, ValueError) as exc:
            try:
                signature = inspect.signature(constructor)
                hint = f"; expected {layer.type}{signature}"
            except (TypeError, ValueError):
                hint = ""
            raise ConfigurationError(f"layers[{index}].params: {exc}{hint}") from exc

    model = nn.Sequential(*modules)
    model.architecture_name = config.name
    model.input_shape = config.input_shape
    return model


def validate_forward_pass(model: Any, input_shape: tuple[int, ...]) -> tuple[int, ...]:
    """Run a batch of one through the model and return its output shape."""
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("PyTorch is required; install the project with `pip install -e .`") from exc
    try:
        with torch.no_grad():
            output = model(torch.zeros((1, *input_shape)))
    except Exception as exc:
        raise ConfigurationError(f"model failed its validation forward pass: {exc}") from exc
    return tuple(output.shape)
