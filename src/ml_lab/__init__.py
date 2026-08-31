"""Build PyTorch architectures from declarative YAML files."""

from .config import (
    CURRENT_FORMAT_VERSION,
    ArchitectureConfig,
    ConfigurationError,
    LayerConfig,
    ValidationIssue,
    load_config,
)
from .editable import EditableGraph, EditableLayer
from .serialization import config_to_dict, dump_config, save_config

__all__ = [
    "ArchitectureConfig",
    "ArchitectureValidation",
    "CURRENT_FORMAT_VERSION",
    "ConfigurationError",
    "EditableGraph",
    "EditableLayer",
    "GraphModel",
    "LayerConfig",
    "LayerValidation",
    "ValidationIssue",
    "build_model",
    "config_to_dict",
    "dump_config",
    "load_config",
    "save_config",
    "validate_architecture",
]


def build_model(config: ArchitectureConfig):
    """Build a model lazily so reading configs does not require PyTorch."""
    from .generator import build_model as _build_model

    return _build_model(config)


def validate_architecture(config: ArchitectureConfig):
    """Validate construction and tensor compatibility for every layer."""
    from .generator import validate_architecture as _validate_architecture

    return _validate_architecture(config)


from .generator import ArchitectureValidation, GraphModel, LayerValidation
