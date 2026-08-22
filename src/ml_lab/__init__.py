"""Build PyTorch architectures from declarative YAML files."""

from .config import ArchitectureConfig, ConfigurationError, LayerConfig, load_config

__all__ = [
    "ArchitectureConfig",
    "ArchitectureValidation",
    "ConfigurationError",
    "GraphModel",
    "LayerConfig",
    "LayerValidation",
    "build_model",
    "load_config",
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
