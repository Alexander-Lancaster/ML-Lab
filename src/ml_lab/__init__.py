"""Build PyTorch architectures from declarative YAML files."""

from .config import ArchitectureConfig, ConfigurationError, LayerConfig, load_config

__all__ = ["ArchitectureConfig", "ConfigurationError", "LayerConfig", "load_config"]


def build_model(config: ArchitectureConfig):
    """Build a model lazily so reading configs does not require PyTorch."""
    from .generator import build_model as _build_model

    return _build_model(config)
