"""Command-line interface for model generation."""

from __future__ import annotations

import argparse

from .config import ConfigurationError, load_config
from .generator import build_model, validate_architecture


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate a PyTorch architecture from YAML")
    parser.add_argument("config", help="path to the YAML architecture file")
    parser.add_argument(
        "--check-shapes", action="store_true", help="run a zero-valued sample through the model"
    )
    args = parser.parse_args(argv)

    try:
        config = load_config(args.config)
        model = build_model(config)
        print(f"Model: {config.name}")
        print(model)
        if args.check_shapes:
            validation = validate_architecture(config)
            print("Shape trace:")
            for layer in validation.layers:
                source_names = ", ".join(config.layers[layer.index].inputs)
                shapes = ", ".join(str(shape) for shape in layer.input_shapes)
                print(
                    f"  [{layer.index}] {layer.id} ({layer.type}) "
                    f"<- [{source_names}]: [{shapes}] -> {layer.output_shape}"
                )
            print("Model inputs:")
            for name, shape in validation.input_shapes.items():
                print(f"  {name}: {shape}")
            print("Model outputs:")
            for name, shape in validation.output_shapes.items():
                print(f"  {name}: {shape}")
    except (ConfigurationError, RuntimeError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
