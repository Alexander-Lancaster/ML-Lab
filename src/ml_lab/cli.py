"""Command-line interface for model generation."""

from __future__ import annotations

import argparse

from .config import ConfigurationError, load_config
from .generator import build_model, validate_forward_pass


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
            output_shape = validate_forward_pass(model, config.input_shape)
            print(f"Input shape:  (1, {', '.join(map(str, config.input_shape))})")
            print(f"Output shape: {output_shape}")
    except (ConfigurationError, RuntimeError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
