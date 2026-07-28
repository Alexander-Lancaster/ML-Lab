# ML-Lab

ML-Lab generates PyTorch neural-network architectures from validated YAML files.
The initial version intentionally supports sequential architectures only.

## Install

Python 3.10 or newer is required. From the project directory:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

For development and tests, install the development extra:

```bash
python -m pip install -e '.[dev]'
pytest
```

## Define an architecture

```yaml
model:
  name: small_classifier

input:
  shape: [4]

layers:
  - type: Linear
    params:
      in_features: 4
      out_features: 8
  - type: ReLU
  - type: Linear
    params:
      in_features: 8
      out_features: 2
```

Each layer's `params` are passed to the corresponding allow-listed PyTorch layer.
Unknown YAML fields and unsupported layer types are rejected.

## Generate the model

```bash
ml-lab examples/mnist.yaml
ml-lab examples/mnist.yaml --check-shapes
```

From Python:

```python
from ml_lab import build_model, load_config

config = load_config("examples/mnist.yaml")
model = build_model(config)
```

`--check-shapes` performs a forward pass with a zero-valued batch of one. This
catches incompatible dimensions in addition to validating the YAML structure.
