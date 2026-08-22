# ML-Lab

ML-Lab generates executable PyTorch neural-network graphs from validated YAML
files. It supports sequential networks, branches, residual connections, named
inputs, and multiple outputs.

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

The original sequential format remains supported. ML-Lab assigns generated IDs
such as `layer_0` and connects each layer to the preceding one.

## Define a graph with branches

Graph architectures use named inputs, unique layer IDs, explicit input
references, and declared outputs. This example creates a residual connection:

```yaml
model:
  name: residual_example

inputs:
  image:
    shape: [3, 32, 32]

layers:
  - id: stem
    type: Conv2d
    inputs: [image]
    params:
      in_channels: 3
      out_channels: 16
      kernel_size: 3
      padding: 1

  - id: transformed
    type: Conv2d
    inputs: [stem]
    params:
      in_channels: 16
      out_channels: 16
      kernel_size: 3
      padding: 1

  - id: residual
    type: Add
    inputs: [stem, transformed]

outputs: [residual]
```

`Add` requires at least two tensors with identical shapes. `Concatenate`
combines at least two tensors and accepts a `dim` parameter:

```yaml
- id: combined
  type: Concatenate
  inputs: [left_branch, right_branch]
  params:
    dim: 1
```

Graph validation rejects duplicate IDs, unknown references, dependency cycles,
unused nodes, unknown outputs, and invalid merge shapes before the model is used.

## Generate the model

```bash
ml-lab examples/mnist.yaml
ml-lab examples/mnist.yaml --check-shapes
ml-lab examples/residual.yaml --check-shapes
```

From Python:

```python
from ml_lab import build_model, load_config

config = load_config("examples/mnist.yaml")
model = build_model(config)
```

`--check-shapes` performs a forward pass with a zero-valued batch of one. This
catches incompatible dimensions in addition to validating the YAML structure.
It prints the input and output shape at every layer, making dimension errors
easy to locate:

```text
Shape trace:
  [0] layer_0 (Conv2d) <- [input]: [(1, 1, 28, 28)] -> (1, 16, 28, 28)
  [1] layer_1 (ReLU) <- [layer_0]: [(1, 16, 28, 28)] -> (1, 16, 28, 28)
  [2] layer_2 (MaxPool2d) <- [layer_1]: [(1, 16, 28, 28)] -> (1, 16, 14, 14)
  [3] layer_3 (Flatten) <- [layer_2]: [(1, 16, 14, 14)] -> (1, 3136)
  [4] layer_4 (Linear) <- [layer_3]: [(1, 3136)] -> (1, 10)
```

Architecture validation is also available in Python:

```python
from ml_lab import load_config, validate_architecture

result = validate_architecture(load_config("examples/mnist.yaml"))
print(result.output_shape)
```

The generated `GraphModel` accepts a positional tensor for a single input. For
multiple inputs, pass tensors by name:

```python
output = model(image=image_tensor, metadata=metadata_tensor)
```

A single declared output is returned as one tensor. Multiple declared outputs
are returned as a tuple in the same order as the YAML `outputs` list.
