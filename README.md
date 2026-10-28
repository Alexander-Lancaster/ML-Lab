# ML-Lab: PyTorch Architecture Generator and Visual Editor

ML-Lab is a machine learning developer tool that turns declarative YAML configurations into executable PyTorch neural networks. It combines a Python library and command-line interface with a React and TypeScript visual editor for inspecting and editing model architectures.

The project addresses a practical deep learning workflow: defining network structure, checking graph connections, and locating tensor dimension errors before training. It supports sequential models, branching computation graphs, residual connections, and multiple inputs and outputs.

## Project highlights

- **Model generation:** Build executable PyTorch `nn.Module` graphs from YAML, including convolutional neural networks (CNNs) and residual architectures.
- **Graph algorithms:** Topologically sort layer dependencies and reject duplicate IDs, unknown references, cycles, and nodes unused by declared outputs.
- **Tensor shape validation:** Run a sample forward pass and report input and output dimensions at each layer to help locate incompatible connections.
- **Visual architecture editor:** Inspect vertical or horizontal graph layouts; add and remove layers; edit connections, layer types, and parameters; import and export YAML locally in the browser.
- **Configuration design:** Separate mutable editor state from validated configurations, return structured validation errors, and serialize models into a canonical, versioned format with backward compatibility for sequential files.
- **Automated quality checks:** Test configuration parsing, graph execution, tensor compatibility, serialization, and a shared Python/JavaScript schema contract. GitHub Actions runs Python tests and frontend build and tests on pushes and pull requests.

## Technology stack

| Area | Technologies |
| --- | --- |
| Machine learning and backend | Python 3.10+, PyTorch, PyYAML |
| Frontend | TypeScript, React, Next.js, vinext, Vite, Tailwind CSS |
| Configuration and contracts | YAML, JSON Schema, Python dataclasses, Ajv, jsonschema |
| Testing and development | pytest, Node.js test runner, ESLint, GitHub Actions |

## Example result

The included [MNIST architecture](examples/mnist.yaml) defines a five-layer CNN. Shape validation traces a single image through the network to a tensor containing 10 output values:

```text
Input       (1, 1, 28, 28)
Conv2d      (1, 16, 28, 28)
ReLU        (1, 16, 28, 28)
MaxPool2d   (1, 16, 14, 14)
Flatten     (1, 3136)
Linear      (1, 10)
```

This example demonstrates model construction and shape validation. The repository does not include a training pipeline or measured classification accuracy.

## Quick start

### Run the Python tool

Requires Python 3.10 or newer. From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .

ml-lab examples/mnist.yaml --check-shapes
ml-lab examples/residual.yaml --check-shapes
```

The command prints the generated model. With `--check-shapes`, it also runs a zero-valued batch of one and prints each layer's tensor dimensions.

### Open the visual editor

Requires Node.js 22.13.0 or newer. From the repository root:

```bash
cd gui
npm ci
npm run dev
```

Open the local URL printed by the development server. The editor includes MNIST and residual examples and can open a local ML-Lab YAML file. Imported files and edits stay in the browser; **Save YAML** downloads the edited architecture, including connection changes and deleted layers.

## How it works

```text
YAML architecture
    -> Parse and validate configuration
    -> Sort graph dependencies
    -> Build PyTorch GraphModel
    -> Optionally execute a sample forward pass and trace tensor shapes

Browser editor
    -> Import or edit architecture
    -> Export canonical YAML
    -> Load and execute with the Python tool
```

The Python implementation separates configuration parsing, model execution, serialization, and mutable editor state. An explicit layer allow-list controls which PyTorch modules a configuration can instantiate. The shared JSON Schema and validation fixtures provide a common structural contract for Python and JavaScript; PyTorch performs runtime tensor shape checks.

## Usage examples

### Define a sequential network

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

Each layer's `params` are passed to the corresponding supported PyTorch module. Unknown fields and unsupported layer types are rejected. Sequential configurations receive generated IDs such as `layer_0` and connections to the preceding layer.

### Define a residual graph

Graph configurations use named inputs, unique layer IDs, explicit connections, and declared outputs:

```yaml
format_version: 1

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

`Add` requires at least two tensors with identical shapes. `Concatenate` joins at least two tensors along a configured `dim`; the remaining dimensions must match. Runtime shape validation checks these constraints during a sample forward pass.

### Build and validate from Python

```python
import torch
from ml_lab import build_model, load_config, validate_architecture

config = load_config("examples/mnist.yaml")
model = build_model(config)
output = model(torch.zeros(1, 1, 28, 28))

result = validate_architecture(config)
print(output.shape)          # torch.Size([1, 10])
print(result.output_shape)   # (1, 10)
```

For multiple inputs, pass tensors by name, for example `model(image=image_tensor, metadata=metadata_tensor)`. A single declared output returns one tensor; multiple outputs return a tuple in the order of the YAML `outputs` list.

### Save configurations and validate editor state

```python
from ml_lab import EditableGraph, dump_config, load_config, save_config

config = load_config("examples/mnist.yaml")
print(dump_config(config))
save_config(config, "generated-model.yaml")

editable = EditableGraph.from_config(config)
editable.layers[0].inputs = ["missing_node"]

for issue in editable.validate():
    print(issue.code, issue.node_id, issue.path, issue.message)
```

`EditableGraph` permits temporarily invalid edits. `validate(check_shapes=True)` includes PyTorch shape checks; `to_config()` produces a validated configuration or raises `ConfigurationError`. Structured issues expose a stable code, field path, optional node ID, and readable message.

Files without `format_version` are treated as version 1. Saved files always include the version and use the canonical graph representation, including files originally written in sequential shorthand.

## Tests and continuous integration

Run the Python suite from the repository root:

```bash
python -m pip install -e '.[dev]'
pytest
```

Run the frontend build and tests:

```bash
cd gui
npm ci
npm test
```

The suites cover valid and invalid graph structures, residual and concatenation operations, multiple inputs and outputs, per-layer shape tracing, YAML round trips, editable-state validation, shared schema fixtures, and frontend server rendering. The [CI workflow](.github/workflows/ci.yml) runs both suites.

## Code map

| Location | Responsibility |
| --- | --- |
| [src/ml_lab/config.py](src/ml_lab/config.py) | Configuration parsing, graph validation, and dependency ordering |
| [src/ml_lab/generator.py](src/ml_lab/generator.py) | PyTorch model construction, graph execution, and tensor shape tracing |
| [src/ml_lab/editable.py](src/ml_lab/editable.py) | Mutable graph state and structured validation issues |
| [src/ml_lab/serialization.py](src/ml_lab/serialization.py) | Canonical YAML serialization and file export |
| [src/ml_lab/cli.py](src/ml_lab/cli.py) | Command-line interface |
| [gui/app/GraphViewer.tsx](gui/app/GraphViewer.tsx) | Interactive architecture viewer and editor |
| [schema/](schema/) | Shared JSON Schema and validation fixtures |
| [examples/](examples/) | MNIST CNN and residual architecture configurations |
| [tests/](tests/) and [gui/tests/](gui/tests/) | Python and frontend automated tests |
