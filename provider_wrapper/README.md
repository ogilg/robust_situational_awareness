# Custom Evals Package

A Python package containing HuggingFace provider functionality for custom evaluations.

## Contents

- `huggingface_provider.py`: Core provider functionality with CUDA-optimized model inference
- `request_templates.py`: Data structures for API requests and responses

## Installation

Install the package in development mode:

```bash
cd custom_evals
uv pip install -e .
```

## Usage

The package provides:

- `generate_single_token()`: Generate single tokens with CUDA handling
- `HUGGINGFACE_MODEL_MAPPING`: Mapping of model IDs to HuggingFace model names
- Request/Response templates for structured API interactions

## Integration

The run scripts in `sad/anti_imitation/output_control/` use this package:

```python
from custom_evals import HUGGINGFACE_MODEL_MAPPING, generate_single_token
``` 