# Code Generation Models Project

A production-ready code generation system using transformer-based models.

## Features

- Multiple model architectures (GPT-2, CodeT5, CodeGen)
- Comprehensive evaluation metrics (BLEU, CodeBLEU, pass@k)
- Interactive web interface
- Configurable training and inference
- Support for multiple programming languages

## Quick Start

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Download datasets:
```bash
python scripts/download_datasets.py
```

3. Train a model:
```bash
python scripts/train.py --config configs/gpt2_codegen.yaml
```

4. Run interactive demo:
```bash
streamlit run demo/app.py
```

## Project Structure

```
src/
├── models/          # Model implementations
├── data/            # Data loading and preprocessing
├── training/        # Training loops and utilities
├── evaluation/      # Evaluation metrics and tools
└── utils/           # Utility functions

configs/             # Configuration files
scripts/             # Training and evaluation scripts
demo/                # Interactive web interface
tests/               # Unit tests
assets/              # Generated samples and checkpoints
```

## Models

- **GPT-2**: General-purpose transformer for code generation
- **CodeT5**: Specialized for code understanding and generation
- **CodeGen**: Large-scale code generation model

## Evaluation

The project includes comprehensive evaluation metrics:
- BLEU score for text similarity
- CodeBLEU for code-specific evaluation
- Pass@k for functional correctness
- Execution success rate

## Configuration

All training and inference parameters are configurable via YAML files in the `configs/` directory.

## Contributing

1. Install pre-commit hooks: `pre-commit install`
2. Run tests: `pytest tests/`
3. Format code: `black src/` and `ruff check src/`
# Code-Generation-Models-Project
