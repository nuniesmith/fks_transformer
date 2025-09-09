# FKS Transformer Service

Performs sequence modeling / inference over price series & enriched features.

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install .[ml]
python -m fks_transformer.main
```

## Next Steps

- Add model registry & hot-reload
- Add ONNX export path
- Implement batch inference queue
