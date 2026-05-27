# SynChain Hate Triplet Adaptation

This directory contains the retained final SynChain-style adaptation code included in the public release.

## Retained files

- `src/synchain_hate_triplet/run_inference.py`
- `src/synchain_hate_triplet/api_adapter.py`
- `src/synchain_hate_triplet/json_utils.py`
- `src/synchain_hate_triplet/syntax_parser.py`
- `src/synchain_hate_triplet/evaluation.py`
- `requirements.txt`

## Task

- English output: `(target, argument, harmful)`
- Chinese output: `(target, argument, hateful)`

## Run

```bash
PYTHONPATH=src python -m synchain_hate_triplet.run_inference \
  --input-jsonl ../../datasets/tbo_manual_split/cleaned_en_test.jsonl \
  --output-root outputs \
  --language en
```
