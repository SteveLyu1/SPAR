# Dance Hate Triplet Adaptation

This directory contains the retained final Dance-style adaptation code included in the public release.

## Retained files

- `scripts/run_dance_core_api_inference.py`
- `evaluation/target_level_eval.py`
- `evaluation/tbo_eval.py`
- `evaluation/tbo_harm_paperlike_eval.py`
- `lib/api_adapter.py`
- `lib/json_utils.py`
- `requirements.txt`

## Task

- English output: `(target, argument, harmful)`
- Chinese output: `(target, argument, hateful)`

## Run

```bash
python scripts/run_dance_core_api_inference.py \
  --input-jsonl ../../datasets/tbo_manual_split/cleaned_en_test.jsonl \
  --output-root outputs \
  --language en
```

## Evaluation

```bash
python evaluation/tbo_eval.py --input-jsonl outputs/YOUR_RUN/phase0_like.jsonl
python evaluation/tbo_harm_paperlike_eval.py --input-jsonl outputs/YOUR_RUN/phase0_like.jsonl
```
