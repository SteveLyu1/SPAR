# Phase 0 (Final Cleaned JSONL Only)

Phase 0 now only does schema normalization and split assignment.

- STATE-ToxiCN keeps original train/test entries.
- TBO is split deterministically to train=3200, test=800.
- No language balancing.
- No multi-target divide in Phase 0.

## Final Output Files

Only four cleaned JSONL files are kept under `DAR/outputs/phase0/`:

- `cleaned_zh_train.jsonl`
- `cleaned_zh_test.jsonl`
- `cleaned_en_train.jsonl`
- `cleaned_en_test.jsonl`

## Record Schema

Root fields:

- `id`
- `original_id`
- `source`
- `language`
- `split`
- `text`
- `meta_info`
- `ground_truth`

`meta_info` currently keeps only `topic`.

Ground truth by language:

- ZH: `target`, `argument`, `group`, `hateful`
- EN: `target`, `argument`, `harmful`

ID policy:

- STATE: `zh_<original_id>`
- TBO: sequential `en_0001` ... `en_4000`

## Run

```bash
python DAR/datapreprocess/phase0_pipeline.py \
  --config DAR/configs/phase0_template.json
```

Smoke test:

```bash
python DAR/datapreprocess/phase0_pipeline.py \
  --config DAR/configs/phase0_template.json \
  --max-state-train-docs 200 \
  --max-state-test-docs 80 \
  --max-tbo-docs 1000 \
  --tbo-train-size 800 \
  --tbo-test-size 200
```

Cleanup existing JSONL files in `DAR/outputs/phase0` (remove redundant `sample_id` and `meta_info.platform`):

```bash
python DAR/datapreprocess/cleanup_phase0_outputs.py \
  --output-root DAR/outputs/phase0
```
