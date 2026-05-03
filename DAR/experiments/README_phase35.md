# Phase 3.5 Aggregator

This stage restores sample-level multi-target outputs after Phase 3 group-level decisions.

- Input: `DAR/outputs/phase3/latest/candidates/final_tuples.jsonl`
- Entry: `DAR/experiments/phase35_aggregator.py`
- Config: `DAR/configs/phase35_template.json`

## What It Does

1. Groups Phase 3 `final_tuples` by `sample_id`.
2. Merges group-level final tuples back into sample-level multi-target tuples.
3. Preserves every group-level final tuple during sample-level restoration, so `g1/g2/...` clause outputs are purely concatenated instead of being deduplicated away.
4. Exports split-wise restored outputs for downstream evaluation or distillation.
5. Rebuilds a Phase-0-like sample schema by joining back to `DAR/outputs/phase0/cleaned_*.jsonl`.

The new `phase0_like` export is evaluation-oriented:

- Root fields align with Phase 0: `id`, `original_id`, `source`, `language`, `split`, `text`, `meta_info`
- Restored predictions are stored in `predictions`
- If reference Phase 0 rows are available, original `ground_truth` is copied in for direct metric calculation

## Run

Smoke test:

```bash
/usr/2Tusr/lvyifan3/.conda/envs/DAR/bin/python DAR/experiments/phase35_aggregator.py \
  --config DAR/configs/phase35_template.json \
  --max-samples 8 \
  --run-id phase35_smoke \
  --dry-run
```

Full run:

```bash
/usr/2Tusr/lvyifan3/.conda/envs/DAR/bin/python DAR/experiments/phase35_aggregator.py \
  --config DAR/configs/phase35_template.json \
  --run-id phase35_full
```

## Outputs

Per run output root:

- `DAR/outputs/phase3.5/<run_id>/logs/events.jsonl`
- `DAR/outputs/phase3.5/<run_id>/reports/quality_gates.json`
- `DAR/outputs/phase3.5/<run_id>/reports/summary.json`
- `DAR/outputs/phase3.5/<run_id>/run_summary.json`
- `DAR/outputs/phase3.5/<run_id>/candidates/aggregated_samples.jsonl` (skipped in `--dry-run`)
- `DAR/outputs/phase3.5/<run_id>/candidates/restored_multi_target.jsonl` (skipped in `--dry-run`)
- `DAR/outputs/phase3.5/<run_id>/candidates/by_split/*.jsonl` (skipped in `--dry-run`)
- `DAR/outputs/phase3.5/<run_id>/candidates/phase0_like.jsonl` (skipped in `--dry-run`)
- `DAR/outputs/phase3.5/<run_id>/candidates/phase0_like_by_split/*.jsonl` (skipped in `--dry-run`)
- `DAR/outputs/phase3.5/<run_id>/candidates/phase0_like_by_lang_split/*.jsonl` (skipped in `--dry-run`)

A stable latest pointer is also written to:

- `DAR/outputs/phase3.5/latest/...`
