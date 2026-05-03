# Phase3 Decision Engine

This folder includes the runnable Phase3 decision implementation:

- `phase3_decision_engine.py`

It consumes Phase 2 role candidates and produces in one pass:

1. Structured group-level conflict diagnosis and deterministic rectification actions.
2. Rectified role candidates with traceable decision metadata.
3. Final tuple arbitration output (merged former Phase 4 behavior).

## Design Policy

- Default mode is deterministic and rule-only (`mode=rule_only`), with no model dependency.
- Optional model-assist mode is available (`mode=qwen_assist`) and can be limited to high-risk groups only.
- Action set is constrained to:
  - `reject_hypothesis`
  - `revise_group`
  - `revise_harmful`
  - `keep_candidate`

## Config

Template config:

- `DAR/configs/phase3_template.json`
- `DAR/configs/phase3_dynamic_beacon_template.json`

Dynamic-beacon bilingual auditor assets:

- `DAR/prompts/phase3_auditor_prompt_zh_v3_fewshot.txt`
- `DAR/prompts/phase3_auditor_prompt_en_v3_fewshot.txt`

The dynamic-beacon Phase 3 path supports:

- Language-specific auditor prompt selection through `paths.auditor_prompt_map`
- Separate zh/en runs through `runtime.default_languages` or `--languages`
- Language-homogeneous auditor micro-batches through `phase3.auditor_language_homogeneous_batches=true`

Key defaults:

- `phase3.mode=rule_only`
- `phase3.enable_model_assist=false`
- `phase3.model_assist_on_high_risk_only=true`
- `phase3.disable_rectification=false` (set `true` for `w/o Decision Rectification` ablation)

Optional model-assist prompt:

- `DAR/prompts/phase3_auditor_prompt_v1.txt`

## Run

Rule-only smoke test:

`/usr/2Tusr/lvyifan3/.conda/envs/DAR/bin/python DAR/experiments/phase3_decision_engine.py --config DAR/configs/phase3_template.json --mode rule_only --max-samples 1 --run-id phase3_smoke_rule --dry-run`

Optional model-assist smoke test:

`/usr/2Tusr/lvyifan3/.conda/envs/DAR/bin/python DAR/experiments/phase3_decision_engine.py --config DAR/configs/phase3_template.json --mode qwen_assist --enable-model-assist --max-samples 1 --max-new-tokens 192 --run-id phase3_smoke_assist --dry-run`

## Outputs

Per run output root:

- `DAR/outputs/phase3/<run_id>/logs/events.jsonl`
- `DAR/outputs/phase3/<run_id>/reports/quality_gates.json`
- `DAR/outputs/phase3/<run_id>/reports/summary.json`
- `DAR/outputs/phase3/<run_id>/run_summary.json`
- `DAR/outputs/phase3/<run_id>/candidates/group_decisions.jsonl` (skipped in `--dry-run`)
- `DAR/outputs/phase3/<run_id>/candidates/rectified_roles.jsonl` (skipped in `--dry-run`)
- `DAR/outputs/phase3/<run_id>/candidates/final_tuples.jsonl` (skipped in `--dry-run`)
- `DAR/outputs/phase3/<run_id>/candidates/final_tuples_by_split/{train,dev,test}.jsonl` (skipped in `--dry-run`)
- `DAR/outputs/phase3/<run_id>/candidates/final_tuples_by_lang_split/{zh_train,zh_test,en_train,en_test,...}.jsonl` (skipped in `--dry-run`)

## Phase 3.5 Handoff

After Phase 3, run the Aggregator to restore sample-level multi-target outputs:

- Entry: `DAR/experiments/phase35_aggregator.py`
- Config: `DAR/configs/phase35_template.json`
- Input: `DAR/outputs/phase3/latest/candidates/final_tuples.jsonl`
- Output: `DAR/outputs/phase3.5/latest/candidates/restored_multi_target.jsonl`
- Eval-friendly export: `DAR/outputs/phase3.5/latest/candidates/phase0_like.jsonl`
