# Phase 4 Distillation: Single-Pass Superprompt

## Goal
Implement the confirmed Phase4 setting as language-separated training tracks:
- data from prior phases
- deterministic pseudo think chain
- one super prompt
- one-pass SFT per language adapter

No Phase4 internal ablation is required in this version.

## Implemented Artifacts
Phase-level new outputs:
- Phase1 writes candidates/phase4_decompose.jsonl and embeds phase4_decompose into each group row.
- Phase2 writes candidates/phase4_role_hypotheses.jsonl and embeds phase4_role_hypothesis into each role row.
- Phase3 writes candidates/phase4_trace_pack.jsonl (single joined trace per decision_id).

Phase4 new builder:
- DAR/fine_tuning/build_phase4_superprompt_dataset.py

Phase4 config template:
- DAR/configs/phase4_distill_template.json

## Data Sources
Primary source:
- DAR/outputs/phase3/latest/candidates/phase4_trace_pack.jsonl

Fallback source (if trace pack is missing):
- phase3 group_decisions + rectified_roles + final_tuples (supports legacy names group_audits and rectified_all)
- optional phase1/phase2 lookup to recover missing decompose or role payloads

## Pseudo-Chain Construction
The chain is deterministic and template-based:
- Step A: Decompose
- Step B: Parallel Role Hypotheses
- Step C: Conflict Diagnosis
- Step D: Rectification Actions
- Step E: Final Arbitration

Each step only uses values already present in phase artifacts.
The current baseline renders these steps as short natural-language pseudo reasoning chains in `<think>`, closer to DanceHA-style supervision than the earlier compressed slot-only format.

## Output Dataset Format
One primary dataset plus language-split views are produced:
- outputs/phase4/<run_id>/datasets/single_pass_superprompt.jsonl

Each row includes:
- messages[system]: super prompt
- messages[user]: structured phase trace input package
- messages[assistant]:
  - <think> deterministic chain </think>
  - strict final tuple JSON

By-split exports:
- outputs/phase4/<run_id>/datasets/by_split/train.jsonl
- outputs/phase4/<run_id>/datasets/by_split/dev.jsonl
- outputs/phase4/<run_id>/datasets/by_split/test.jsonl

By-language-and-split exports:
- outputs/phase4/<run_id>/datasets/variants/<variant>/by_lang_split/zh_train.jsonl
- outputs/phase4/<run_id>/datasets/variants/<variant>/by_lang_split/en_train.jsonl
- outputs/phase4/<run_id>/datasets/variants/<variant>/by_lang_split/zh_test.jsonl
- outputs/phase4/<run_id>/datasets/variants/<variant>/by_lang_split/en_test.jsonl

Latest pointer:
- outputs/phase4/latest/datasets/single_pass_superprompt.jsonl

## Quality Gates
Builder enforces:
- unique decision_id
- valid 3-turn chat message schema
- final tuple JSON parseability
- strict JSON tuple keys normalization

Reports:
- outputs/phase4/<run_id>/reports/quality_gates.json
- outputs/phase4/<run_id>/reports/summary.json

## Run Command
Example:

python DAR/fine_tuning/build_phase4_superprompt_dataset.py \
  --config DAR/configs/phase4_distill_template.json

Common overrides:
- --hard-case-only
- --hard-case-threshold 0.67
- --max-samples N
- --splits train,dev

## Training Policy (Current Baseline)
Phase4 uses one-pass SFT on single_pass_superprompt.jsonl.

Recommended students:
- Qwen3-4B-Instruct ZH adapter
- Qwen3-4B-Instruct EN adapter

Optional control:
- Qwen3-8B ZH/EN adapters

Phase4 is treated as an extra exploration track outside core ablation grid.
