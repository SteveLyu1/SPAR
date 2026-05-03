# Phase 2 Localized Multi-Role Generation

This folder now includes a runnable Phase 2 implementation:

- `phase2_role_generator.py`

It consumes Phase 1 thought-group candidates and produces role-specific tuple candidates for:

- `victim`
- `moderator`
- `cultural_bystander`

Key baseline-aligned behavior:

1. Three role agents with role-specific charters.
2. Structured JSON output validation.
3. One parse retry for malformed model output.
4. One localized self-check round per role output.
5. Uncertainty and escalation flags for downstream auditor ingestion.
6. Physical micro-batch execution for 24GB GPU-friendly processing.

Efficiency defaults (phase1-style):

- Batched role inference is enabled (`phase2.batch_role_inference=true`).
- CUDA cache generation is enabled (`inference.use_cache=true`).
- Thinking mode is disabled by default (`inference.enable_thinking=false`) to reduce `<think>` verbosity and improve strict JSON parse stability.
- Optional 8-bit loading is enabled (`inference.load_in_8bit_prefer=true`) with automatic fallback.
- Multi-GPU memory floor is enforced (`inference.min_multi_gpu_mem_ratio=0.85`) when `device_map=balanced`.
- Prompt truncation limit is enabled (`inference.max_input_tokens=1536`) to avoid long-tail latency spikes.

## Config

Template config: `DAR/configs/phase2_template.json`

Default model:

- `/usr/2Tusr/llm/Qwen3-14B`

Teacher-grade override (when available):

- `/usr/2Tusr/llm/Qwen3.5-35B-A3B-FP8`

Role-synced baseline plan (2026-04-10):

- `victim`: Qwen3-14B (prompt-only)
- `moderator`: Qwen3-14B (prompt-only)
- `cultural_bystander`: Qwen3-14B (prompt-only + annotated lexicon retrieval injection)
- Lexicon policy: on ZH track, Python retrieves hits from `Datasets/STATE-ToxiCN-main/data/annotated lexicon.json` and injects them into `{{LEXICON_KNOWLEDGE}}` before generation.

Current runner note:

- `phase2_role_generator.py` accepts one model path per run.
- Mainline execution uses one pass with all three roles on Qwen3-14B.
- Runtime split inputs are pinned to focus-text files for train/test in both languages.
- Runtime filters support split/language selection (`--splits`, `--languages`).
- Checkpoint metadata stores expected finish ETA (`latest_eta_iso`) and sampling timestamp (`eta_last_sampled_at`).

Prompt template:

- Language-aware role overrides are configured through `paths.role_prompt_map`:
  - `DAR/prompts/phase2_role_prompt_victim_en_v2.txt`
  - `DAR/prompts/phase2_role_prompt_victim_zh_v2.txt`
  - `DAR/prompts/phase2_role_prompt_moderator_en_v2.txt`
  - `DAR/prompts/phase2_role_prompt_moderator_zh_v2.txt`
  - `DAR/prompts/phase2_role_prompt_cultural_bystander_en_v2.txt`
  - `DAR/prompts/phase2_role_prompt_cultural_bystander_zh_v2.txt`

## Run

Heuristic smoke test:

```bash
/usr/2Tusr/lvyifan3/.conda/envs/DAR/bin/python DAR/experiments/phase2_role_generator.py \
  --config DAR/configs/phase2_template.json \
  --mode heuristic \
  --max-samples 4 \
  --run-id phase2_smoke_heuristic \
  --dry-run
```

Local Qwen smoke test:

```bash
/usr/2Tusr/lvyifan3/.conda/envs/DAR/bin/python DAR/experiments/phase2_role_generator.py \
  --config DAR/configs/phase2_template.json \
  --mode qwen_local \
  --max-samples 1 \
  --max-new-tokens 128 \
  --run-id phase2_smoke_qwen \
  --dry-run
```

Qwen3.5 upper-bound control smoke test (optional):

```bash
/usr/2Tusr/lvyifan3/.conda/envs/DAR/bin/python DAR/experiments/phase2_role_generator.py \
  --config DAR/configs/phase2_template.json \
  --mode qwen_local \
  --model-path /usr/2Tusr/llm/Qwen3.5-35B-A3B-FP8 \
  --max-samples 1 \
  --max-new-tokens 128 \
  --run-id phase2_smoke_qwen35 \
  --dry-run
```

## Outputs

Per run:

- `DAR/outputs/phase2/<run_id>/logs/events.jsonl`
- `DAR/outputs/phase2/<run_id>/reports/quality_gates.json`
- `DAR/outputs/phase2/<run_id>/reports/summary.json`
- `DAR/outputs/phase2/<run_id>/run_summary.json`
- `DAR/outputs/phase2/<run_id>/candidates/all_roles.jsonl` (skipped in `--dry-run`)
- `DAR/outputs/phase2/<run_id>/candidates/roles/*.jsonl` (skipped in `--dry-run`)
