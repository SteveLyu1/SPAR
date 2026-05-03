# Phase 1 Segmenter (Compact)

This folder contains a compact Phase 1 Segmenter implementation for DAR-Hate.

## Architecture

- `phase1_segmenter.py`: single-file implementation containing all pipeline logic (split/combine/normalize/evidence/confidence/export) and CLI entrypoint.
- `DAR/prompts/phase1_segmenter_prompt_v2_lang_source_fewshot.txt`: externalized segmenter prompt template.

## Baseline Alignment

The implementation explicitly covers baseline Phase 1 requirements:

1. High-recall target-centric decomposition.
2. Split one sentence into multiple groups when it contains multiple targets or harmful intents.
3. Merge adjacent groups with same canonical target and compatible harmful direction.
4. Normalize aliases/pronouns/coreference into stable canonical target identifiers.
5. Keep sentence-level and span-level evidence.
6. Output group-level and field-level confidence plus escalation flags.

Core output fields include:

- `group_id`, `language`, `source`, `sentence_ids`, `group_text`
- `canonical_target`, `canonical_target_id`, `target_surface_mentions`
- `target_spans`, `argument_spans`, `harmful_cue_spans`
- `segmenter_confidence`, `field_confidence`, `escalation_flag`

## Default Config

Config file: `DAR/configs/phase1_template.json`

- `model_root`: `/usr/2Tusr/llm`
- `segmenter_model_path`: `/usr/2Tusr/llm/Qwen3-14B`
- `segmenter_prompt_path`: `DAR/prompts/phase1_segmenter_prompt_v2_lang_source_fewshot.txt`
- default mode: `qwen_local`

Recommended efficiency fallback when needed:

- `/usr/2Tusr/llm/Qwen3.5-9B`

GPU-first defaults (notebook-aligned):

- `inference.require_cuda=true`: fail fast instead of silently running on CPU.
- `inference.device_map=balanced`: distribute model layers across visible GPUs.
- `inference.prefer_4090_order=true`: prefer 4090-first GPU ordering when auto-detecting devices.
- `inference.preferred_gpu_mem_ratio=0.97`, `other_gpu_mem_ratio=0.90`: build `max_memory` map for multi-GPU placement.
- `inference.dtype=float16`: CUDA-friendly default dtype for throughput.

Throughput-oriented knobs:

- `inference.batch_size`: qwen_local micro-batch size (default 6).
- `inference.max_input_tokens`: truncation cap for faster and more stable batched generation.
- `inference.min_multi_gpu_mem_ratio`: floor for secondary GPU memory budget when using balanced multi-GPU mapping.
- `inference.use_cache=true`: enable KV cache for faster autoregressive decoding.

## Run

Heuristic smoke test:

```bash
/usr/2Tusr/lvyifan3/.conda/envs/DAR/bin/python DAR/experiments/phase1_segmenter.py \
  --config DAR/configs/phase1_template.json \
  --mode heuristic \
  --max-samples 64 \
  --run-id phase1_smoke_heuristic \
  --dry-run
```

Qwen local smoke test:

```bash
/usr/2Tusr/lvyifan3/.conda/envs/DAR/bin/python DAR/experiments/phase1_segmenter.py \
  --config DAR/configs/phase1_template.json \
  --mode qwen_local \
  --max-samples 1 \
  --max-new-tokens 64 \
  --run-id phase1_smoke_qwen \
  --dry-run
```

## Outputs

Per run output root:

- `DAR/outputs/phase1/<run_id>/logs/events.jsonl`
- `DAR/outputs/phase1/<run_id>/reports/quality_gates.json`
- `DAR/outputs/phase1/<run_id>/reports/summary.json`
- `DAR/outputs/phase1/<run_id>/run_summary.json`
- `DAR/outputs/phase1/<run_id>/candidates/*.jsonl` (skipped in `--dry-run`)
