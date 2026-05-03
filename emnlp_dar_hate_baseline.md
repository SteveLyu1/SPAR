# EMNLP DAR-Hate Baseline (Project Memory)
DAR-Hate: An Auditor-Guided Multi-Agent Distillation Framework for Bilingual Hate Speech Parsing

Synchronized to current implementation and released results on 2026-04-30.

## Scope
- Task: bilingual (ZH/EN), language-decoupled cultural-aware hate parsing.
- Serialized tuple contract: ZH uses `(target, argument, group, hateful)` with Harmful/Hateful semantics aligned to the Chinese protocol; EN uses `(target, argument, harmful)`.
- Corpus interpretation lock: treat STATE-ToxiCN as the Chinese hate-focused reference track and TBO as the English broader offensive / targeted-abuse track; do not collapse the two corpora as fully identical label spaces.
- Reporting lock: core execution and paper-facing reporting are language-decoupled by default; bilingual joint-training or mixed-data claims must be treated as supplementary settings unless explicitly matched and disclosed.
- Goal: publishable EMNLP pipeline under dual 24GB GPU constraints (RTX 3090 + RTX 4090D) with micro-batch execution.

## Frozen Core Pipeline
1) Phase 0 Dataset tagging + split freeze (no cross-corpus group alignment)
- Normalize STATE-ToxiCN and TBO to one document-level schema with root fields: id, original_id, source, language, split, text, meta_info, ground_truth.
- Keep root text untouched for segmenter-ready prompting; remove flat root target/argument keys.
- Keep explicit provenance by source/language/split and original_id; use lightweight ids only.
- ID policy lock: STATE id = zh_<original_id>; TBO id uses sequential short ids (en_0001 ... en_N), with original snowflake id stored in original_id.
- Ground-truth schema lock: ZH tuple uses (target, argument, group, hateful); EN tuple uses (target, argument, harmful) only.
- Label normalization lock: raw labels are mapped to integer 0/1 while preserving field names (ZH keeps hateful, EN keeps harmful).
- Freeze at document granularity: one source document corresponds to one JSONL row, with all tuples kept in ground_truth.
- Phase0 is formatting-only: do not perform multi-target divide at this stage; keep multi-target structure for Phase1 Segmenter.
- Split policy lock: STATE keeps original train/test; TBO uses fixed train/test sizes (3200/800) with deterministic shuffling by seed.
- Meta policy lock: Phase0 no longer emits sample_id and does not keep meta_info.platform in outputs.
- No cross-language balancing in Phase0 mainline because downstream uses language-decoupled prompts and students.
- Deliverables: cleaned split JSONL files only: cleaned_zh_train.jsonl, cleaned_zh_test.jsonl, cleaned_en_train.jsonl, cleaned_en_test.jsonl.

2) Phase 1 Segmenter (dual-track text-only decomposition)
- Backbone: Qwen3-14B as the primary Segmenter for both languages with language-specific prompt templates; GPU inference settings remain enabled (dtype/device_map/batch/memory policy preserved in config).
- Dual-track routing: process ZH and EN in separate queues and separate batches (no mixed-language batching).
- De-label policy: before LLM inference, remove ground_truth and only send text into prompt (metadata stays in Python memory for routing/injection only).
- LLM contract: model output must be strict JSON array containing exactly local_group_id, group_text, canonical_target.
- Postprocess hardcode injection:
	- group_id = sample_id_g{local_group_id}
	- metadata inheritance = original_id/source/language/split
	- escalation_flag = (canonical_target == IMPLICIT_TARGET)
- Canonical-target lock (must keep for downstream routing): `canonical_target` is the shared trace anchor for regrouping, soft target hinting, and stable Phase 3 clustering plus Phase 3.5 aggregation, but it is no longer a hard generation constraint inside Phase 2.
- Language-decoupled reasoning lock: ZH/EN prompts and inference are fully separated to prevent context contamination and allow per-language compute allocation.
- Output (mainline compact): group_id, sample_id (compatibility), original_id, source, language, split, local_group_id, group_text, canonical_target, escalation_flag.
- Compact-export lock: the frozen Phase 1 baseline keeps compact group rows as the canonical export; span/confidence-rich traces are optional analysis artifacts and are not part of the mainline contract.
- Deliverables: four split candidate JSONL files (zh_train, zh_test, en_train, en_test) under Phase1 outputs.

Additional Declarations (Phase1 rewrite)
- Python handles global id/routing/state; LLM handles semantic decomposition only.
- JSON parsing path includes repair + balanced array extraction + strict deserialization retry.
- sample_id is retained in Phase1 export as a compatibility field for existing Phase2/Phase3 code paths, while group_id/original_id are treated as canonical trace keys.

3) Phase 2 Localized multi-perspective role generation
- Backbone (role-synced): Victim, Moderator, and Cultural Bystander all use Qwen3-14B with role-specific prompts; Cultural Bystander additionally uses annotated-lexicon retrieval then prompt injection (`{{LEXICON_KNOWLEDGE}}`) for cultural-context reasoning.
- Topology lock: the 3-agent Victim / Moderator / Cultural-Bystander setup is the frozen mainline. Compressed 2-agent variants remain exploratory efficiency controls only and are outside the locked main ablation grid.
- Data ingestion lock: Phase 2 reads language/split files via `runtime.phase1_language_split_inputs` and routes by language/split.
- Focus-file lock (current default): inference inputs are pinned to `DAR/outputs/phase1/phase1_focus/phase1_zh_train_focus_text_min.jsonl`, `DAR/outputs/phase1/phase1_focus/phase1_zh_test_focus_text.jsonl`, `DAR/outputs/phase1/phase1_focus/phase1_en_train_focus_text_min.jsonl`, and `DAR/outputs/phase1/phase1_focus/phase1_en_test_focus_text.jsonl`.
- Focus-lock prompt input: LLM prompt input uses `focus_text` (with `<focus>...</focus>` tags) as the primary clause, with `group_text` as fallback when `focus_text` is missing. `{{FOCUS_TEXT}}` must receive the tagged focus span, while `{{GROUP_TEXT}}` keeps the raw clause text and must never be silently replaced by the whole tagged prompt string.
- Target-hint policy lock: `canonical_target` is no longer a hard shared target lock inside Phase 2 generation. It is injected only as `{{TARGET_HINT}}` soft hint and may be used only when the focus clause is fragmentary or lacks a subject; it must never override an explicit target already present inside `<focus>`.
- Prompt contract lock: keep six built-in prompt constants for manual evolution (`EN_VICTIM_PROMPT`, `EN_MODERATOR_PROMPT`, `EN_BYSTANDER_PROMPT`, `ZH_VICTIM_PROMPT`, `ZH_MODERATOR_PROMPT`, `ZH_BYSTANDER_PROMPT`) and allow language-aware prompt-file overrides via config (`paths.role_prompt_map`) for controlled updates.
- Victim v2 lock: EN/ZH victim prompts enforce focus-tag attention lock, soft-only target hint usage, shortest-decisive-span extraction, multi-argument split, anti-merge rules for conjunction/comma-linked attacks, and strict JSON-array key order (`rationale,target,argument,harmful` for EN; `rationale,target,argument,group,hateful` for ZH).
- Moderator v2 lock: EN/ZH moderator prompts enforce focus-tag attention lock, soft-only target hint usage, shortest-decisive-span extraction, multi-argument split, anti-merge rules for conjunction/comma-linked attacks, and strict JSON-array key order (`rationale,target,argument,harmful` for EN; `rationale,target,argument,group,hateful` for ZH).
- Cultural Bystander v2 lock: EN/ZH bystander prompts enforce focus-tag attention lock, soft-only target hint usage, shortest-decisive-span extraction, anti-merge argument rules, and cultural-context reasoning (reclaimed slang, coded derogation, quoting/irony stance); EN keeps key order `rationale,target,argument,harmful`, ZH keeps key order `rationale,target,argument,group,hateful` with `{{LEXICON_KNOWLEDGE}}` as context-only evidence.
- Three-track execution: run Victim/Moderator/Cultural-Bystander in isolated tracks with micro-batch serial scheduling to avoid OOM.
- Single-pass role-sync lock: because the current runner uses one active model path per Phase 2 run, the baseline mainline executes all three roles in one pass on Qwen3-14B; two-pass or mixed-backbone merges remain control configurations only.
- Checkpoint lock: persist atomic `phase2_checkpoint.json` snapshots by role-group interval; resume is allowed only when `role_order` and `group_id_signature` match current input ordering.
- Partial export lock: checkpoint-time writes may additionally emit `group_predictions.partial.jsonl`, `all_roles.partial.jsonl`, and `phase4_role_hypotheses.partial.jsonl` so interrupted long runs still preserve usable intermediate artifacts.
- Progress observability lock: emit hourly ETA samples from processed role-groups (`processed_role_groups / total_role_groups`) to support long-run scheduling.
- GPU memory guard: local generation uses CUDA OOM split-retry plus per-batch cache cleanup; if retries still fail under fallback mode, degrade the affected role/group to `[]` without terminating the full run.
- Reasoning-mode policy: Phase 2 supports both think and no-think generation. The current exact-match-oriented mainline pins `inference.enable_thinking=false`, while template configs still expose think mode for controlled A/B tests; retry attempts can still force direct-JSON no-think fallback.
- Bystander lexicon retrieval and injection: Cultural Bystander runs on Qwen3-14B; before bystander generation on ZH track, Python retrieves relevant entries from `Datasets/STATE-ToxiCN-main/data/annotated lexicon.json` using `focus_text` and `canonical_target`, then injects snippets into `{{LEXICON_KNOWLEDGE}}`; miss keeps this block empty.
- Array output lock: every role must return JSON Array to support 1-to-N targets from one `group_text`.
- Role schema lock:
	- EN object requires `rationale`, `target`, `argument`, `harmful`.
	- ZH object requires `rationale`, `target`, `argument`, `group`, `hateful`.
- Parser robustness: regex array capture + strict JSON deserialization + per-language field validation; malformed output degrades to `[]` for that role/group and logs parse error without stopping the batch.
- Text normalization lock: Phase 2 normalizes both prompt-side text and prediction-side fields with `html.unescape`, whitespace cleanup, and focus-tag stripping for grounded matching, so forms such as `&amp;` vs `&` do not create artificial exact-match misses.
- Postprocess lock: after schema validation, Phase 2 performs language-aware groundedness filtering and tuple cleanup. `argument` must be grounded in the current focus clause under relaxed containment rules; exact duplicates are removed; within the same `(target, label, group)` bucket, longer merged arguments are pruned when a shorter decisive grounded span is already present.
- Group normalization lock (ZH): Chinese `group` values are canonicalized to the fixed set `Sexism`, `Region`, `Racism`, `LGBTQ`, `others`, `non-hate`, `unknown`, with non-hate rows forced to `non-hate`.
- Final zipper merge: join by `group_id` and export group-level nested rows:
	- root metadata: `group_id`, `sample_id`, `original_id`, `source`, `language`, `split`, `local_group_id`, `focus_text`, `group_text`, `canonical_target`, `escalation_flag`
	- prediction payload: `phase2_predictions = {victim: [...], moderator: [...], bystander: [...]}`
- Compatibility export: keep a flattened `all_roles.jsonl` view for existing Phase 3 reader while treating group-level nested JSONL as the Phase 2 canonical output; current implementation also emits `phase4_role_hypotheses.jsonl`, per-role files (`roles/*.jsonl`), split views (`by_split/group_{split}.jsonl`), and language-split views (`by_lang_split/group_{lang}_{split}.jsonl` plus `phase2_{lang}_{split}.jsonl`).
- Minimal hypothesis export lock: Phase 2 compatibility/Phase 4 handoff rows keep only compact role-hypothesis fields (`task`, `role`, `role_candidate_id`, `group_id`, `sample_id`, `tuple`, `rationale`, `uncertainty_flag`, `escalation_flag`) and no longer emit legacy `evidence.*_spans`, `role_confidence`, or `field_confidence`.

4) Phase 3 Dynamic Beacon Auditor Engine (auditor-led soft fusion)
- Core positioning: replace crude majority vote with an auditor-led arbitration engine that keeps all role evidence, soft-weights it, and selects the group-level tuple most likely to match ground truth.
- Canonical runtime: `DAR/experiments/phase3_dynamic_beacon_engine.py`; canonical input is `DAR/outputs/phase2/latest/candidates/group_predictions.jsonl`; `all_roles.jsonl` is retained only as a legacy compatibility export.
- Stage A, candidate normalization and quality scoring: normalize every role hypothesis into a language-aware tuple contract, score each candidate by grounding/target alignment/group consistency/rationale quality, and mark `valid_for_fusion` vs defective outputs.
- Grounding policy lock: Phase 3 candidate quality uses the same relaxed clause-grounding policy as Phase 2 postprocess, so `argument` must still be recoverable from the current focus clause after HTML-entity decoding and punctuation-trimmed matching.
- Canonical-target policy lock: Phase 3 does not hard-reject candidates that differ from `canonical_target`; instead it applies a soft penalty for explicit target conflicts and preserves the candidate if grounding/support remains strong.
- Stage B, cluster construction: merge semantically aligned role hypotheses into tuple clusters using target/argument/group/label similarity, then compute cluster-level support scores rather than raw majority counts.
- Stage C, intelligent triage: classify each group into `consensus`, `conflict`, or `defective`; extract structured signals including `lexicon_hit`, `victim_microaggression`, `moderator_strong`, `quote_caution`, and `complete_consensus`.
- Consensus fastpath lock: all three triage lanes (`consensus`, `conflict`, `defective`) remain auditor-eligible in the full run. Complete-consensus groups can bypass auditor invocation only when they have a single low-risk aligned cluster and no `lexicon_hit` / `victim_microaggression` / `quote_caution` risk, even if `auditor_on_consensus=true`.
- Stage D, dynamic beacon assembly: inject soft beacons into the auditor prompt according to triage signals. Current beacon family includes `grounding`, `lexicon`, `victim_microaggression`, `moderator_floor`, `quote_caution`, and `consensus`.
- Auditor prompt policy lock: Phase 3 auditor prompting is language-separated, mirroring the Phase 2 prompt-map design. ZH and EN use different few-shot auditor prompts and runtime language filters, so Chinese runs explicitly optimize for `(target, argument, group, hateful)` while English runs optimize for `(target, argument, harmful)` without `group`.
- Stage E, selective regeneration: if one or more roles are missing or defective, rerun only the affected roles with Qwen3-14B Phase 2 prompts; ZH cultural bystander regeneration keeps annotated-lexicon injection enabled and reuses the same focus-lock / soft-target-hint / postprocess contract as mainline Phase 2.
- Stage F, Auditor-as-Refiner: Phase 3 preprocesses rows in chunked language-split buckets and then audits retained requests in batched mode. The current baseline implementation uses a shared Qwen3-14B checkpoint for both auditor and selective regeneration; when the two stages share the same checkpoint, the runtime explicitly releases the regeneration engine before batched auditing to avoid dual-loading the same model. Separate-model configs (for example 35B auditor + 14B regeneration) remain supported as upper-bound controls.
- Auditor reasoning policy: keep think mode enabled for the first attempt, then force direct-JSON retry on parse failure; the auditor must not emit long repair plans, but must preserve compact correction signal in `fused_rationale` and `correction_trace`.
- Lexicon policy lock: `[Lexicon Hit]` is a soft prior that materially increases ZH cultural-bystander interpretive credibility on coded slang, but it is not a hard override; final labeling must still be auditor-led and cross-checked against victim harm, moderator floor, grounding, and quoted/reclaimed usage.
- Batching policy lock: when mixed-language inputs are present, auditor micro-batches are kept language-homogeneous by default to prevent bilingual prompt/schema drift inside the same generation wave.
- Bucketed execution lock: inputs are processed in homogeneous `(language, split)` buckets with preferred order `zh -> en` and `train -> dev -> test`; for current train/test full runs this means `zh_train -> zh_test -> en_train -> en_test`, with checkpointed exports written after interval, chunk, and bucket boundaries.
- Multi-cluster retention lock: supplemental clusters are now conservative by default. Retention is capped to two clusters, requires stronger cluster/member quality, disables lexicon-only singleton retention by default, and rejects singleton clusters that conflict with the soft target hint unless they have stronger explicit harmful support.
- Arbitration contract: the auditor selects one primary cluster (or repairs the closest grounded cluster in place) and returns `final_target`, `final_argument`, `final_group`, `final_label`, `fused_rationale`, `downweighted_roles`, `rejected_roles`, `correction_trace`, and `confidence`; final export may additionally retain at most one conservative supplemental cluster in `final_tuples`.
- Fallback policy: if the auditor still fails strict JSON parsing after retry, Phase 3 falls back to the top deterministic cluster while keeping the raw `auditor_trace` for auditability.
- Deliverables: `group_decisions.jsonl`, `rectified_roles.jsonl`, `final_tuples.jsonl`, and `phase4_trace_pack.jsonl`, with group-level traces preserved for Phase 3.5 merge and Phase 4 distillation. Current implementation exports all four artifacts both as root files and as split / `language x split` views, so the terminal Phase 3 artifact can be consumed directly as `zh_train`, `zh_test`, `en_train`, and `en_test` subsets.

5) Phase 3.5 Aggregator (sample-level multi-target restoration)
- Positioning: run after Phase 3 final decisions and before evaluation/distillation export.
- Input: `DAR/outputs/phase3/latest/candidates/final_tuples.jsonl` (group-level final decisions).
- Core action: regroup by `sample_id` and stitch Phase 1 segmented groups back to sample-level multi-target tuples.
- Merge policy: purely concatenate all clause-level outputs under the same `sample_id` without tuple deduplication, so segmented groups such as `[id]_g1`, `[id]_g2`, ... are all preserved in the restored prediction list; provenance (`group_id`, `decision_id`, `sentence_ids`, confidence/veto counters) is retained per restored tuple.
- Reference-join policy: join back to `DAR/outputs/phase0/cleaned_zh_train.jsonl`, `DAR/outputs/phase0/cleaned_zh_test.jsonl`, `DAR/outputs/phase0/cleaned_en_train.jsonl`, and `DAR/outputs/phase0/cleaned_en_test.jsonl` by `sample_id/id` to recover `original_id`, root `text`, `meta_info`, and optional `ground_truth` for evaluation-ready export.
- Primary restored output: sample-level rows with `restored_multi_target_tuples`, `restored_text`, and per-group trace payloads for downstream inspection/distillation.
- Evaluation-ready export: additionally emit a Phase-0-like sample schema with root fields `id`, `original_id`, `source`, `language`, `split`, `text`, `meta_info`, and `predictions`; ZH predictions use `(target, argument, group, hateful)`, EN predictions use `(target, argument, harmful)`.
- Evaluation normalization lock: exact-match evaluation must normalize HTML entities on both prediction and reference sides before tuple matching; this is now treated as evaluation-level hygiene rather than a prompt burden.
- Output roots: default run root and latest pointer live under `DAR/outputs/phase3.5`, not legacy `DAR/outputs/phase35`.
- Deliverables: `candidates/restored_multi_target.jsonl`, `candidates/phase0_like.jsonl`, plus split-wise exports (`by_split/*`, `phase0_like_by_split/*`, `phase0_like_by_lang_split/*`).
- Runtime artifact: `DAR/experiments/phase35_aggregator.py` with template config `DAR/configs/phase35_template.json`.

6) Phase 4 Distillation (Reverse pseudo-chain, post-core)
- Positioning: finish and lock the core pipeline first, then distill decision-process knowledge into low-cost students.
- Teacher runtime (current baseline): run two language-separated teacher pipelines (ZH stack and EN stack), each as Phase 1 Segmenter -> Phase 2 role generation -> Phase 3 dynamic beacon auditor engine -> Phase 3.5 Aggregator. The default teacher path uses single-backbone Qwen3-14B Phase 2 generation plus the shared-14B Phase 3 auditor / regeneration route; larger-auditor or mixed-backbone teachers are control or upper-bound settings only.
- Principle: do not ask the teacher to regenerate long CoT; reconstruct pseudo-chain from already-produced intermediate artifacts.
- Primary source artifact: Phase 4 dataset building reads Phase 3 `candidates/phase4_trace_pack.jsonl` as the canonical joined teacher trace.
- Reconstruction fallback: if the trace pack or some step payloads are missing, rebuild from `group_decisions + rectified_roles + final_tuples`, using Phase 2 `phase4_role_hypotheses.jsonl` and Phase 1 group / decompose payloads only as lookup or gap-filling sources.
- Join keys and trace identity: sample_id + group_id + role_candidate_id + decision_id (strict consistency checks required).
- Reverse construction order (backward from verdict): Step E Final Arbitration -> Step D Rectification Actions -> Step C Conflict Diagnosis -> Step B Parallel Role Hypotheses -> Step A Decomposition Evidence.
- Student-facing linearization (forward order): A Decompose -> B Role Hypotheses -> C Conflict Diagnosis -> D Rule Rectification -> E Final Decision.
- Template-constrained pseudo-chain text: chain text is generated by deterministic template fill from traces, not free-form generation; every statement must map to a verifiable field.
- Phase 4 pseudo-chain payload policy: Step A decompose uses compact divider fields (`sentence_ids`, `group_text`, `focus_text`, `canonical_target`, regroup keys) and does not include confidence fields.
- Distillation training setting (confirmed): single-pass SFT with data + deterministic pseudo think chain + super prompt, but train two separate students (ZH-only, EN-only) rather than one mixed-language student.
- CoT format policy: chat-style supervision with explicit think block for training samples, while keeping inference output schema strict JSON.
- Quality gates: JSON parseability, message-schema validity, chain-final consistency, clause-grounding, and cross-file key consistency.
- Validation protocol (mandatory): compare `Student-final-only` vs `Student-pseudo-chain` separately in ZH and EN tracks, with special attention to hard cultural cases in each language track.
- Hard-case retention: prioritize groups that satisfy at least one operational trigger: Phase 2 uncertainty_flag=true, Phase 2 escalation_flag=true, Phase 3 overall_conflict_score>=0.67, non-empty Phase 3 high_risk_fields, or cultural-misread-related error tags (for example cultural_metaphor_sarcasm_misread). Keep sample_id/source/split for traceability to original corpora.
- Student models: Qwen3-4B-Instruct (primary deployment student) and Qwen3-8B (capacity-control student), with optional sub-4B edge students. Use QLoRA preferred (LoRA fallback).
- Local-availability fallback: if no 4B checkpoint is available, temporarily use Qwen3.5-9B as bootstrap student control and mark it as resource-substitution in reporting.
- Training schedule: one-pass SFT on single_pass_superprompt dataset.
- Implementation note: concrete build/training plan is documented in DAR/fine_tuning/README_phase4_distillation.md.
- Efficiency and deployment outputs: release deployable 4B checkpoints plus optional sub-4B checkpoints, with latency, throughput, peak VRAM, and cost-quality tradeoff tables.
- Extra comparison tracks for reporting: Gemma-3-12B-it, Mistral-Small-3.1-24B-Instruct, Aya-Expanse-32B as language-capability controls; Qwen3-32B, Llama-3.3-70B-Instruct, and DeepSeek-V3 as upper-bound references.

## Current Reporting Status
- Canonical machine-readable phase comparisons currently live under `DAR/evaluation/testset_results_mainline_20260430/metrics/{phase2,phase3,phase35,phase4}`. These JSON metrics are the authoritative phase snapshot; missing or unfinished auto-generated markdown tables should not be treated as the baseline source of truth.
- Current phase-by-phase English test trend under the TBO protocol: `TAF1` improves from `0.1755` (Phase 2) to `0.2202` (Phase 3) to `0.2203` (Phase 3.5 mainline). The current Phase 4 EN student reaches `0.2368` on the 763-sample official eval subset.
- Current bilingual diagnostic snapshot from `target_level.json` shows soft `T-A-H Tri.` F1 moving from `0.2343` (Phase 2) to `0.2750` (Phase 3) to `0.2763` (Phase 3.5), with the official-eval Phase 4 student at `0.2931` on 2,175 rows (`1412` ZH + `763` EN).
- Paper-facing release packaging is intentionally language-selected rather than forced into one monolithic route. The current stable bundle is `DAR/outputs/release/paper_main_zh_moderator_anchor_en_phase35_restored_20260429`, assembled without new inference, with `ZH = phase2 moderator_anchor` and `EN = phase3.5 phase35_restored`.
- Quick paper-facing metrics from that release bundle:
	- Chinese `moderator_anchor`: `TA Pair Hard/Soft F1 = 0.1073 / 0.3993`, `TA-H Tri. Hard/Soft F1 = 0.0830 / 0.3050`, `Quad Hard/Soft F1 = 0.0665 / 0.2455`.
	- English `phase35_restored`: `TA Pair Hard/Soft F1 = 0.1865 / 0.3513`, `TA-H Tri. Hard/Soft F1 = 0.1623 / 0.3040`.

## Model Strategy
- Family priority lock: Qwen3 / Qwen3.5 are the default mainline model families across phases; legacy Qwen2.5-era settings are archived controls rather than current defaults.
- Mainline Phase 0: preprocessing and metadata tagging only in core path (no mandatory Group completion model for EN/TBO).
- Mainline Phase 1 (Segmenter): Qwen3-14B (primary), with Qwen3.5-9B efficiency variant.
- Mainline Phase 2 (three-perspective generation): all three roles run on Qwen3-14B with language-aware prompts, while Cultural Bystander additionally uses annotated-lexicon retrieval + prompt injection.
- Mainline Phase 3 (Dynamic Beacon + Auditor): shared Qwen3-14B for both auditor and selective regeneration under the current baseline implementation, with language-homogeneous micro-batching and checkpointed bucket execution.
- Mainline Phase 4 (distillation students): language-separated students (ZH student + EN student), with Qwen3-4B-Instruct primary and Qwen3-8B capacity-control per language.
- Baselines (language-capability control): Llama3.1-8B-Instruct, Gemma-2-9B-it, Mistral-Small-3.1-24B-Instruct, Aya-Expanse-8B/32B. Keep at least one strong non-Qwen family control in the main paper comparison grid.
- Upper-bound references: Qwen3.5-35B-A3B-FP8 local inference upper-bound, Qwen3-30B-A3B local reference, DeepSeek-V3 (API/reference only).

## LoRA / QLoRA Policy
- <=14B: LoRA or QLoRA are both viable; prefer QLoRA when VRAM is tight.
- 24B/30B+: QLoRA preferred and usually treated as inference-priority under dual-24GB constraints.
- 35B+: use as inference teacher (4-bit) by default; if role specialization is required, allow adapter-level QLoRA for Cultural Bystander (no routine full-parameter fine-tuning).

## Required Evaluation
- Chinese main table: STATE-style Hard/Soft Macro-F1 for Target, Argument, T-A Pair, T-A-H Tri., and Quad on the Chinese test split.
- English main table: TBO-style Target Span F1, Argument Span F1, Targeted F1, NTAF1, and TAF1 on the English test split.
- Supplemental diagnostics: sample-level harmful accuracy + macro-F1, plus soft gold-anchor scoring for analysis and release-subset tracking.
- Model comparison policy: report ZH and EN leaderboards separately; do not use a merged pooled score as the primary ranking.
- Adversarial Cultural Robustness (hard cultural subset).
- Efficiency: latency, throughput, peak VRAM.

## STATE-Compatible Chinese Reference Baseline
- Primary setting: run language-decoupled pipelines and report Chinese (STATE-compatible) and English (TBO-compatible) results separately.
- Chinese reference protocol: on the Chinese test split, additionally report STATE-compatible metrics for Target, Argument, T-A Pair, T-A-H Tri., and Quad. under both Hard and Soft matching Macro-F1.
- Label alignment rule: map Harmful (this work) to Hateful (STATE protocol) explicitly; publish a Group taxonomy mapping table for Chinese evaluation.
- Fair-comparison tags: explicitly mark language-specific runs (ZH-only pipeline, EN-only pipeline) and avoid claiming strict bilingual transfer gains unless a dedicated bilingual experiment is run.
- Strict comparability control: within each language, keep identical architecture/prompting/decoding across compared models.
- Disclosure lock: if mixed-data or bilingual experiments are reported against STATE-style Chinese references, explicitly disclose extra-data scope and mark them as supplementary settings rather than strict apples-to-apples baseline claims.
- Optional protocol reference: include STATE-style prompt settings (basic prompt, and basic prompt + 2 examples) on the Chinese split as additional baselines.

## Required Ablations
- w/o Multi-Perspective.
- w/o Decision Rectification.
- w/o Cultural Prompting.

## Ablation Priority and Pruning Rule
- Mandatory (main-claim): w/o Multi-Perspective, w/o Decision Rectification, w/o Cultural Prompting.
- Scope lock: this baseline version keeps exactly these three ablations and does not include extra w/o variants.
- Pruning guidance: if budget is tight, keep all three in the main paper and reduce only seed count or model scale.
- Fairness constraint: all ablations must keep identical data splits, seeds, decoding setup, and metrics; only targeted module is changed.


## Efficiency-First Ablation Prompt Template
Use this prompt to run ablations with maximal reuse and minimal reruns.

Role
You are the DAR-Hate ablation runner. Keep scientific fairness while maximizing execution efficiency.

Hard Constraints
- Do not change dataset splits, prompts (except targeted ablation), seeds, or metrics outside ablation scope.
- Keep source-aware and language-aware reporting (ZH and EN separated) for every run.
- Use identical evaluation protocol as the core pipeline.

Execution Order
1) Freeze core artifacts: data manifests, role prompts, decision-engine rules/config, seed list.
2) Run reuse-first ablation:
- w/o Decision Rectification: recompute decisions from cached Phase 2 role outputs with rectification actions disabled.
3) Run strict-rerun ablations:
- w/o Cultural Prompting: rerun Phase 2 to Phase 3 with cultural instructions removed.
- w/o Multi-Perspective: rerun Phase 2 to Phase 3 with single-role setting.
4) Evaluate in two stages:
- Stage A: dev plus hard-case subset for fast screening.
- Stage B: full test with fixed multi-seed runs for final reporting.

Reuse Policy
- Reuse cached intermediate outputs whenever the upstream component is unchanged.
- Rerun only from the earliest changed phase.

Output Requirements
For each ablation, report:
- Main metrics: Harmful Macro-F1, Group Macro-F1 (ZH only), Target/Argument Span-EM and Token-F1.
- Cultural metrics: transfer gap and adversarial cultural robustness.
- Efficiency metrics: latency, throughput, peak VRAM.
- Delta vs core pipeline and significance for key comparisons.

Quality Gates
- Abort and mark invalid if output schema fails, seed mismatch occurs, or non-targeted config drift is detected.
- Keep run logs traceable with run_id, ablation_tag, seed, and artifact pointers.

## Non-negotiable Reporting
- Separate results for ZH and EN; if a pooled cross-lingual number is reported, place it in supplementary and do not use it as main ranking.
- Report significance test for key comparisons.
- Distinguish method gain from backbone language prior gain; keep at least one strong non-Qwen family control in the core comparison table.
- Include a STATE-compatible Chinese table (Hard/Soft over Target, Argument, T-A Pair, T-A-H Tri., Quad.) and clearly mark language-specific setting.

## Priority if time is limited
1) Core pipeline + 3 primary ablations.
2) Language-capability control (at least Qwen vs Llama).
3) Distillation enhancement.


## Historical Note
- Earlier exploratory settings such as Qwen2.5-default paths, 35B-default auditor routes, monolithic pre-bucket Phase 3 processing, and mixed-language joint runs remain in the repo only as archived controls. The main sections above are the authoritative baseline definition as of 2026-04-30.
