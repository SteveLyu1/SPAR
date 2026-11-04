# Reproducibility guide

This guide documents the **released code paths** for SPAR-Hate, not a claim that the entire arXiv:2608.22018v4 paper can be reproduced from the current repository alone.

## 1. Prerequisites

- Run from the repository root and use a Python environment compatible with the backend actually selected in the JSON config.
- Install `SPAR/requirements.txt`; use `SPAR/requirements.local.txt` only when preparing a compatible local model stack. The latter is not a GPU/CUDA setup guide.
- For the included English API configuration, set the environment variable `OPENAI_API_KEY` using your normal secret management. Never commit keys or raw provider responses containing sensitive information.
- Verify the model identifier, provider access, prompt templates, and numerical sampling settings in each JSON file. The release configs may use models/settings other than those in individual manuscript experiments.
- The bundled TBO English JSONL files provide source inputs for the English settings. The Chinese configuration files require separately prepared Chinese input data.
- `SPAR/resources/annotated_lexicon_placeholder.json` is intentionally empty and cannot reproduce lexicon-dependent Chinese decisions.

## 2. Validate repository references

```bash
python -m spar_hate validate-release
```

This **static** check confirms that the four documented English config files can be parsed and that their local prompt paths exist. It does not validate API access, input artifacts generated during execution, dataset licensing, or the paper's experimental results.

## 3. English stage sequence

From the repository root:

```bash
python -m spar_hate segment --config SPAR/configs/phase1_en_api.json
python -m spar_hate generate-perspectives --config SPAR/configs/phase2_en_api.json
python -m spar_hate audit-and-arbitrate --config SPAR/configs/phase3_en_api.json
python -m spar_hate aggregate-predictions --config SPAR/configs/phase35_en_api.json
```

| Step | Config's expected input | Where output is configured |
| --- | --- | --- |
| Phase 1 | `datasets/tbo_manual_split/cleaned_en_test.jsonl` | `SPAR/outputs/en_api/phase1` |
| Phase 2 | `SPAR/outputs/en_api/phase1/phase1_en_test.jsonl` | `SPAR/outputs/en_api/phase2` |
| Phase 3 | `SPAR/outputs/en_api/phase2/latest/candidates/group_predictions.jsonl` | `SPAR/outputs/en_api/phase3` |
| Phase 3.5 | `SPAR/outputs/en_api/phase3/latest/candidates/final_tuples.jsonl` | `SPAR/outputs/en_api/phase3.5` |

These are **configuration references**, not an assertion that the files are already present. Each stage may write into run-specific subdirectories and a `latest` output directory. Before running the next stage, inspect the previous stage's completion summary and verify that the referenced outputs correspond to the intended same dataset, model, and prompt versions.

The aggregation script exports `candidates/phase0_like.jsonl` beneath its run directory and `latest` directory when enabled. Example downstream English evaluation:

```bash
python -m spar_hate evaluate-tbo --input-jsonl SPAR/outputs/en_api/phase3.5/latest/candidates/phase0_like.jsonl
python -m spar_hate evaluate-targets --input-jsonl SPAR/outputs/en_api/phase3.5/latest/candidates/phase0_like.jsonl
```

**API cost warning:** these stage commands may call hosted models. Do not assume the underlying `--dry-run` option is a complete end-to-end simulation: consult each module's own `--help` and semantics before using it.

## 4. Chinese workflow

The following phase-specific configurations are present:

- `phase1_zh_quadruple_api.json`, `phase2_zh_quadruple_api.json`, `phase3_zh_quadruple_api.json`, `phase35_zh_quadruple_api.json`
- `phase1_zh_triplet_nogroup_api.json`, `phase2_zh_triplet_nogroup_api.json`, `phase3_zh_triplet_nogroup_api.json`

The two Chinese output schemas are different, and one listed variant does not provide an aggregation-stage config. Do not substitute one schema for the other or interpret the empty lexicon placeholder as a complete lexical resource. The released Chinese configs require source data and preparation not supplied here.

## 5. Sources of experimental variation

To compare an execution with the manuscript, record at least: source data version and split; model and API snapshot; prompt file contents; RNG seeds; decoding and retry settings; runtime environment; phase-specific run IDs; postprocessing/aggregation schema; exact-versus-overlap matching; and the evaluation script/version.

Do not treat a successful run of the public English API configs as proof that the paper's reported local-backbone or teacher–student experiments were replicated. The release lacks a packaged student-training stage and a complete set of frozen experimental outputs.

## 6. Provenance and rights

The bundled English TBO split should be cited and reused subject to the source dataset's terms; a public repository alone is not a substitute for checking data redistribution rights. Dance and SynChain are **baseline adaptations**, not original SPAR methodological contributions. See their respective READMEs and original sources for method-specific attribution.
