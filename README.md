# SPAR Public Release

This directory is a cleaned public release built from the final code used for the SPAR framework and the final task adaptations retained for release.

## Directory structure

- `SPAR/experiments/`
  Final SPAR main pipeline code for Phase 1, Phase 2, Phase 3, and Phase 3.5.
- `SPAR/prompts/`
  Final prompts corresponding to the retained pipeline variants.
- `SPAR/configs/`
  Cleaned runnable configs for the English mainline and the retained Chinese variants.
- `SPAR/datapreprocess/`
  Existing preprocessing code for producing cleaned phase-0-style JSONL data.
- `SPAR/evaluation/`
  Existing evaluation scripts retained for English and Chinese outputs.
- `SPAR/resources/annotated_lexicon_placeholder.json`
  Placeholder lexicon file kept only so the retained Chinese configs resolve to an in-repo path.
- `datasets/tbo_manual_split/`
  Retained manual TBO split files.
- `task_adaptations/dance_hate_triplet/`
  Final retained Dance adaptation code.
- `task_adaptations/synchain_hate_triplet/`
  Final retained SynChain adaptation code.


## Main pipeline

Default template configs are provided for the English mainline:

- `SPAR/configs/phase1_template.json`
- `SPAR/configs/phase2_template.json`
- `SPAR/configs/phase3_dynamic_beacon_template.json`
- `SPAR/configs/phase35_template.json`

Explicit retained configs are also provided:

- English mainline: `phase1_en_api.json`, `phase2_en_api.json`, `phase3_en_api.json`, `phase35_en_api.json`
- Chinese quadruple: `phase1_zh_quadruple_api.json`, `phase2_zh_quadruple_api.json`, `phase3_zh_quadruple_api.json`, `phase35_zh_quadruple_api.json`
- Chinese triplet-no-group: `phase1_zh_triplet_nogroup_api.json`, `phase2_zh_triplet_nogroup_api.json`, `phase3_zh_triplet_nogroup_api.json`


## Data notes

The retained packaged dataset is the manual TBO split under `datasets/tbo_manual_split/`:

- `cleaned_en_train.jsonl`
- `cleaned_en_test.jsonl`

The preprocessing scripts under `SPAR/datapreprocess/` are retained as-is for reconstructing cleaned inputs from source data.

The bundled data in this release is limited to the retained English TBO manual split. The retained Chinese configs are path-cleaned and kept for completeness, but running the Chinese pipeline still requires separately prepared Chinese cleaned inputs.

## Adaptations

Dance and SynChain are kept as separate retained adaptation directories. Each contains its own existing code and README.

## Environment

Install the retained SPAR dependencies with:

```bash
pip install -r SPAR/requirements.txt
```

For SynChain, install its additional dependency set if needed:

```bash
pip install -r task_adaptations/synchain_hate_triplet/requirements.txt
```
