# SPAR-Hate: Auditor-Guided Multi-Perspective Role Reasoning for Bilingual Hate Speech Parsing

**Research code accompanying [arXiv:2608.22018v4](https://arxiv.org/abs/2608.22018v4)**  
Yifan Lyu · Dianqing Lin · Xinran Li · Jiaqi Qiao · Xiujuan Xu  
Latest referenced manuscript: 14 October 2026 (v4)

## Overview

**SPAR-Hate** studies structured hate speech parsing across English and Chinese, where models must jointly recover the relevant target, the supporting argument, and a target-level label. Multi-target documents, implicit references, and culturally specific expressions make such structure difficult to extract consistently.

Instead of relying on a single model judgment, SPAR-Hate decomposes each input into focused units, elicits evidence-grounded candidate tuples from three perspectives—**Victim**, **Moderator**, and **Cultural Bystander**—and uses auditor-guided arbitration to diagnose disagreements, selectively regenerate candidates, and assemble a sample-level prediction.

The manuscript evaluates SPAR-Hate on **STATE-ToxiCN** and a controlled **TBO** split. It reports improvements in structured extraction, especially under strict joint target–argument–label evaluation. The reported findings are conditional on the study's datasets, backbone models, prompts, and evaluation protocols; this repository does not include every artifact needed to reproduce all paper results.

**Paper:** [arXiv abstract](https://arxiv.org/abs/2608.22018) · [v4 PDF](https://arxiv.org/pdf/2608.22018v4)

## Framework

| Stage | Component | Purpose |
| --- | --- | --- |
| **Phase 1** | Focus-unit construction | Partition an input into localized decision units with explicit source evidence |
| **Phase 2** | Perspective-conditioned candidate generation | Produce structured predictions independently from Victim, Moderator, and Cultural Bystander roles |
| **Phase 3** | Auditor-guided arbitration (Beacon) | Evaluate grounding and consistency, identify conflicts, perform conditional regeneration, and arbitrate candidate tuples |
| **Phase 3.5** | Sample-level aggregation | Reassemble local decisions into the structured predictions required for evaluation |

Phase numbers describe the **released inference workflow**. The paper also discusses structured teacher traces and student-model distillation; that training stage is **not** included in this release. Consult the paper for the full framework illustration and experimental ablations.

## Repository organization

| Path | Contents |
| --- | --- |
| `SPAR/experiments/` | Original phase-specific implementation and API adapter |
| `SPAR/configs/` | Existing English and Chinese stage configurations |
| `SPAR/prompts/` | Language- and role-conditioned prompt templates |
| `SPAR/datapreprocess/` | Source-data preprocessing utilities |
| `SPAR/evaluation/` | STATE-style, TBO, and target-level evaluation scripts |
| `datasets/tbo_manual_split/` | Packaged English TBO train/test JSONL splits |
| `task_adaptations/dance_hate_triplet/` | Dance-style task adaptation, with its own README |
| `task_adaptations/synchain_hate_triplet/` | SynChain-style task adaptation, with its own README |
| `spar_hate/` | Descriptive command-line entry points that preserve the original modules |
| `docs/` | Reproduction prerequisites, evaluation notes, and code naming map |

**No `phase*_template.json` files are included** in the current tree. The four English configuration files used below do exist: `phase1_en_api.json`, `phase2_en_api.json`, `phase3_en_api.json`, and `phase35_en_api.json`.

## Environment

Run commands from the **repository root**, using a Python environment compatible with the selected model backend:

```bash
python -m venv .venv
python -m pip install -r SPAR/requirements.txt
python -m spar_hate --help
python -m spar_hate validate-release
```

The published `SPAR/requirements.txt` covers lightweight components; local model inference may also require `SPAR/requirements.local.txt` and backend-specific CUDA, PyTorch, Transformers, or vLLM compatibility. The English API configurations reference an OpenAI provider profile and expect `OPENAI_API_KEY` in the environment. Running model inference may incur API costs.

## Quick start: English pipeline

The released English configuration files provide stage-wise entry points:

```bash
python -m spar_hate segment --config SPAR/configs/phase1_en_api.json
python -m spar_hate generate-perspectives --config SPAR/configs/phase2_en_api.json
python -m spar_hate audit-and-arbitrate --config SPAR/configs/phase3_en_api.json
python -m spar_hate aggregate-predictions --config SPAR/configs/phase35_en_api.json
```

**Read before executing:** these commands are **not** a guaranteed one-click paper replication. Phase 2 and Phase 3 configurations depend on outputs from preceding phases, including paths under `SPAR/outputs/en_api/`; confirm those inputs exist and correspond to the same run. Commands invoke model inference rather than a no-cost dry run. For configuration-specific requirements, input/output paths, and evaluation commands, see [Reproducibility](docs/REPRODUCIBILITY.md).

## Evaluation

The repository includes evaluation implementations for the two language/task settings:

- **English / TBO:** exact tuple extraction and label-based metrics in `SPAR/evaluation/tbo_eval.py`, as well as an additional harm-label evaluation protocol.
- **Chinese / STATE-style:** structured tuple and sample-level metrics in `SPAR/evaluation/state_style_eval.py`.
- **Target-level diagnostics:** exact (hard) and overlap-aware (soft) matching in `SPAR/evaluation/target_level_eval.py`.

The exact scoring mode, schema, and benchmark subset must match the manuscript before comparing published results. See [Evaluation protocols](docs/EVALUATION.md).

## Data availability and limitations

The repo includes `datasets/tbo_manual_split/cleaned_en_train.jsonl` and `cleaned_en_test.jsonl`. **Chinese cleaned input files are not bundled**. The file `SPAR/resources/annotated_lexicon_placeholder.json` contains an empty `terms` list and is **not** a substantive replacement for the Chinese annotated lexicon used in lexicon-dependent experiments.

Model checkpoints, full experiment traces, intermediate outputs, and the paper's student-distillation training stage are not provided as a complete reproduction package. Access to original datasets and third-party resources remains governed by their respective terms. The included adaptations should not be taken to imply ownership of the original Dance or SynChain methods.

## Citation

If you use this research code or methodology, please cite the manuscript:

```bibtex
@misc{lyu2026sparhate,
  title         = {{SPAR-Hate}: Auditor-Guided Multi-Perspective Role Reasoning for Bilingual Hate Speech Parsing},
  author        = {Yifan Lyu and Dianqing Lin and Xinran Li and Jiaqi Qiao and Xiujuan Xu},
  year          = {2026},
  eprint        = {2608.22018},
  archivePrefix = {arXiv},
  primaryClass  = {cs.AI},
  url           = {https://arxiv.org/abs/2608.22018}
}
```

Machine-readable metadata are available in [`CITATION.cff`](CITATION.cff).

## License and third-party materials

No software license has been assigned to this repository; public visibility alone does not grant permission to reuse or redistribute the code. Data and third-party adaptation components may carry separate rights or conditions. Please review those terms before redistribution.
