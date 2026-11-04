# Naming and compatibility map

New readers should navigate the code by **scientific task**, rather than by release status or internal phase number. Existing source filenames and config keys are intentionally retained to protect experiment provenance.

| Public task | New command | Existing implementation |
| --- | --- | --- |
| Focus-unit construction | `python -m spar_hate segment` | `SPAR.experiments.phase1_segmenter` |
| Multi-perspective reasoning | `python -m spar_hate generate-perspectives` | `SPAR.experiments.phase2_role_generator` |
| Evidence-constrained arbitration | `python -m spar_hate audit-and-arbitrate` | `SPAR.experiments.phase3_dynamic_beacon_engine` |
| Sample-level aggregation | `python -m spar_hate aggregate-predictions` | `SPAR.experiments.phase35_aggregator` |
| English structured evaluation | `python -m spar_hate evaluate-tbo` | `SPAR.evaluation.tbo_eval` |
| Chinese structured evaluation | `python -m spar_hate evaluate-state` | `SPAR.evaluation.state_style_eval` |
| Target-level metrics | `python -m spar_hate evaluate-targets` | `SPAR.evaluation.target_level_eval` |
| Static release check | `python -m spar_hate validate-release` | `spar_hate.release_check` |

Flags following the command are forwarded to the original entry point without changing the configured experiment. Original `python -m SPAR.experiments.<module>` calls are still supported.

### Why not mass-rename the source tree?

Experiment configs and historical run artifacts refer to `phase1`, `phase2`, `phase3`, and `phase35` paths; schema-specific variants also distinguish English triplets and Chinese quadruples. Replacing every path for cosmetic consistency would risk invalidating comparisons with prior execution records.

### Release boundaries

- The standalone adaptations under `task_adaptations/` preserve their original names as provenance references.
- The `SPAR/requirements.local.txt` file describes optional local-model dependencies; it is not the same as the base dependencies in `SPAR/requirements.txt`.
- Stage aliases do not implement missing student distillation, supply Chinese data, or perform full-paper replication.

### Future internal renaming

If a later release restructures the main package, update all configuration paths, original execution scripts, evaluation imports, and any immutable artifact hashes together, and run regression tests with recorded fixtures. Until then, the stable public command names are the preferred interface.
