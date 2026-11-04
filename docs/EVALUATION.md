# Evaluation protocols

SPAR-Hate is a **structured prediction** method. Scores depend on the tuple schema, equality criteria, aggregation level, target inventory, and dataset split; these choices are part of the experimental protocol.

## English / TBO

`SPAR/evaluation/tbo_eval.py` evaluates the English `phase0_like` output representation, including exact tuple matching and its labelled/target-aware variants. The resulting report includes a TAF1-style exact metric (`taf1`) and harm-label analyses. For example:

```bash
python -m spar_hate evaluate-tbo --input-jsonl SPAR/outputs/en_api/phase3.5/latest/candidates/phase0_like.jsonl
```

`SPAR/evaluation/tbo_harm_paperlike_eval.py` provides a separate harm-label evaluation variant. The historical `paperlike` filename is retained for compatibility, not because its metrics are interchangeable with every paper result.

## Chinese / STATE-style

`SPAR/evaluation/state_style_eval.py` computes STATE-style structured extraction and sample-level quantities, including exact-set accuracy and precision/recall/F1. A Chinese `phase0_like` result file is needed; the input is **not bundled** with this release.

```bash
python -m spar_hate evaluate-state --input-jsonl /path/to/your/chinese_predictions.jsonl
```

## Target-level hard and soft metrics

`SPAR/evaluation/target_level_eval.py` implements target-level diagnostics with both hard (exact) and soft (overlap-based) matching. It also exposes `--scoring-mode` with `standard` and `gold_anchor` choices. These metrics should be reported by their precise names rather than collapsed into a generic 'accuracy' claim.

```bash
python -m spar_hate evaluate-targets --input-jsonl SPAR/outputs/en_api/phase3.5/latest/candidates/phase0_like.jsonl --scoring-mode standard
```

## Reporting checklist

For comparisons with arXiv:2608.22018v4, state which dataset/split, prediction schema, model/backbone, prompting configuration, scoring script, matching protocol, and inference/aggregation settings are used. Never mix strict joint target–argument–label scores with an overlap-based or sample-level metric without explicitly distinguishing them.
