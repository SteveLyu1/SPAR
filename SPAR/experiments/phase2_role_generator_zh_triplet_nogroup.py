from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from SPAR.experiments import phase2_role_generator as _base


_PATCHED = False


def _normalize_prediction_array_by_language(
    parsed: Any,
    language: str,
    focus_text: str = "",
    group_text: str = "",
) -> List[Dict[str, Any]]:
    if not isinstance(parsed, list):
        raise ValueError("Model output must be JSON array")

    normalized: List[Dict[str, Any]] = []
    for idx, item in enumerate(parsed):
        if not isinstance(item, dict):
            raise ValueError(f"Array item at index {idx} is not an object")

        rationale = _base.normalize_string(item.get("rationale"))
        target = _base.normalize_string(item.get("target"))

        if language == "zh":
            hateful = _base.normalize_hate_label(item.get("hateful"))
            argument = (
                _base.normalize_string(item.get("argument"))
                or _base.fallback_argument_for_harmless_item(
                    target=target,
                    focus_text=focus_text,
                    group_text=group_text,
                    language=language,
                )
                if hateful == 0
                else _base.normalize_string(item.get("argument"))
            )

            if rationale is None or target is None or argument is None:
                raise ValueError(f"Missing rationale/target/argument at index {idx}")

            expected_triplet_keys = {"rationale", "target", "argument", "hateful"}
            expected_quadruple_keys = {"rationale", "target", "argument", "group", "hateful"}
            item_keys = set(item.keys())
            if item_keys != expected_triplet_keys and item_keys != expected_quadruple_keys:
                raise ValueError(
                    "ZH triplet schema must contain exact keys "
                    f"{sorted(expected_triplet_keys)} at index {idx}"
                )
            normalized.append(
                {
                    "rationale": rationale,
                    "target": target,
                    "argument": argument,
                    "hateful": hateful,
                }
            )
            continue

        harmful = _base.normalize_hate_label(item.get("harmful"))
        argument = (
            _base.normalize_string(item.get("argument"))
            or _base.fallback_argument_for_harmless_item(
                target=target,
                focus_text=focus_text,
                group_text=group_text,
                language=language,
            )
            if harmful == 0
            else _base.normalize_string(item.get("argument"))
        )
        if rationale is None or target is None or argument is None:
            raise ValueError(f"Missing rationale/target/argument at index {idx}")

        expected_keys = {"rationale", "target", "argument", "harmful"}
        if set(item.keys()) != expected_keys:
            raise ValueError(f"EN schema must contain exact keys {sorted(expected_keys)} at index {idx}")
        normalized.append(
            {
                "rationale": rationale,
                "target": target,
                "argument": argument,
                "harmful": harmful,
            }
        )

    return _base.postprocess_prediction_array_by_language(
        predictions=normalized,
        language=language,
        focus_text=focus_text,
        group_text=group_text,
    )


def _validate_group_predictions_array(items: Any, language: str) -> bool:
    if not isinstance(items, list):
        return False
    for item in items:
        if not isinstance(item, dict):
            return False
        if language == "zh":
            valid_key_sets = [
                {"rationale", "target", "argument", "hateful"},
                {"rationale", "target", "argument", "group", "hateful"},
            ]
            if set(item.keys()) not in valid_key_sets:
                return False
        else:
            required = {"rationale", "target", "argument", "harmful"}
            if set(item.keys()) != required:
                return False
    return True


def _build_heuristic_role_array_prediction(row: Dict[str, Any], role_name: str) -> List[Dict[str, Any]]:
    group_text = _base.normalize_string(row.get("group_text")) or ""
    rationale = f"Heuristic fallback for role={role_name}; text-only estimate."
    if str(row.get("language")) == "zh":
        return [
            {
                "rationale": rationale,
                "target": "IMPLICIT_TARGET",
                "argument": group_text,
                "hateful": None,
            }
        ]
    return [
        {
            "rationale": rationale,
            "target": "IMPLICIT_TARGET",
            "argument": group_text,
            "harmful": None,
        }
    ]


def _build_compat_role_rows(
    group_rows: Sequence[Dict[str, Any]],
    role_names: Sequence[str],
    parse_fail_by_role: Dict[str, int],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    def harmful_to_direction(raw_value: Any) -> str:
        label = _base.normalize_hate_label(raw_value)
        if label == 1:
            return "harmful"
        if label == 0:
            return "non-harmful"
        return "unknown"

    def build_predicted_tuple(item: Dict[str, Any], language: str) -> Dict[str, Any]:
        harmful_key = "hateful" if language == "zh" else "harmful"
        harmful = _base.normalize_hate_label(item.get(harmful_key))
        harmful_direction = harmful_to_direction(item.get(harmful_key))
        if language == "zh":
            group_value = None
        else:
            group_value = None
        return {
            "target": _base.normalize_string(item.get("target")) or "IMPLICIT_TARGET",
            "argument": _base.normalize_string(item.get("argument")),
            "group": group_value,
            "harmful": harmful,
            "harmful_direction": harmful_direction,
        }

    for row in group_rows:
        group_id = str(row.get("group_id") or "")
        sample_id = str(
            row.get("sample_id")
            or _base.parse_sample_id_from_group_id(group_id)
            or row.get("original_id")
            or group_id
        )
        original_id = str(row.get("original_id") or sample_id)
        source = str(row.get("source") or "unknown")
        language = str(row.get("language") or "en")
        split = str(row.get("split") or "train")
        local_group_id = int(row.get("local_group_id") or 1)
        focus_text = str(row.get("focus_text") or f"<focus>{str(row.get('group_text') or '')}</focus>")
        group_text = str(row.get("group_text") or "")
        canonical_target = str(row.get("canonical_target") or "IMPLICIT_TARGET")
        escalation_flag = bool(row.get("escalation_flag"))
        preds_by_role = row.get("phase2_predictions") if isinstance(row.get("phase2_predictions"), dict) else {}

        for role_name in role_names:
            role_key = _base.ROLE_OUTPUT_KEY_MAP.get(role_name, role_name)
            role_items = preds_by_role.get(role_key, [])
            if not role_items:
                continue

            for idx, item in enumerate(role_items, start=1):
                if not isinstance(item, dict):
                    continue
                predicted_tuple = build_predicted_tuple(item=item, language=language)
                rationale = _base.normalize_string(item.get("rationale")) or ""
                role_candidate_id = f"{group_id}:{role_name}:{idx}"
                input_package = {
                    "sample_id": sample_id,
                    "group_id": group_id,
                    "original_id": original_id,
                    "source": source,
                    "split": split,
                    "language": language,
                    "local_group_id": local_group_id,
                    "focus_text": focus_text,
                    "group_text": group_text,
                    "canonical_target": canonical_target,
                    "source_target": canonical_target if canonical_target != "IMPLICIT_TARGET" else None,
                    "source_argument": group_text or None,
                    "source_harmful": None,
                    "segmenter_confidence": 0.0,
                    "escalation_flag": escalation_flag,
                }
                phase4_role_hypothesis = {
                    "task": role_name,
                    "role": role_name,
                    "role_candidate_id": role_candidate_id,
                    "group_id": group_id,
                    "sample_id": sample_id,
                    "tuple": predicted_tuple,
                    "rationale": rationale,
                    "uncertainty_flag": False,
                    "escalation_flag": escalation_flag,
                }
                rows.append(
                    {
                        "sample_id": sample_id,
                        "group_id": group_id,
                        "original_id": original_id,
                        "source": source,
                        "split": split,
                        "role": role_name,
                        "language": language,
                        "local_group_id": local_group_id,
                        "focus_text": focus_text,
                        "group_text": group_text,
                        "canonical_target": canonical_target,
                        "escalation_flag": escalation_flag,
                        "role_candidate_id": role_candidate_id,
                        "prediction": item,
                        "predicted_tuple": predicted_tuple,
                        "rationale": rationale,
                        "uncertainty_flag": False,
                        "fallback_used": False,
                        "model_trace": {
                            "parse_fail": False,
                            "phase2_parse_fail_count_for_role": int(parse_fail_by_role.get(role_name, 0)),
                        },
                        "input_package": input_package,
                        "phase4_role_hypothesis": phase4_role_hypothesis,
                    }
                )
    return rows


def apply_triplet_patches() -> Any:
    global _PATCHED
    if _PATCHED:
        return _base
    _base.normalize_prediction_array_by_language = _normalize_prediction_array_by_language
    _base._validate_group_predictions_array = _validate_group_predictions_array
    _base.build_heuristic_role_array_prediction = _build_heuristic_role_array_prediction
    _base.build_compat_role_rows = _build_compat_role_rows
    _PATCHED = True
    return _base


def get_patched_module() -> Any:
    return apply_triplet_patches()


def run_phase2(args: Optional[Any] = None) -> Dict[str, Any]:
    base = apply_triplet_patches()
    return base.run_phase2(args)


def main() -> None:
    run_phase2()


if __name__ == "__main__":
    main()
