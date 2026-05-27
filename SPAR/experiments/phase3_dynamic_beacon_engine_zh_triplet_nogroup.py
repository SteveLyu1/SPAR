from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from SPAR.experiments import phase3_dynamic_beacon_engine as _base
from SPAR.experiments import phase2_role_generator_zh_triplet_nogroup as _phase2_triplet


_PATCHED = False
_ORIGINAL_BUILD_ROLE_PAYLOAD = _base.build_role_payload
_ORIGINAL_BUILD_CLUSTERS_PAYLOAD = _base.build_clusters_payload
_ORIGINAL_AUDITOR_OUTPUT_SCHEMA = _base.AuditorEngine._build_output_schema


def _strip_group_from_tuple_payload(tuple_obj: Any) -> Any:
    if not isinstance(tuple_obj, dict):
        return tuple_obj
    cleaned = dict(tuple_obj)
    cleaned.pop("group", None)
    return cleaned


def _is_zh_candidate_sequence(candidates: Sequence[Any]) -> bool:
    for candidate in candidates:
        if getattr(candidate, "language", None) == "zh":
            return True
    return False


def _normalize_final_tuple(tuple_obj: Dict[str, Any], language: str) -> Dict[str, Any]:
    if language != "zh":
        return _base.normalize_final_tuple(tuple_obj, language)
    target = _base.normalize_string(tuple_obj.get("target")) or "IMPLICIT_TARGET"
    argument = _base.normalize_string(tuple_obj.get("argument"))
    harmful = _base.normalize_binary_label(
        tuple_obj.get("hateful") if "hateful" in tuple_obj else tuple_obj.get("harmful")
    )
    direction = _base.harmful_direction_from_label(harmful)
    return {
        "target": target,
        "argument": argument,
        "group": None,
        "harmful": harmful,
        "harmful_direction": direction,
    }


def _zh_group_consistent(group: Optional[str], harmful: Optional[int]) -> bool:
    return True


def _normalize_auditor_payload(
    payload: Dict[str, Any],
    language: str,
    focus_text: str,
    fallback_cluster: Optional[Any],
) -> Dict[str, Any]:
    selected_cluster_id = _base.normalize_string(payload.get("selected_cluster_id"))
    final_target = _base.scrub_placeholder_literal(payload.get("final_target"))
    final_argument = _base.scrub_placeholder_literal(payload.get("final_argument"))
    final_label = _base.normalize_binary_label(payload.get("final_label"))
    fused_rationale = _base.scrub_placeholder_literal(payload.get("fused_rationale")) or ""
    confidence = round(_base.clamp(_base.safe_float(payload.get("confidence"), default=0.75)), 4)
    downweighted_roles = [
        role
        for role in (_base.normalize_auditor_role_name(x) for x in payload.get("downweighted_roles", []))
        if role
    ]
    rejected_roles = [
        role
        for role in (_base.normalize_auditor_role_name(x) for x in payload.get("rejected_roles", []))
        if role
    ]
    correction_trace: List[Dict[str, Any]] = []
    raw_correction_trace = payload.get("correction_trace", [])
    if isinstance(raw_correction_trace, list):
        for item in raw_correction_trace:
            if not isinstance(item, dict):
                continue
            role = _base.normalize_auditor_role_name(item.get("role"))
            action = _base.scrub_placeholder_literal(item.get("action")) or "note"
            reason = _base.scrub_placeholder_literal(item.get("reason"))
            if not reason:
                continue
            correction_trace.append(
                {
                    "role": role,
                    "action": action.lower(),
                    "reason": reason,
                }
            )
    focus_inner = _base.extract_focus_inner_text(focus_text, focus_text)
    if final_argument and final_argument not in focus_inner:
        final_argument = fallback_cluster.canonical_tuple.get("argument") if fallback_cluster else None

    return {
        "selected_cluster_id": selected_cluster_id,
        "final_target": final_target
        or (fallback_cluster.canonical_tuple.get("target") if fallback_cluster else "IMPLICIT_TARGET"),
        "final_argument": final_argument
        or (fallback_cluster.canonical_tuple.get("argument") if fallback_cluster else None),
        "final_group": None,
        "final_label": final_label
        if final_label is not None
        else (fallback_cluster.canonical_tuple.get("harmful") if fallback_cluster else 0),
        "fused_rationale": fused_rationale or _base.build_deterministic_rationale(fallback_cluster, {}, language),
        "downweighted_roles": _base.dedupe_preserve_order(downweighted_roles, case_insensitive=False),
        "rejected_roles": _base.dedupe_preserve_order(rejected_roles, case_insensitive=False),
        "correction_trace": correction_trace,
        "confidence": confidence,
    }


def _select_span_refinement_member(
    selected_cluster: Optional[Any],
    final_tuple: Dict[str, Any],
    language: str,
) -> Optional[Any]:
    if selected_cluster is None:
        return None

    final_target = _base.normalize_string(final_tuple.get("target")) or "IMPLICIT_TARGET"
    final_label = _base.normalize_binary_label(final_tuple.get("harmful"))

    compatible_members: List[Any] = []
    for member in sorted(selected_cluster.members, key=lambda c: (-c.quality_score, c.role, c.role_index)):
        member_tuple = member.tuple_obj
        member_argument = _base.normalize_string(member_tuple.get("argument"))
        if member_argument is None:
            continue

        member_label = _base.normalize_binary_label(member_tuple.get("harmful"))
        if final_label is not None and member_label is not None and member_label != final_label:
            continue

        if final_target != "IMPLICIT_TARGET":
            member_target = _base.normalize_string(member_tuple.get("target")) or "IMPLICIT_TARGET"
            if _base.text_similarity(member_target, final_target, language) < 0.5:
                continue

        compatible_members.append(member)

    if final_target != "IMPLICIT_TARGET":
        exact_target_members: List[Any] = []
        has_expanded_same_target = False
        for member in compatible_members:
            member_target = _base.normalize_string(member.tuple_obj.get("target")) or "IMPLICIT_TARGET"
            member_argument = _base.normalize_string(member.tuple_obj.get("argument")) or ""
            if member_target == "IMPLICIT_TARGET" or _base.text_similarity(member_target, final_target, language) < 0.9:
                continue
            if member_argument == member_target:
                exact_target_members.append(member)
                continue
            if final_target in member_argument and len(member_argument) > len(final_target):
                has_expanded_same_target = True
        if exact_target_members and has_expanded_same_target:
            return sorted(exact_target_members, key=lambda c: (-c.quality_score, c.role, c.role_index))[0]

    if compatible_members:
        return compatible_members[0]
    return None


def _final_tuple_identity_key(tuple_obj: Dict[str, Any], language: str) -> Tuple[str, str, str]:
    harmful = _base.normalize_binary_label(tuple_obj.get("harmful"))
    return (
        _base.normalized_identity_text(tuple_obj.get("target") or "IMPLICIT_TARGET", language),
        _base.normalized_identity_text(tuple_obj.get("argument"), language),
        _base.harmful_direction_from_label(harmful),
    )


def _build_role_payload(
    candidates: Sequence[Any],
    expected_roles: Sequence[str],
    compact_cfg: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    payload = _ORIGINAL_BUILD_ROLE_PAYLOAD(candidates, expected_roles, compact_cfg=compact_cfg)
    if not _is_zh_candidate_sequence(candidates):
        return payload
    for item in payload:
        if not isinstance(item, dict):
            continue
        top_candidate = item.get("top_candidate")
        if isinstance(top_candidate, dict):
            top_candidate["tuple"] = _strip_group_from_tuple_payload(top_candidate.get("tuple"))
        all_candidates = item.get("all_candidates")
        if isinstance(all_candidates, list):
            for cand in all_candidates:
                if isinstance(cand, dict):
                    cand["tuple"] = _strip_group_from_tuple_payload(cand.get("tuple"))
    return payload


def _build_clusters_payload(
    clusters: Sequence[Any],
    compact_cfg: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    payload = _ORIGINAL_BUILD_CLUSTERS_PAYLOAD(clusters, compact_cfg=compact_cfg)
    is_zh = any(getattr(member, "language", None) == "zh" for cluster in clusters for member in getattr(cluster, "members", []))
    if not is_zh:
        return payload
    for item in payload:
        if not isinstance(item, dict):
            continue
        item["canonical_tuple"] = _strip_group_from_tuple_payload(item.get("canonical_tuple"))
        members = item.get("members")
        if isinstance(members, list):
            for member in members:
                if isinstance(member, dict):
                    member["tuple"] = _strip_group_from_tuple_payload(member.get("tuple"))
    return payload


def _auditor_output_schema(self: Any, language: str) -> Dict[str, Any]:
    lang_key = _base.normalize_prompt_language_key(language)
    if lang_key != "zh":
        return _ORIGINAL_AUDITOR_OUTPUT_SCHEMA(self, language)
    return {
        "selected_cluster_id": "c1",
        "final_target": "北京人",
        "final_argument": "说河南人偷井盖的",
        "final_label": 1,
        "fused_rationale": "聚焦片段将地域刻板印象归因到明确目标，保留更贴近文本的候选并下调误读角色。",
        "downweighted_roles": ["moderator"],
        "rejected_roles": [],
        "correction_trace": [
            {
                "role": "moderator",
                "action": "downweighted",
                "reason": "过度依赖表层句式，忽略了焦点片段中的真实归因对象。",
            }
        ],
        "confidence": 0.85,
    }


def _load_phase2_module() -> Any:
    return _phase2_triplet.get_patched_module()


def apply_triplet_patches() -> Any:
    global _PATCHED
    if _PATCHED:
        return _base
    _phase2_triplet.apply_triplet_patches()
    _base.normalize_final_tuple = _normalize_final_tuple
    _base.zh_group_consistent = _zh_group_consistent
    _base.normalize_auditor_payload = _normalize_auditor_payload
    _base.select_span_refinement_member = _select_span_refinement_member
    _base.final_tuple_identity_key = _final_tuple_identity_key
    _base.build_role_payload = _build_role_payload
    _base.build_clusters_payload = _build_clusters_payload
    _base.AuditorEngine._build_output_schema = _auditor_output_schema
    _base.load_phase2_module = _load_phase2_module
    _PATCHED = True
    return _base


def get_patched_module() -> Any:
    return apply_triplet_patches()


def run_phase3(args: Optional[Any] = None) -> Dict[str, Any]:
    base = apply_triplet_patches()
    return base.run_phase3(args)


def main() -> None:
    run_phase3()


if __name__ == "__main__":
    main()
