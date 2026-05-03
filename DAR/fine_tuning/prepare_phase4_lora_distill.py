from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from build_phase4_superprompt_dataset import (
    ensure_dir,
    load_config,
    normalize_string,
    read_jsonl,
    resolve_path,
    run_phase4_distill,
    write_json,
)


def parse_args() -> argparse.Namespace:
    default_config = str((Path(__file__).resolve().parents[1] / "configs" / "phase4_distill_template.json"))
    parser = argparse.ArgumentParser(description="Prepare Phase4 LoRA/QLoRA distillation assets")
    parser.add_argument("--config", default=default_config, help=f"Config path. Default: {default_config}")
    parser.add_argument("--run-id", default=None, help="Optional phase4 run id")
    parser.add_argument("--output-root", default=None, help="Override output root")

    parser.add_argument("--phase0-samples", default=None, help="Override combined phase0 sample path")
    parser.add_argument("--phase1-candidates", default=None, help="Override phase1 all.jsonl path")
    parser.add_argument("--phase2-candidates", default=None, help="Override phase2 all_roles.jsonl path")
    parser.add_argument("--phase3-trace-pack", default=None, help="Override phase3 trace pack path")
    parser.add_argument("--phase3-group-decisions", default=None, help="Override phase3 group decisions path")
    parser.add_argument("--phase3-rectified-roles", default=None, help="Override phase3 rectified roles path")
    parser.add_argument("--phase3-final-tuples", default=None, help="Override phase3 final tuples path")

    parser.add_argument("--max-samples", type=int, default=None, help="Process at most N samples")
    parser.add_argument("--splits", default=None, help="Comma-separated splits, e.g. train,dev,test")
    parser.add_argument("--hard-case-only", action="store_true", help="Keep hard cases only")
    parser.add_argument("--hard-case-threshold", type=float, default=None, help="Override hard-case threshold")
    parser.add_argument("--strict", action="store_true", help="Fail on quality issues")

    parser.add_argument("--students", default=None, help="Comma-separated student preset names. Default: all presets")
    parser.add_argument("--tuning-modes", default="lora,qlora", help="Comma-separated tuning modes")
    parser.add_argument("--model-name-or-path", default=None, help="Override model path when exactly one student is selected")
    parser.add_argument("--template", default=None, help="Override chat template")
    parser.add_argument("--execute-train", action="store_true", help="Actually launch LLaMA-Factory training jobs")
    return parser.parse_args()


def yaml_scalar(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    if isinstance(value, (int, float)):
        return str(value)
    return json.dumps(str(value), ensure_ascii=False)


def write_simple_yaml(path: Path, data: Sequence[Tuple[str, Any]]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        for key, value in data:
            f.write(f"{key}: {yaml_scalar(value)}\n")


def detect_runtime_python(cfg: Dict[str, Any]) -> str:
    paths_cfg = cfg.get("paths", {}) if isinstance(cfg.get("paths"), dict) else {}
    explicit = normalize_string(paths_cfg.get("runtime_python"))
    candidates = [
        explicit,
        normalize_string(os.environ.get("DAR_RUNTIME_PYTHON")),
        str(Path.home() / ".conda" / "envs" / "DAR" / "bin" / "python"),
        sys.executable,
    ]
    for candidate in candidates:
        if candidate is None:
            continue
        candidate_path = Path(candidate).expanduser()
        if candidate_path.exists():
            return str(candidate_path)
    return sys.executable


def count_rows(path: Path) -> int:
    if not path.exists():
        return 0
    return len(read_jsonl(path))


def dataset_name_for_variant(dataset_name_prefix: str, variant_name: str) -> str:
    return f"{dataset_name_prefix}__{variant_name}"


def dataset_name_for_variant_language(dataset_name_prefix: str, variant_name: str, language: str) -> str:
    return f"{dataset_name_prefix}__{variant_name}__{language}"


def build_dataset_info(
    dataset_name_prefix: str,
    variant_names: Sequence[str],
    variant_languages: Optional[Dict[str, Sequence[str]]] = None,
) -> Dict[str, Any]:
    common = {
        "formatting": "sharegpt",
        "columns": {
            "messages": "messages",
        },
        "tags": {
            "role_tag": "role",
            "content_tag": "content",
            "user_tag": "user",
            "assistant_tag": "assistant",
            "system_tag": "system",
        },
    }

    dataset_info = {
        f"{dataset_name_prefix}_train": {
            "file_name": "by_split/train.jsonl",
            **common,
        },
        f"{dataset_name_prefix}_dev": {
            "file_name": "by_split/dev.jsonl",
            **common,
        },
        f"{dataset_name_prefix}_test": {
            "file_name": "by_split/test.jsonl",
            **common,
        },
    }

    for variant_name in variant_names:
        variant_prefix = dataset_name_for_variant(dataset_name_prefix, variant_name)
        dataset_info[f"{variant_prefix}_train"] = {
            "file_name": f"variants/{variant_name}/by_split/train.jsonl",
            **common,
        }
        dataset_info[f"{variant_prefix}_dev"] = {
            "file_name": f"variants/{variant_name}/by_split/dev.jsonl",
            **common,
        }
        dataset_info[f"{variant_prefix}_test"] = {
            "file_name": f"variants/{variant_name}/by_split/test.jsonl",
            **common,
        }
        for language in sorted({str(item) for item in (variant_languages or {}).get(variant_name, []) if str(item).strip()}):
            language_prefix = dataset_name_for_variant_language(dataset_name_prefix, variant_name, language)
            dataset_info[f"{language_prefix}_train"] = {
                "file_name": f"variants/{variant_name}/by_lang_split/{language}_train.jsonl",
                **common,
            }
            dataset_info[f"{language_prefix}_dev"] = {
                "file_name": f"variants/{variant_name}/by_lang_split/{language}_dev.jsonl",
                **common,
            }
            dataset_info[f"{language_prefix}_test"] = {
                "file_name": f"variants/{variant_name}/by_lang_split/{language}_test.jsonl",
                **common,
            }

    return dataset_info


def load_prompt_bundle(cfg: Dict[str, Any], workspace_root: Path, config_path: Path) -> Dict[str, Any]:
    prompts_cfg = cfg.get("prompts", {})
    phase4_cfg = cfg.get("phase4", {})
    config_dir = config_path.parent

    def resolve_prompt_node(node: Any) -> Any:
        if isinstance(node, dict):
            return {str(key): resolve_prompt_node(value) for key, value in node.items()}
        if isinstance(node, str) and node.strip():
            prompt_path = resolve_path(workspace_root, config_dir, node)
            payload: Dict[str, Any] = {"path": str(prompt_path)}
            if prompt_path.exists():
                payload["content"] = prompt_path.read_text(encoding="utf-8")
            else:
                payload["missing"] = True
            return payload
        return node

    bundle: Dict[str, Any] = {
        "prompts": resolve_prompt_node(prompts_cfg),
    }

    phase4_profiles = phase4_cfg.get("prompt_profiles", {})
    if isinstance(phase4_profiles, dict):
        phase4_bundle: Dict[str, Any] = {}
        for name, profile in phase4_profiles.items():
            if not isinstance(profile, dict):
                continue
            entry: Dict[str, Any] = {}
            for key in ["match_languages", "match_sources", "package_label", "schema_label", "step_titles", "assistant_chain_style"]:
                if key in profile:
                    entry[key] = profile.get(key)
            for key in ["system_prompt_path", "user_instruction_path"]:
                raw_path = normalize_string(profile.get(key))
                if raw_path is None:
                    continue
                prompt_path = resolve_path(workspace_root, config_dir, raw_path)
                payload: Dict[str, Any] = {"path": str(prompt_path)}
                if prompt_path.exists():
                    payload["content"] = prompt_path.read_text(encoding="utf-8")
                else:
                    payload["missing"] = True
                entry[key] = payload
            phase4_bundle[str(name)] = entry
        bundle["phase4_prompt_profiles"] = phase4_bundle

    return bundle


def parse_csv_list(value: Optional[str]) -> List[str]:
    if value is None:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def count_split_rows(base_dir: Path) -> Dict[str, int]:
    return {
        "train": count_rows(base_dir / "train.jsonl"),
        "dev": count_rows(base_dir / "dev.jsonl"),
        "test": count_rows(base_dir / "test.jsonl"),
    }


def detect_languages_from_variant_dir(variant_dir: Path) -> List[str]:
    by_lang_dir = variant_dir / "by_lang_split"
    if not by_lang_dir.exists():
        return []
    languages = set()
    for split in ["train", "dev", "test"]:
        suffix = f"_{split}.jsonl"
        for path in by_lang_dir.glob(f"*{suffix}"):
            name = path.name
            if not name.endswith(suffix):
                continue
            language = name[: -len(suffix)].strip()
            if language:
                languages.add(language)
    return sorted(languages)


def count_language_split_rows(variant_dir: Path, languages: Sequence[str]) -> Dict[str, Dict[str, int]]:
    by_lang_dir = variant_dir / "by_lang_split"
    out: Dict[str, Dict[str, int]] = {}
    for language in languages:
        out[str(language)] = {
            "train": count_rows(by_lang_dir / f"{language}_train.jsonl"),
            "dev": count_rows(by_lang_dir / f"{language}_dev.jsonl"),
            "test": count_rows(by_lang_dir / f"{language}_test.jsonl"),
        }
    return out


def build_dataset_args(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        config=args.config,
        run_id=args.run_id,
        output_root=args.output_root,
        phase0_samples=args.phase0_samples,
        phase1_candidates=args.phase1_candidates,
        phase2_candidates=args.phase2_candidates,
        phase3_trace_pack=args.phase3_trace_pack,
        phase3_group_decisions=args.phase3_group_decisions,
        phase3_rectified_roles=args.phase3_rectified_roles,
        phase3_final_tuples=args.phase3_final_tuples,
        max_samples=args.max_samples,
        splits=args.splits,
        hard_case_only=args.hard_case_only,
        hard_case_threshold=args.hard_case_threshold,
        strict=args.strict,
        dry_run=False,
    )


def resolve_students(
    args: argparse.Namespace,
    training_cfg: Dict[str, Any],
) -> List[Tuple[str, Dict[str, Any]]]:
    presets = training_cfg.get("student_presets", {})
    if not isinstance(presets, dict) or not presets:
        raise ValueError("training.student_presets must be a non-empty mapping in phase4 config.")

    requested = parse_csv_list(args.students)
    if not requested:
        requested = list(presets.keys())

    resolved: List[Tuple[str, Dict[str, Any]]] = []
    for name in requested:
        preset = presets.get(name)
        if not isinstance(preset, dict):
            raise KeyError(f"Unknown student preset: {name}")
        resolved.append((name, dict(preset)))

    if args.model_name_or_path is not None:
        if len(resolved) != 1:
            raise ValueError("--model-name-or-path can only be used when exactly one student is selected.")
        resolved[0][1]["model_name_or_path"] = args.model_name_or_path

    if args.template is not None:
        for _, preset in resolved:
            preset["template"] = args.template

    return resolved


def load_curriculum(training_cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    raw = training_cfg.get("curriculum")
    if raw is None:
        return [
            {
                "stage_name": "stage1_clean",
                "dataset_variant": "stage1_clean",
                "num_train_epochs": 1.0,
            },
            {
                "stage_name": "stage2_silver",
                "dataset_variant": "stage2_silver",
                "num_train_epochs": 2.0,
                "continue_from_previous_adapter": True,
                "create_new_adapter": False,
            },
        ]
    if not isinstance(raw, list) or not raw:
        raise ValueError("training.curriculum must be a non-empty list when provided.")

    curriculum: List[Dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        stage_name = normalize_string(item.get("stage_name")) or normalize_string(item.get("name"))
        dataset_variant = normalize_string(item.get("dataset_variant"))
        if stage_name is None or dataset_variant is None:
            raise ValueError("Each training.curriculum stage must define stage_name and dataset_variant.")
        stage = dict(item)
        stage["stage_name"] = stage_name
        stage["dataset_variant"] = dataset_variant
        curriculum.append(stage)

    if not curriculum:
        raise ValueError("training.curriculum resolved to zero valid stages.")
    return curriculum


def resolve_training_value(training_cfg: Dict[str, Any], stage_cfg: Dict[str, Any], key: str, default: Any) -> Any:
    if key in stage_cfg:
        return stage_cfg[key]
    return training_cfg.get(key, default)


def build_train_yaml_items(
    dataset_dir: Path,
    dataset_name: str,
    training_cfg: Dict[str, Any],
    stage_cfg: Dict[str, Any],
    preset_name: str,
    preset_cfg: Dict[str, Any],
    tuning_mode: str,
    output_dir: Path,
    has_dev: bool,
    resume_adapter_path: Optional[Path] = None,
) -> List[Tuple[str, Any]]:
    model_name_or_path = normalize_string(preset_cfg.get("model_name_or_path"))
    if model_name_or_path is None:
        raise ValueError(f"Student preset {preset_name} is missing model_name_or_path")

    template = normalize_string(preset_cfg.get("template")) or normalize_string(training_cfg.get("template")) or "qwen3_nothink"
    mode_cfg = training_cfg.get(tuning_mode, {})
    if not isinstance(mode_cfg, dict):
        raise KeyError(f"Missing training config for tuning mode: {tuning_mode}")

    items: List[Tuple[str, Any]] = [
        ("model_name_or_path", model_name_or_path),
        ("trust_remote_code", True),
        ("stage", "sft"),
        ("do_train", True),
        ("finetuning_type", mode_cfg.get("finetuning_type", "lora")),
        ("lora_rank", mode_cfg.get("lora_rank", 32)),
        ("lora_alpha", mode_cfg.get("lora_alpha", mode_cfg.get("lora_rank", 32))),
        ("lora_dropout", mode_cfg.get("lora_dropout", 0.0)),
        ("lora_target", mode_cfg.get("lora_target", "all")),
        ("dataset_dir", str(dataset_dir)),
        ("dataset", f"{dataset_name}_train"),
        ("template", template),
        ("cutoff_len", resolve_training_value(training_cfg, stage_cfg, "cutoff_len", 4096)),
        ("preprocessing_num_workers", resolve_training_value(training_cfg, stage_cfg, "preprocessing_num_workers", 8)),
        ("dataloader_num_workers", resolve_training_value(training_cfg, stage_cfg, "dataloader_num_workers", 4)),
        ("output_dir", str(output_dir)),
        ("logging_steps", resolve_training_value(training_cfg, stage_cfg, "logging_steps", 10)),
        ("save_steps", resolve_training_value(training_cfg, stage_cfg, "save_steps", 100)),
        ("plot_loss", resolve_training_value(training_cfg, stage_cfg, "plot_loss", True)),
        ("overwrite_output_dir", resolve_training_value(training_cfg, stage_cfg, "overwrite_output_dir", True)),
        ("save_only_model", resolve_training_value(training_cfg, stage_cfg, "save_only_model", False)),
        ("report_to", resolve_training_value(training_cfg, stage_cfg, "report_to", "none")),
        ("per_device_train_batch_size", resolve_training_value(training_cfg, stage_cfg, "per_device_train_batch_size", 1)),
        ("gradient_accumulation_steps", resolve_training_value(training_cfg, stage_cfg, "gradient_accumulation_steps", 8)),
        ("learning_rate", resolve_training_value(training_cfg, stage_cfg, "learning_rate", 2e-4)),
        ("num_train_epochs", resolve_training_value(training_cfg, stage_cfg, "num_train_epochs", 3.0)),
        ("lr_scheduler_type", resolve_training_value(training_cfg, stage_cfg, "lr_scheduler_type", "cosine")),
        ("warmup_ratio", resolve_training_value(training_cfg, stage_cfg, "warmup_ratio", 0.1)),
        ("bf16", resolve_training_value(training_cfg, stage_cfg, "bf16", True)),
        ("ddp_timeout", resolve_training_value(training_cfg, stage_cfg, "ddp_timeout", 180000000)),
    ]

    if tuning_mode == "qlora":
        items.extend(
            [
                ("quantization_bit", mode_cfg.get("quantization_bit", 4)),
                ("quantization_method", mode_cfg.get("quantization_method", "bnb")),
                ("double_quantization", mode_cfg.get("double_quantization", False)),
            ]
        )

    if has_dev:
        items.extend(
            [
                ("eval_dataset", f"{dataset_name}_dev"),
                ("per_device_eval_batch_size", resolve_training_value(training_cfg, stage_cfg, "per_device_eval_batch_size", 1)),
                ("eval_strategy", resolve_training_value(training_cfg, stage_cfg, "eval_strategy", "steps")),
                (
                    "eval_steps",
                    resolve_training_value(
                        training_cfg,
                        stage_cfg,
                        "eval_steps",
                        resolve_training_value(training_cfg, stage_cfg, "save_steps", 100),
                    ),
                ),
            ]
        )

    if resume_adapter_path is not None:
        items.extend(
            [
                ("adapter_name_or_path", str(resume_adapter_path)),
                ("create_new_adapter", bool(stage_cfg.get("create_new_adapter", False))),
            ]
        )

    return items


def write_train_script(path: Path, llamafactory_root: Path, yaml_path: Path, runtime_python: str) -> None:
    ensure_dir(path.parent)
    script = (
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        f"cd {json.dumps(str(llamafactory_root), ensure_ascii=False)}\n"
        f"{json.dumps(str(runtime_python), ensure_ascii=False)} -m llamafactory.cli train {json.dumps(str(yaml_path), ensure_ascii=False)}\n"
    )
    path.write_text(script, encoding="utf-8")
    path.chmod(0o755)


def main() -> None:
    args = parse_args()
    workspace_root = Path(__file__).resolve().parents[2]
    config_path = Path(args.config).resolve()
    cfg = load_config(config_path)

    dataset_summary = run_phase4_distill(build_dataset_args(args))
    run_root = Path(dataset_summary["output_root"]).resolve()
    dataset_dir = run_root / "datasets"
    by_split_dir = dataset_dir / "by_split"

    training_cfg = cfg.get("training", {})
    if not isinstance(training_cfg, dict):
        raise ValueError("Missing training config in phase4 config.")

    dataset_name_prefix = normalize_string(training_cfg.get("dataset_name_prefix")) or "dar_hate_phase4"
    phase4_summary_cfg = dataset_summary.get("phase4", {}) if isinstance(dataset_summary.get("phase4"), dict) else {}
    variant_names = phase4_summary_cfg.get("dataset_variants") if isinstance(phase4_summary_cfg.get("dataset_variants"), list) else []
    variant_names = [str(name) for name in variant_names if str(name).strip()]
    primary_variant_name = normalize_string(phase4_summary_cfg.get("primary_variant_name")) or "stage2_silver"

    variant_languages = {
        variant_name: detect_languages_from_variant_dir(dataset_dir / "variants" / variant_name)
        for variant_name in variant_names
    }
    dataset_info = build_dataset_info(dataset_name_prefix, variant_names=variant_names, variant_languages=variant_languages)
    write_json(dataset_dir / "dataset_info.json", dataset_info)

    prompt_bundle = load_prompt_bundle(cfg, workspace_root=workspace_root, config_path=config_path)
    write_json(dataset_dir / "prompt_bundle.json", prompt_bundle)

    split_counts = count_split_rows(by_split_dir)
    variant_split_counts = {
        variant_name: count_split_rows(dataset_dir / "variants" / variant_name / "by_split")
        for variant_name in variant_names
    }
    variant_language_split_counts = {
        variant_name: count_language_split_rows(dataset_dir / "variants" / variant_name, variant_languages.get(variant_name, []))
        for variant_name in variant_names
    }

    students = resolve_students(args, training_cfg=training_cfg)
    tuning_modes = parse_csv_list(args.tuning_modes) or ["lora", "qlora"]
    curriculum = load_curriculum(training_cfg)
    runtime_python = detect_runtime_python(cfg)

    llamafactory_root_raw = normalize_string((cfg.get("paths", {}) or {}).get("llamafactory_root")) or "LLaMA-Factory"
    llamafactory_root = resolve_path(workspace_root, config_path.parent, llamafactory_root_raw)

    generated_runs: List[Dict[str, Any]] = []
    for preset_name, preset_cfg in students:
        for tuning_mode in tuning_modes:
            previous_stage_output_dir: Optional[Path] = None
            previous_stage_name: Optional[str] = None
            for stage_cfg in curriculum:
                stage_name = normalize_string(stage_cfg.get("stage_name")) or "stage"
                dataset_variant = normalize_string(stage_cfg.get("dataset_variant"))
                if dataset_variant is None:
                    raise ValueError(f"Curriculum stage {stage_name} is missing dataset_variant.")
                if dataset_variant not in variant_names:
                    raise ValueError(f"Curriculum stage {stage_name} references unknown dataset variant: {dataset_variant}")

                language_filter = (
                    normalize_string(preset_cfg.get("language_filter"))
                    or normalize_string(preset_cfg.get("language"))
                    or normalize_string(preset_cfg.get("dataset_language"))
                )
                if language_filter is not None:
                    stage_dataset_name = dataset_name_for_variant_language(dataset_name_prefix, dataset_variant, language_filter)
                    stage_counts = (variant_language_split_counts.get(dataset_variant) or {}).get(language_filter)
                    if stage_counts is None:
                        available_languages = variant_languages.get(dataset_variant, [])
                        raise ValueError(
                            f"Curriculum stage {stage_name} requested language={language_filter} for variant {dataset_variant}, "
                            f"but available languages are {available_languages}."
                        )
                else:
                    stage_dataset_name = dataset_name_for_variant(dataset_name_prefix, dataset_variant)
                    stage_counts = variant_split_counts.get(dataset_variant) or count_split_rows(
                        dataset_dir / "variants" / dataset_variant / "by_split"
                    )
                if stage_counts["train"] <= 0:
                    raise ValueError(f"Curriculum stage {stage_name} has zero train rows in variant {dataset_variant}.")

                continue_from_previous = bool(stage_cfg.get("continue_from_previous_adapter"))
                resume_adapter_path = previous_stage_output_dir if continue_from_previous else None
                if continue_from_previous and resume_adapter_path is None:
                    raise ValueError(f"Curriculum stage {stage_name} is configured to continue, but no previous stage exists.")

                output_dir = run_root / "training" / "runs" / f"{preset_name}_{tuning_mode}" / stage_name
                yaml_path = run_root / "training" / "configs" / f"{preset_name}_{tuning_mode}__{stage_name}.yaml"
                train_script_path = run_root / "training" / "scripts" / f"run_{preset_name}_{tuning_mode}__{stage_name}.sh"

                yaml_items = build_train_yaml_items(
                    dataset_dir=dataset_dir,
                    dataset_name=stage_dataset_name,
                    training_cfg=training_cfg,
                    stage_cfg=stage_cfg,
                    preset_name=preset_name,
                    preset_cfg=preset_cfg,
                    tuning_mode=tuning_mode,
                    output_dir=output_dir,
                    has_dev=stage_counts["dev"] > 0,
                    resume_adapter_path=resume_adapter_path,
                )
                write_simple_yaml(yaml_path, yaml_items)
                write_train_script(
                    train_script_path,
                    llamafactory_root=llamafactory_root,
                    yaml_path=yaml_path,
                    runtime_python=runtime_python,
                )

                generated_runs.append(
                    {
                        "student": preset_name,
                        "tuning_mode": tuning_mode,
                        "stage_name": stage_name,
                        "dataset_variant": dataset_variant,
                        "language_filter": language_filter,
                        "dataset_name": stage_dataset_name,
                        "split_counts": stage_counts,
                        "model_name_or_path": preset_cfg.get("model_name_or_path"),
                        "yaml_path": str(yaml_path),
                        "train_script": str(train_script_path),
                        "runtime_python": runtime_python,
                        "output_dir": str(output_dir),
                        "continue_from_previous_adapter": continue_from_previous,
                        "resume_from_stage": previous_stage_name if continue_from_previous else None,
                        "adapter_name_or_path": str(resume_adapter_path) if resume_adapter_path is not None else None,
                    }
                )

                previous_stage_output_dir = output_dir
                previous_stage_name = stage_name

    manifest = {
        "config_path": str(config_path),
        "phase4_run_root": str(run_root),
        "dataset_summary": dataset_summary,
        "dataset_dir": str(dataset_dir),
        "dataset_info_path": str(dataset_dir / "dataset_info.json"),
        "prompt_bundle_path": str(dataset_dir / "prompt_bundle.json"),
        "runtime_python": runtime_python,
        "primary_variant_name": primary_variant_name,
        "split_counts": split_counts,
        "variant_split_counts": variant_split_counts,
        "variant_languages": variant_languages,
        "variant_language_split_counts": variant_language_split_counts,
        "curriculum": curriculum,
        "training_jobs": generated_runs,
        "execute_train": bool(args.execute_train),
    }
    write_json(run_root / "training" / "manifest.json", manifest)

    if args.execute_train:
        for job in generated_runs:
            subprocess.run(
                [runtime_python, "-m", "llamafactory.cli", "train", job["yaml_path"]],
                cwd=str(llamafactory_root),
                check=True,
            )

    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
