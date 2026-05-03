from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import yaml

from build_phase4_superprompt_dataset import ensure_dir, normalize_string, read_jsonl, write_json, write_jsonl
from prepare_phase4_lora_distill import build_dataset_info


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run phase4 pilot training, generation evaluation, and optionally launch or resume full training."
    )
    parser.add_argument("--manifest", required=True, help="Path to phase4 training manifest.json")
    parser.add_argument("--student", default=None, help="Student preset name from manifest")
    parser.add_argument("--tuning-mode", default="lora", help="Tuning mode from manifest")
    parser.add_argument("--runtime-python", default=None, help="Python executable used for LLaMA-Factory and eval")
    parser.add_argument("--cuda-visible-devices", default="1", help="CUDA_VISIBLE_DEVICES for child jobs")
    parser.add_argument("--llamafactory-root", default=None, help="Optional override for LLaMA-Factory repo root")
    parser.add_argument(
        "--use-modelscope",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Resolve remote models through ModelScope instead of Hugging Face.",
    )

    parser.add_argument("--pilot-root-name", default=None, help="Override orchestrator subdirectory name")
    parser.add_argument("--pilot-stage1-train-per-lang", type=int, default=64)
    parser.add_argument("--pilot-stage2-train-per-lang", type=int, default=128)
    parser.add_argument("--pilot-test-per-lang", type=int, default=24)
    parser.add_argument("--pilot-stage1-epochs", type=float, default=1.0)
    parser.add_argument("--pilot-stage2-epochs", type=float, default=2.0)
    parser.add_argument("--pilot-logging-steps", type=int, default=5)
    parser.add_argument("--pilot-save-steps", type=int, default=20)
    parser.add_argument("--pilot-eval-max-samples", type=int, default=48)
    parser.add_argument("--pilot-eval-max-new-tokens", type=int, default=1024)
    parser.add_argument("--pilot-eval-temperature", type=float, default=0.0)

    parser.add_argument("--skip-pilot", action="store_true", help="Skip pilot training and evaluation")
    parser.add_argument("--run-full", action="store_true", help="Run full training after pilot or in full-only mode")
    parser.add_argument(
        "--background-full",
        action="store_true",
        help="Detach the full run into background after pilot completes",
    )
    parser.add_argument("--full-only", action="store_true", help="Run only the full training workflow")
    parser.add_argument("--retry-attempts", type=int, default=3, help="Retries per training job")
    parser.add_argument("--retry-sleep-seconds", type=int, default=30, help="Sleep between retries")
    parser.add_argument("--full-logging-steps", type=int, default=10)
    parser.add_argument("--full-save-steps", type=int, default=25)
    return parser.parse_args()


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json_atomic(path: Path, payload: Dict[str, Any]) -> None:
    ensure_dir(path.parent)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp_path.replace(path)


def resolve_runtime_python(args: argparse.Namespace, manifest: Dict[str, Any]) -> str:
    candidates = [
        normalize_string(args.runtime_python),
        normalize_string(manifest.get("runtime_python")),
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


def resolve_llamafactory_root(args: argparse.Namespace, repo_root: Path) -> Path:
    if args.llamafactory_root:
        return Path(args.llamafactory_root).expanduser().resolve()
    return (repo_root / "LLaMA-Factory").resolve()


def stable_seed(variant_name: str, split: str) -> int:
    text = f"{variant_name}:{split}"
    return 3407 + sum((index + 1) * ord(ch) for index, ch in enumerate(text))


def resolve_local_model_path(model_name_or_path: str, use_modelscope: bool) -> str:
    candidate_path = Path(model_name_or_path).expanduser()
    if candidate_path.exists():
        return str(candidate_path.resolve())

    if use_modelscope:
        modelscope_path = Path.home() / ".cache" / "modelscope" / "hub" / "models" / Path(model_name_or_path)
        if modelscope_path.exists():
            return str(modelscope_path.resolve())

    return model_name_or_path


def pick_jobs(manifest: Dict[str, Any], student: str, tuning_mode: str) -> List[Dict[str, Any]]:
    jobs = [
        job
        for job in manifest.get("training_jobs", [])
        if normalize_string(job.get("student")) == student and normalize_string(job.get("tuning_mode")) == tuning_mode
    ]
    if not jobs:
        raise ValueError(f"No jobs found for student={student}, tuning_mode={tuning_mode}")
    return sorted(jobs, key=lambda item: str(item.get("stage_name")))


def resolve_student_name(manifest: Dict[str, Any], requested_student: Optional[str]) -> str:
    student = normalize_string(requested_student)
    if student is not None:
        return student
    candidates = sorted(
        {
            normalize_string(job.get("student"))
            for job in manifest.get("training_jobs", [])
            if normalize_string(job.get("student")) is not None
        }
    )
    if len(candidates) == 1:
        return candidates[0]
    raise ValueError(f"Manifest contains multiple students {candidates}; please pass --student explicitly.")


def job_by_stage(jobs: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    mapping: Dict[str, Dict[str, Any]] = {}
    for job in jobs:
        stage_name = normalize_string(job.get("stage_name"))
        if stage_name is None:
            continue
        mapping[stage_name] = job
    return mapping


def resolve_stage_order(manifest: Dict[str, Any], jobs: Sequence[Dict[str, Any]]) -> List[str]:
    requested: List[str] = []
    curriculum = manifest.get("curriculum")
    if isinstance(curriculum, list):
        for item in curriculum:
            if not isinstance(item, dict):
                continue
            stage_name = normalize_string(item.get("stage_name")) or normalize_string(item.get("name"))
            if stage_name is not None:
                requested.append(stage_name)

    available = {
        normalize_string(job.get("stage_name"))
        for job in jobs
        if normalize_string(job.get("stage_name")) is not None
    }
    ordered = [stage_name for stage_name in requested if stage_name in available]
    if ordered:
        return ordered

    return sorted(str(stage_name) for stage_name in available)


def infer_dataset_name_prefix(stage_job: Dict[str, Any]) -> str:
    dataset_name = normalize_string(stage_job.get("dataset_name"))
    if dataset_name is None:
        raise ValueError("Selected job is missing dataset_name.")
    if "__" in dataset_name:
        return dataset_name.split("__", 1)[0]
    return dataset_name.rsplit("_", 1)[0]


def group_rows_by_language(rows: Sequence[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        lang = normalize_string(row.get("language")) or "unknown"
        grouped.setdefault(lang, []).append(row)
    return grouped


def sample_rows_by_language(rows: Sequence[Dict[str, Any]], per_lang: int, seed: int) -> List[Dict[str, Any]]:
    if per_lang <= 0:
        return list(rows)

    rng = random.Random(seed)
    selected: List[Dict[str, Any]] = []
    for language, language_rows in sorted(group_rows_by_language(rows).items()):
        if len(language_rows) <= per_lang:
            picked = list(language_rows)
        else:
            picked = rng.sample(language_rows, per_lang)
        selected.extend(picked)
    rng.shuffle(selected)
    return selected


def detect_variant_path(dataset_dir: Path, variant_name: str, split: str) -> Path:
    return dataset_dir / "variants" / variant_name / "by_split" / f"{split}.jsonl"


def detect_variant_language_path(dataset_dir: Path, variant_name: str, language: str, split: str) -> Path:
    return dataset_dir / "variants" / variant_name / "by_lang_split" / f"{language}_{split}.jsonl"


def count_jsonl_rows(path: Path) -> int:
    if not path.exists():
        return 0
    return len(read_jsonl(path))


def resolve_variant_language_filters(
    stage_jobs: Dict[str, Dict[str, Any]],
    stage_order: Sequence[str],
) -> Dict[str, Optional[str]]:
    filters: Dict[str, Optional[str]] = {}
    for stage_name in stage_order:
        stage_job = stage_jobs.get(stage_name)
        if not isinstance(stage_job, dict):
            continue
        variant_name = normalize_string(stage_job.get("dataset_variant"))
        if variant_name is None:
            continue
        language_filter = normalize_string(stage_job.get("language_filter"))
        existing = filters.get(variant_name)
        if existing is None:
            filters[variant_name] = language_filter
            continue
        if existing != language_filter:
            raise ValueError(
                f"Pilot dataset build found conflicting language filters for variant={variant_name}: "
                f"{existing!r} vs {language_filter!r}"
            )
    return filters


def build_pilot_dataset(
    manifest: Dict[str, Any],
    stage_jobs: Dict[str, Dict[str, Any]],
    stage_order: Sequence[str],
    pilot_dataset_dir: Path,
    state: Dict[str, Any],
    state_path: Path,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    full_dataset_dir = Path(manifest["dataset_dir"]).resolve()
    primary_variant_name = normalize_string(manifest.get("primary_variant_name")) or "stage2_silver"
    dataset_name_prefix = infer_dataset_name_prefix(next(iter(stage_jobs.values())))
    variant_names: List[str] = []
    for stage_name in stage_order:
        stage_job = stage_jobs.get(stage_name)
        if not isinstance(stage_job, dict):
            continue
        dataset_variant = normalize_string(stage_job.get("dataset_variant"))
        if dataset_variant is None or dataset_variant in variant_names:
            continue
        variant_names.append(dataset_variant)
    if not variant_names:
        raise ValueError("No dataset variants resolved for pilot dataset build.")
    first_stage_variant = normalize_string((stage_jobs.get(stage_order[0]) or {}).get("dataset_variant")) if stage_order else None
    variant_language_filters = resolve_variant_language_filters(stage_jobs=stage_jobs, stage_order=stage_order)

    if pilot_dataset_dir.exists():
        shutil.rmtree(pilot_dataset_dir)
    ensure_dir(pilot_dataset_dir)
    ensure_dir(pilot_dataset_dir / "by_split")
    ensure_dir(pilot_dataset_dir / "variants")

    variant_summary: Dict[str, Dict[str, Dict[str, int]]] = {}
    primary_split_rows: Dict[str, List[Dict[str, Any]]] = {"train": [], "dev": [], "test": []}

    for variant_name in variant_names:
        variant_dir = pilot_dataset_dir / "variants" / variant_name / "by_split"
        ensure_dir(variant_dir)
        variant_lang_dir = pilot_dataset_dir / "variants" / variant_name / "by_lang_split"
        ensure_dir(variant_lang_dir)
        variant_summary[variant_name] = {}
        language_filter = normalize_string(variant_language_filters.get(variant_name))
        for split in ["train", "dev", "test"]:
            if language_filter is not None:
                source_path = detect_variant_language_path(full_dataset_dir, variant_name, language_filter, split)
            else:
                source_path = detect_variant_path(full_dataset_dir, variant_name, split)
            rows = read_jsonl(source_path) if source_path.exists() else []
            if split == "train":
                per_lang = args.pilot_stage1_train_per_lang if variant_name == first_stage_variant else args.pilot_stage2_train_per_lang
            elif split == "test":
                per_lang = args.pilot_test_per_lang
            else:
                per_lang = 0

            sampled_rows = sample_rows_by_language(rows, per_lang=per_lang, seed=stable_seed(variant_name, split))
            write_jsonl(variant_dir / f"{split}.jsonl", sampled_rows)
            for language, language_rows in sorted(group_rows_by_language(sampled_rows).items()):
                write_jsonl(variant_lang_dir / f"{language}_{split}.jsonl", language_rows)
            variant_summary[variant_name][split] = {
                "rows": len(sampled_rows),
                "languages": dict(sorted((lang, len(lang_rows)) for lang, lang_rows in group_rows_by_language(sampled_rows).items())),
            }

            if variant_name == primary_variant_name:
                primary_split_rows[split] = sampled_rows

    for split, rows in primary_split_rows.items():
        write_jsonl(pilot_dataset_dir / "by_split" / f"{split}.jsonl", rows)
        for language, language_rows in sorted(group_rows_by_language(rows).items()):
            write_jsonl(pilot_dataset_dir / "by_lang_split" / f"{language}_{split}.jsonl", language_rows)

    variant_languages: Dict[str, List[str]] = {}
    manifest_variant_languages = manifest.get("variant_languages") if isinstance(manifest.get("variant_languages"), dict) else {}
    for variant_name in variant_names:
        language_filter = normalize_string(variant_language_filters.get(variant_name))
        if language_filter is not None:
            variant_languages[variant_name] = [language_filter]
        else:
            variant_languages[variant_name] = list(manifest_variant_languages.get(variant_name, []))
    dataset_info = build_dataset_info(
        dataset_name_prefix=dataset_name_prefix,
        variant_names=variant_names,
        variant_languages=variant_languages,
    )
    write_json(pilot_dataset_dir / "dataset_info.json", dataset_info)

    prompt_bundle_src = full_dataset_dir / "prompt_bundle.json"
    if prompt_bundle_src.exists():
        shutil.copy2(prompt_bundle_src, pilot_dataset_dir / "prompt_bundle.json")

    pilot_manifest = {
        "created_at": utc_now(),
        "source_manifest": manifest.get("phase4_run_root"),
        "primary_variant_name": primary_variant_name,
        "dataset_name_prefix": dataset_name_prefix,
        "variant_names": variant_names,
        "variant_language_filters": variant_language_filters,
        "variant_summary": variant_summary,
    }
    write_json(pilot_dataset_dir / "pilot_manifest.json", pilot_manifest)

    state.setdefault("pilot", {})["dataset"] = {
        "prepared_at": utc_now(),
        "dataset_dir": str(pilot_dataset_dir),
        "manifest_path": str(pilot_dataset_dir / "pilot_manifest.json"),
        "summary": variant_summary,
    }
    write_json_atomic(state_path, state)
    return pilot_manifest


def build_pilot_job_yaml(
    full_job: Dict[str, Any],
    pilot_dataset_dir: Path,
    pilot_output_dir: Path,
    yaml_path: Path,
    resume_output_dir: Optional[Path],
    is_first_stage: bool,
    use_modelscope: bool,
    args: argparse.Namespace,
) -> None:
    source_yaml_path = Path(full_job["yaml_path"]).resolve()
    payload = yaml.safe_load(source_yaml_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Unexpected YAML format: {source_yaml_path}")

    stage_name = normalize_string(full_job.get("stage_name")) or "stage"
    model_name_or_path = normalize_string(payload.get("model_name_or_path"))
    if model_name_or_path is not None:
        payload["model_name_or_path"] = resolve_local_model_path(model_name_or_path, use_modelscope=use_modelscope)
    payload["dataset_dir"] = str(pilot_dataset_dir)
    payload["output_dir"] = str(pilot_output_dir)
    payload["overwrite_output_dir"] = False
    payload["logging_steps"] = args.pilot_logging_steps
    payload["save_steps"] = args.pilot_save_steps
    payload["plot_loss"] = True
    payload["report_to"] = "none"
    payload["resume_from_checkpoint"] = None

    if is_first_stage:
        payload["num_train_epochs"] = args.pilot_stage1_epochs
        payload.pop("adapter_name_or_path", None)
        payload.pop("create_new_adapter", None)
    else:
        payload["num_train_epochs"] = args.pilot_stage2_epochs
        if bool(full_job.get("continue_from_previous_adapter")):
            if resume_output_dir is None:
                raise ValueError(f"Pilot stage {stage_name} requires a previous stage output dir.")
            payload["adapter_name_or_path"] = str(resume_output_dir)
            payload["create_new_adapter"] = False
        else:
            payload.pop("adapter_name_or_path", None)
            payload.pop("create_new_adapter", None)

    ensure_dir(yaml_path.parent)
    yaml_path.write_text(yaml.safe_dump(payload, allow_unicode=True, sort_keys=False), encoding="utf-8")


def build_full_job_yaml(
    *,
    full_job: Dict[str, Any],
    yaml_path: Path,
    args: argparse.Namespace,
    use_modelscope: bool,
) -> None:
    source_yaml_path = Path(full_job["yaml_path"]).resolve()
    payload = yaml.safe_load(source_yaml_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Unexpected YAML format: {source_yaml_path}")

    model_name_or_path = normalize_string(payload.get("model_name_or_path"))
    if model_name_or_path is not None:
        payload["model_name_or_path"] = resolve_local_model_path(model_name_or_path, use_modelscope=use_modelscope)

    payload["logging_steps"] = args.full_logging_steps
    payload["save_steps"] = args.full_save_steps
    payload["resume_from_checkpoint"] = None
    payload["overwrite_output_dir"] = False
    payload["report_to"] = "none"

    ensure_dir(yaml_path.parent)
    yaml_path.write_text(yaml.safe_dump(payload, allow_unicode=True, sort_keys=False), encoding="utf-8")


def is_pid_running(pid: Optional[int]) -> bool:
    if pid is None or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def find_last_checkpoint(output_dir: Path) -> Optional[str]:
    checkpoints: List[Tuple[int, str]] = []
    if not output_dir.exists():
        return None
    for child in output_dir.iterdir():
        if not child.is_dir() or not child.name.startswith("checkpoint-"):
            continue
        try:
            step = int(child.name.split("-", 1)[1])
        except Exception:
            continue
        checkpoints.append((step, child.name))
    if not checkpoints:
        return None
    return max(checkpoints)[1]


def is_stage_complete(output_dir: Path) -> bool:
    if not output_dir.exists():
        return False
    artifact_names = [
        "adapter_model.safetensors",
        "adapter_model.bin",
        "model.safetensors",
        "pytorch_model.bin",
    ]
    if any((output_dir / name).exists() for name in artifact_names):
        return True
    return False


def base_env(cuda_visible_devices: str, use_modelscope: bool) -> Dict[str, str]:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
    env["USE_MODELSCOPE_HUB"] = "1" if use_modelscope else "0"
    return env


def stream_subprocess(command: Sequence[str], cwd: Path, env: Dict[str, str], log_path: Path) -> int:
    ensure_dir(log_path.parent)
    with log_path.open("a", encoding="utf-8") as log_file:
        log_file.write(f"\n[{utc_now()}] START {' '.join(command)}\n")
        log_file.flush()
        process = subprocess.Popen(
            list(command),
            cwd=str(cwd),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            log_file.write(line)
            log_file.flush()
            print(line, end="")
        return_code = process.wait()
        log_file.write(f"[{utc_now()}] END return_code={return_code}\n")
        log_file.flush()
        return return_code


def run_train_job(
    *,
    runtime_python: str,
    llamafactory_root: Path,
    yaml_path: Path,
    output_dir: Path,
    log_path: Path,
    state: Dict[str, Any],
    state_path: Path,
    state_namespace: str,
    stage_name: str,
    cuda_visible_devices: str,
    retry_attempts: int,
    retry_sleep_seconds: int,
) -> None:
    namespace = state.setdefault(state_namespace, {})
    jobs_state = namespace.setdefault("jobs", {})
    stage_state = jobs_state.setdefault(stage_name, {})

    if is_stage_complete(output_dir):
        stage_state.update(
            {
                "status": "completed",
                "completed_at": utc_now(),
                "output_dir": str(output_dir),
                "yaml_path": str(yaml_path),
                "last_checkpoint": find_last_checkpoint(output_dir),
                "log_path": str(log_path),
            }
        )
        write_json_atomic(state_path, state)
        return

    command = [runtime_python, "-m", "llamafactory.cli", "train", str(yaml_path)]
    env = base_env(cuda_visible_devices, use_modelscope=state.get("use_modelscope", True))

    attempts = 0
    cumulative_attempts = int(stage_state.get("cumulative_attempts") or 0)
    while attempts < retry_attempts:
        attempts += 1
        cumulative_attempts += 1
        stage_state.update(
            {
                "status": "running",
                "attempts": attempts,
                "cumulative_attempts": cumulative_attempts,
                "started_at": utc_now(),
                "output_dir": str(output_dir),
                "yaml_path": str(yaml_path),
                "log_path": str(log_path),
                "last_checkpoint": find_last_checkpoint(output_dir),
            }
        )
        write_json_atomic(state_path, state)

        return_code = stream_subprocess(command=command, cwd=llamafactory_root, env=env, log_path=log_path)
        if return_code == 0 and is_stage_complete(output_dir):
            stage_state.update(
                {
                    "status": "completed",
                    "completed_at": utc_now(),
                    "last_checkpoint": find_last_checkpoint(output_dir),
                }
            )
            write_json_atomic(state_path, state)
            return

        stage_state.update(
            {
                "status": "failed",
                "failed_at": utc_now(),
                "return_code": return_code,
                "last_checkpoint": find_last_checkpoint(output_dir),
            }
        )
        write_json_atomic(state_path, state)
        if attempts >= retry_attempts:
            raise RuntimeError(f"Training job failed after {attempts} attempt(s): {yaml_path}")
        time.sleep(retry_sleep_seconds)


def run_eval(
    *,
    runtime_python: str,
    repo_root: Path,
    dataset_path: Path,
    model_name_or_path: str,
    adapter_name_or_path: Path,
    output_dir: Path,
    state: Dict[str, Any],
    state_path: Path,
    cuda_visible_devices: str,
    max_samples: int,
    max_new_tokens: int,
    temperature: float,
) -> Dict[str, Any]:
    metrics_path = output_dir / "metrics.json"
    if metrics_path.exists():
        metrics = load_json(metrics_path)
        state.setdefault("pilot", {})["evaluation"] = {
            "status": "completed",
            "completed_at": utc_now(),
            "output_dir": str(output_dir),
            "metrics_path": str(metrics_path),
            "metrics": metrics,
        }
        write_json_atomic(state_path, state)
        return metrics

    ensure_dir(output_dir)
    log_path = output_dir / "eval.log"
    resolved_model_path = resolve_local_model_path(model_name_or_path, use_modelscope=bool(state.get("use_modelscope", True)))
    command = [
        runtime_python,
        str((repo_root / "DAR" / "fine_tuning" / "eval_phase4_generation.py").resolve()),
        "--dataset",
        str(dataset_path),
        "--model-name-or-path",
        resolved_model_path,
        "--adapter-name-or-path",
        str(adapter_name_or_path),
        "--output-dir",
        str(output_dir),
        "--protocol",
        "teacher_json",
        "--device",
        "cuda",
        "--max-samples",
        str(max_samples),
        "--batch-size",
        "4",
        "--save-every",
        "100",
        "--max-new-tokens",
        str(max_new_tokens),
        "--temperature",
        str(temperature),
    ]
    state.setdefault("pilot", {})["evaluation"] = {
        "status": "running",
        "started_at": utc_now(),
        "output_dir": str(output_dir),
        "metrics_path": str(metrics_path),
    }
    write_json_atomic(state_path, state)

    return_code = stream_subprocess(
        command=command,
        cwd=repo_root,
        env=base_env(cuda_visible_devices, use_modelscope=state.get("use_modelscope", True)),
        log_path=log_path,
    )
    if return_code != 0:
        state.setdefault("pilot", {})["evaluation"] = {
            "status": "failed",
            "failed_at": utc_now(),
            "return_code": return_code,
            "output_dir": str(output_dir),
            "metrics_path": str(metrics_path),
        }
        write_json_atomic(state_path, state)
        raise RuntimeError("Pilot generation evaluation failed.")

    metrics = load_json(metrics_path)
    state.setdefault("pilot", {})["evaluation"] = {
        "status": "completed",
        "completed_at": utc_now(),
        "output_dir": str(output_dir),
        "metrics_path": str(metrics_path),
        "metrics": metrics,
    }
    write_json_atomic(state_path, state)
    return metrics


def write_helper_scripts(
    *,
    repo_root: Path,
    state_root: Path,
    script_path: Path,
    args: argparse.Namespace,
    manifest_path: Path,
) -> None:
    scripts_dir = state_root / "scripts"
    ensure_dir(scripts_dir)
    runtime_python = normalize_string(args.runtime_python) or str(Path.home() / ".conda" / "envs" / "DAR" / "bin" / "python")
    common = (
        f"{json.dumps(runtime_python)} {json.dumps(str(script_path))} "
        f"--manifest {json.dumps(str(manifest_path))} "
        f"--student {json.dumps(args.student)} "
        f"--tuning-mode {json.dumps(args.tuning_mode)} "
        f"--cuda-visible-devices {json.dumps(args.cuda_visible_devices)} "
        f"--pilot-root-name {json.dumps(normalize_string(args.pilot_root_name) or '')} "
        f"--pilot-stage1-train-per-lang {args.pilot_stage1_train_per_lang} "
        f"--pilot-stage2-train-per-lang {args.pilot_stage2_train_per_lang} "
        f"--pilot-test-per-lang {args.pilot_test_per_lang} "
        f"--pilot-stage1-epochs {args.pilot_stage1_epochs} "
        f"--pilot-stage2-epochs {args.pilot_stage2_epochs} "
        f"--pilot-logging-steps {args.pilot_logging_steps} "
        f"--pilot-save-steps {args.pilot_save_steps} "
        f"--pilot-eval-max-samples {args.pilot_eval_max_samples} "
        f"--pilot-eval-max-new-tokens {args.pilot_eval_max_new_tokens} "
        f"--pilot-eval-temperature {args.pilot_eval_temperature} "
        f"--retry-attempts {args.retry_attempts} "
        f"--retry-sleep-seconds {args.retry_sleep_seconds} "
        f"--full-logging-steps {args.full_logging_steps} "
        f"--full-save-steps {args.full_save_steps} "
        f"{'--use-modelscope' if args.use_modelscope else '--no-use-modelscope'}"
    )
    pilot_script = (
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        f"cd {json.dumps(str(repo_root), ensure_ascii=False)}\n"
        f"{common} --run-full --background-full\n"
    )
    resume_full_script = (
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        f"cd {json.dumps(str(repo_root), ensure_ascii=False)}\n"
        f"{common} --full-only --run-full\n"
    )
    (scripts_dir / "run_pilot_then_full.sh").write_text(pilot_script, encoding="utf-8")
    (scripts_dir / "resume_full.sh").write_text(resume_full_script, encoding="utf-8")
    (scripts_dir / "run_pilot_then_full.sh").chmod(0o755)
    (scripts_dir / "resume_full.sh").chmod(0o755)


def launch_full_in_background(
    *,
    runtime_python: str,
    repo_root: Path,
    script_path: Path,
    manifest_path: Path,
    state_root: Path,
    args: argparse.Namespace,
    state: Dict[str, Any],
    state_path: Path,
) -> Dict[str, Any]:
    full_state = state.setdefault("full", {})
    background = full_state.get("background")
    if isinstance(background, dict) and is_pid_running(background.get("pid")):
        return background

    log_path = state_root / "full" / "background.log"
    ensure_dir(log_path.parent)
    command = [
        runtime_python,
        str(script_path),
        "--manifest",
        str(manifest_path),
        "--student",
        args.student,
        "--tuning-mode",
        args.tuning_mode,
        "--runtime-python",
        runtime_python,
        "--cuda-visible-devices",
        args.cuda_visible_devices,
        "--retry-attempts",
        str(args.retry_attempts),
        "--retry-sleep-seconds",
        str(args.retry_sleep_seconds),
        "--full-logging-steps",
        str(args.full_logging_steps),
        "--full-save-steps",
        str(args.full_save_steps),
        "--full-only",
        "--run-full",
    ]
    command.append("--use-modelscope" if args.use_modelscope else "--no-use-modelscope")
    if args.llamafactory_root:
        command.extend(["--llamafactory-root", args.llamafactory_root])
    if args.pilot_root_name:
        command.extend(["--pilot-root-name", args.pilot_root_name])

    log_file = log_path.open("a", encoding="utf-8")
    process = subprocess.Popen(
        command,
        cwd=str(repo_root),
        env=base_env(args.cuda_visible_devices, use_modelscope=args.use_modelscope),
        stdout=log_file,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        text=True,
    )
    background_state = {
        "pid": process.pid,
        "started_at": utc_now(),
        "log_path": str(log_path),
        "command": command,
    }
    full_state["background"] = background_state
    write_json_atomic(state_path, state)
    return background_state


def main() -> None:
    args = parse_args()
    manifest_path = Path(args.manifest).resolve()
    manifest = load_json(manifest_path)
    repo_root = Path(manifest["phase4_run_root"]).resolve().parents[3]
    runtime_python = resolve_runtime_python(args, manifest)
    args.runtime_python = runtime_python
    llamafactory_root = resolve_llamafactory_root(args, repo_root)
    args.student = resolve_student_name(manifest, args.student)

    selected_jobs = pick_jobs(manifest, student=args.student, tuning_mode=args.tuning_mode)
    stage_jobs = job_by_stage(selected_jobs)
    stage_order = resolve_stage_order(manifest, selected_jobs)
    if len(stage_order) < 2:
        raise ValueError(f"Current workflow expects at least two curriculum stages, got {stage_order}")
    final_stage_name = stage_order[-1]
    final_stage_job = stage_jobs[final_stage_name]
    final_stage_dataset_variant = normalize_string(final_stage_job.get("dataset_variant")) or final_stage_name
    selected_language = normalize_string(final_stage_job.get("language_filter"))

    state_root_name = normalize_string(args.pilot_root_name) or f"{args.student}_{args.tuning_mode}"
    state_root = Path(manifest["phase4_run_root"]).resolve() / "training" / "orchestrator" / state_root_name
    state_path = state_root / "state.json"
    state = load_json(state_path) if state_path.exists() else {}
    state.update(
        {
            "manifest_path": str(manifest_path),
            "phase4_run_root": manifest.get("phase4_run_root"),
            "student": args.student,
            "tuning_mode": args.tuning_mode,
            "runtime_python": runtime_python,
            "llamafactory_root": str(llamafactory_root),
            "use_modelscope": bool(args.use_modelscope),
            "updated_at": utc_now(),
        }
    )
    write_json_atomic(state_path, state)
    write_helper_scripts(
        repo_root=repo_root,
        state_root=state_root,
        script_path=Path(__file__).resolve(),
        args=args,
        manifest_path=manifest_path,
    )

    if not args.full_only and not args.skip_pilot:
        pilot_root = state_root / "pilot"
        pilot_dataset_dir = pilot_root / "datasets"
        build_pilot_dataset(
            manifest=manifest,
            stage_jobs=stage_jobs,
            stage_order=stage_order,
            pilot_dataset_dir=pilot_dataset_dir,
            state=state,
            state_path=state_path,
            args=args,
        )

        pilot_output_dirs: Dict[str, Path] = {}
        previous_stage_output_dir: Optional[Path] = None
        for stage_index, stage_name in enumerate(stage_order):
            stage_job = stage_jobs[stage_name]
            stage_output_dir = pilot_root / "training" / "runs" / f"{args.student}_{args.tuning_mode}" / stage_name
            stage_yaml = pilot_root / "training" / "configs" / f"{args.student}_{args.tuning_mode}__{stage_name}.yaml"
            build_pilot_job_yaml(
                full_job=stage_job,
                pilot_dataset_dir=pilot_dataset_dir,
                pilot_output_dir=stage_output_dir,
                yaml_path=stage_yaml,
                resume_output_dir=previous_stage_output_dir,
                is_first_stage=(stage_index == 0),
                use_modelscope=bool(args.use_modelscope),
                args=args,
            )
            run_train_job(
                runtime_python=runtime_python,
                llamafactory_root=llamafactory_root,
                yaml_path=stage_yaml,
                output_dir=stage_output_dir,
                log_path=pilot_root / "logs" / f"{stage_name}.log",
                state=state,
                state_path=state_path,
                state_namespace="pilot",
                stage_name=stage_name,
                cuda_visible_devices=args.cuda_visible_devices,
                retry_attempts=args.retry_attempts,
                retry_sleep_seconds=args.retry_sleep_seconds,
            )
            pilot_output_dirs[stage_name] = stage_output_dir
            previous_stage_output_dir = stage_output_dir

        eval_dataset_path = (
            detect_variant_language_path(pilot_dataset_dir, final_stage_dataset_variant, selected_language, "test")
            if selected_language is not None
            else detect_variant_path(pilot_dataset_dir, final_stage_dataset_variant, "test")
        )
        eval_dataset_rows = count_jsonl_rows(eval_dataset_path)
        if eval_dataset_rows > 0:
            eval_metrics = run_eval(
                runtime_python=runtime_python,
                repo_root=repo_root,
                dataset_path=eval_dataset_path,
                model_name_or_path=str(final_stage_job["model_name_or_path"]),
                adapter_name_or_path=pilot_output_dirs[final_stage_name],
                output_dir=pilot_root / "eval" / final_stage_name,
                state=state,
                state_path=state_path,
                cuda_visible_devices=args.cuda_visible_devices,
                max_samples=args.pilot_eval_max_samples,
                max_new_tokens=args.pilot_eval_max_new_tokens,
                temperature=args.pilot_eval_temperature,
            )
        else:
            eval_metrics = {
                "status": "skipped",
                "reason": "pilot test split is empty",
                "dataset_path": str(eval_dataset_path),
                "dataset_rows": 0,
            }
            state.setdefault("pilot", {})["evaluation"] = {
                "status": "skipped",
                "skipped_at": utc_now(),
                **eval_metrics,
            }
            write_json_atomic(state_path, state)
        print(json.dumps({"pilot_eval_metrics": eval_metrics}, ensure_ascii=False, indent=2))

    if args.run_full:
        if args.background_full and not args.full_only:
            background_state = launch_full_in_background(
                runtime_python=runtime_python,
                repo_root=repo_root,
                script_path=Path(__file__).resolve(),
                manifest_path=manifest_path,
                state_root=state_root,
                args=args,
                state=state,
                state_path=state_path,
            )
            print(json.dumps({"full_background": background_state, "state_path": str(state_path)}, ensure_ascii=False, indent=2))
            return

        full_logs_dir = state_root / "full" / "logs"
        ordered_full_jobs: List[Dict[str, Any]] = []
        for stage_name in stage_order:
            stage_job = stage_jobs.get(stage_name)
            if isinstance(stage_job, dict):
                ordered_full_jobs.append(stage_job)
        for full_job in ordered_full_jobs:
            stage_name = normalize_string(full_job.get("stage_name")) or "stage"
            runtime_yaml_path = state_root / "full" / "configs" / f"{stage_name}.yaml"
            build_full_job_yaml(
                full_job=full_job,
                yaml_path=runtime_yaml_path,
                args=args,
                use_modelscope=bool(args.use_modelscope),
            )
            run_train_job(
                runtime_python=runtime_python,
                llamafactory_root=llamafactory_root,
                yaml_path=runtime_yaml_path,
                output_dir=Path(full_job["output_dir"]).resolve(),
                log_path=full_logs_dir / f"{stage_name}.log",
                state=state,
                state_path=state_path,
                state_namespace="full",
                stage_name=stage_name,
                cuda_visible_devices=args.cuda_visible_devices,
                retry_attempts=args.retry_attempts,
                retry_sleep_seconds=args.retry_sleep_seconds,
            )


if __name__ == "__main__":
    main()
