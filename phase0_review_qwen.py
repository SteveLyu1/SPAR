#!/usr/bin/env python3
from __future__ import annotations

import runpy
import sys
from pathlib import Path
from typing import List, Optional


REPO_ROOT = Path(__file__).resolve().parent
TARGET_SCRIPT = REPO_ROOT / "DAR" / "datapreprocess" / "phase0_review_qwen.py"


def _fix_common_flag_typos(args: List[str]) -> List[str]:
    typo_map = {
        "-review-pool": "--review-pool",
        "-output-overrides": "--output-overrides",
        "-output-audit": "--output-audit",
        "-output-summary": "--output-summary",
        "-model-path": "--model-path",
        "-inference-backend": "--inference-backend",
        "-openai-base-url": "--openai-base-url",
        "-openai-api-key": "--openai-api-key",
        "-openai-model": "--openai-model",
        "-openai-timeout-sec": "--openai-timeout-sec",
        "-openai-max-concurrency": "--openai-max-concurrency",
        "-openai-enable-thinking": "--openai-enable-thinking",
        "-openai-disable-thinking": "--openai-disable-thinking",
        "-prompt-path": "--prompt-path",
        "-phase0-config": "--phase0-config",
        "-batch-size": "--batch-size",
        "-max-batch-size": "--max-batch-size",
        "-min-batch-size": "--min-batch-size",
        "-auto-batch-grow": "--auto-batch-grow",
        "-disable-auto-batch-grow": "--disable-auto-batch-grow",
        "-batch-grow-interval": "--batch-grow-interval",
        "-baseline-review-size": "--baseline-review-size",
        "-enforce-baseline-review-cap": "--enforce-baseline-review-cap",
        "-disable-baseline-review-cap": "--disable-baseline-review-cap",
        "-strict-baseline-review-size": "--strict-baseline-review-size",
        "-max-samples": "--max-samples",
        "-max-input-tokens": "--max-input-tokens",
        "-max-new-tokens": "--max-new-tokens",
        "-temperature": "--temperature",
        "-top-p": "--top-p",
        "-dtype": "--dtype",
        "-device-map": "--device-map",
        "-load-in-8bit-prefer": "--load-in-8bit-prefer",
        "-enable-int4-fallback": "--enable-int4-fallback",
        "-disable-int4-fallback": "--disable-int4-fallback",
        "-prefer-4090-order": "--prefer-4090-order",
        "-disable-prefer-4090-order": "--disable-prefer-4090-order",
        "-force-4090-order": "--force-4090-order",
        "-auto-expand-single-visible-gpu": "--auto-expand-single-visible-gpu",
        "-disable-auto-expand-single-visible-gpu": "--disable-auto-expand-single-visible-gpu",
        "-enable-tf32": "--enable-tf32",
        "-disable-tf32": "--disable-tf32",
        "-apply-freeze": "--apply-freeze",
        "-freeze-run-id": "--freeze-run-id",
        "-phase0-script": "--phase0-script",
        "-python-bin": "--python-bin",
        "-allow-partial-freeze": "--allow-partial-freeze",
        "-min-parse-rate-for-freeze": "--min-parse-rate-for-freeze",
        "-retry-unparsed-json": "--retry-unparsed-json",
        "-disable-retry-unparsed-json": "--disable-retry-unparsed-json",
        "-retry-max-new-tokens": "--retry-max-new-tokens",
    }
    return [typo_map.get(a, a) for a in args]


def _existing_path(value: str) -> Optional[Path]:
    p = Path(value).expanduser()
    if p.exists():
        return p.resolve()
    q = (Path.cwd() / p).resolve()
    if q.exists():
        return q
    return None


def _latest_review_pool() -> Optional[Path]:
    base = REPO_ROOT / "DAR" / "outputs" / "phase0"
    if not base.exists():
        return None
    candidates = list(base.glob("phase0_*/review/review_pool.jsonl"))
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def _normalize_known_paths(args: List[str]) -> List[str]:
    out = list(args)
    i = 0
    while i < len(out):
        arg = out[i]
        if i + 1 >= len(out):
            i += 1
            continue

        val = out[i + 1]
        if arg == "--review-pool" and _existing_path(val) is None:
            if Path(val).name == "review_pool.jsonl":
                latest = _latest_review_pool()
                if latest is not None:
                    out[i + 1] = str(latest)

        if arg == "--prompt-path" and _existing_path(val) is None:
            candidate = (REPO_ROOT / "DAR" / "prompts" / Path(val).name).resolve()
            if candidate.exists():
                out[i + 1] = str(candidate)

        if arg == "--phase0-config" and _existing_path(val) is None:
            candidate = (REPO_ROOT / "DAR" / "configs" / Path(val).name).resolve()
            if candidate.exists():
                out[i + 1] = str(candidate)

        if arg == "--model-path" and _existing_path(val) is None:
            candidate = Path("/usr/2Tusr/llm") / Path(val).name
            if candidate.exists():
                out[i + 1] = str(candidate.resolve())

        if arg == "--phase0-script" and _existing_path(val) is None:
            candidate = (REPO_ROOT / "DAR" / "datapreprocess" / Path(val).name).resolve()
            if candidate.exists():
                out[i + 1] = str(candidate)

        i += 1
    return out


def main() -> None:
    if not TARGET_SCRIPT.exists():
        raise FileNotFoundError(f"Target script not found: {TARGET_SCRIPT}")

    args = _fix_common_flag_typos(sys.argv[1:])
    args = _normalize_known_paths(args)

    sys.argv = [str(TARGET_SCRIPT)] + args
    runpy.run_path(str(TARGET_SCRIPT), run_name="__main__")


if __name__ == "__main__":
    main()
