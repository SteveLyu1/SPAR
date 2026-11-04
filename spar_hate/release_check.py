"""Static release validation for the published English workflow configs.

This checker deliberately does not execute models or load licensed datasets.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from collections.abc import Sequence

CONFIG_PATHS = (
    "SPAR/configs/phase1_en_api.json",
    "SPAR/configs/phase2_en_api.json",
    "SPAR/configs/phase3_en_api.json",
    "SPAR/configs/phase35_en_api.json",
)


def _prompt_references(value: object) -> set[str]:
    """Return all local prompt paths recursively named by config keys."""
    results: set[str] = set()
    if isinstance(value, dict):
        for key, entry in value.items():
            if "prompt" in key.lower() and "path" in key.lower() and isinstance(entry, str):
                if entry and entry.endswith(".txt"):
                    results.add(entry)
            results.update(_prompt_references(entry))
    elif isinstance(value, list):
        for entry in value:
            results.update(_prompt_references(entry))
    elif isinstance(value, str):
        if value.startswith("SPAR/prompts/") and value.endswith(".txt"):
            results.add(value)
    return results


def check_release(root: Path) -> list[str]:
    """Return actionable errors without touching datasets or provider APIs."""
    errors: list[str] = []
    for config in CONFIG_PATHS:
        path = root / config
        if not path.is_file():
            errors.append(f"Missing config: {config}")
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            errors.append(f"Invalid JSON config: {config}: {exc}")
            continue
        if not isinstance(data, dict):
            errors.append(f"Config root must be a JSON object: {config}")
            continue
        for prompt in sorted(_prompt_references(data)):
            if not (root / prompt).is_file():
                errors.append(f"{config}: missing prompt {prompt}")
    return errors


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    options = parser.parse_args(argv)
    errors = check_release(options.root.resolve())
    if errors:
        for error in errors:
            print(f"FAIL: {error}")
        return 1
    print(f"PASS: validated {len(CONFIG_PATHS)} English configuration files and their local prompt references.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
