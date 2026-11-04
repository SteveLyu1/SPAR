"""Task-oriented entry points that preserve the original SPAR stage modules.

Commands forward arguments to historical modules without changing their
experimental configuration, output schema, or model backends.
"""

from __future__ import annotations

import runpy
import sys
from collections.abc import Sequence

COMMANDS: dict[str, str] = {
    "segment": "SPAR.experiments.phase1_segmenter",
    "generate-perspectives": "SPAR.experiments.phase2_role_generator",
    "audit-and-arbitrate": "SPAR.experiments.phase3_dynamic_beacon_engine",
    "aggregate-predictions": "SPAR.experiments.phase35_aggregator",
    "evaluate-tbo": "SPAR.evaluation.tbo_eval",
    "evaluate-state": "SPAR.evaluation.state_style_eval",
    "evaluate-targets": "SPAR.evaluation.target_level_eval",
}


def usage() -> str:
    commands = "\n".join(f"  {name}" for name in (*COMMANDS, "validate-release"))
    return (
        "Usage: python -m spar_hate <command> [original script flags]\n\n"
        "Research workflows:\n" + commands + "\n\n"
        "Stage commands may call paid inference providers. Run each original "
        "script with --help to inspect its flags.\n"
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Invoke a research workflow without renaming any study artifacts."""
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in {"-h", "--help"}:
        print(usage())
        return 0
    command = args[0]
    if command == "validate-release":
        from .release_check import main as validate_main
        return validate_main(args[1:])
    if command not in COMMANDS:
        print(f"Unknown command: {command}\n\n{usage()}", file=sys.stderr)
        return 2
    prior_argv = sys.argv[:]
    try:
        sys.argv = [COMMANDS[command], *args[1:]]
        runpy.run_module(COMMANDS[command], run_name="__main__", alter_sys=True)
    finally:
        sys.argv = prior_argv
    return 0
