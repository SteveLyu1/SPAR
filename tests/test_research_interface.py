"""Dependency-free contract tests for the SPAR-Hate research interface."""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from spar_hate import cli
from spar_hate.release_check import CONFIG_PATHS, check_release

ROOT = Path(__file__).resolve().parents[1]


class CliContractTests(unittest.TestCase):
    def test_help(self):
        self.assertEqual(cli.main(["--help"]), 0)

    def test_public_modules_exist(self):
        for module_name in cli.COMMANDS.values():
            with self.subTest(module=module_name):
                self.assertIsNotNone(importlib.util.find_spec(module_name))

    def test_forward_exact_flags(self):
        original = sys.argv[:]
        calls = []
        def fake_run_module(name, *, run_name, alter_sys):
            calls.append((name, run_name, alter_sys, sys.argv[:]))
        with patch.object(cli.runpy, "run_module", side_effect=fake_run_module):
            self.assertEqual(cli.main(["segment", "--config", "config.json"]), 0)
        self.assertEqual(
            calls,
            [("SPAR.experiments.phase1_segmenter", "__main__", True,
              ["SPAR.experiments.phase1_segmenter", "--config", "config.json"])],
        )
        self.assertEqual(sys.argv, original)

    def test_unknown_task(self):
        self.assertEqual(cli.main(["not-a-task"]), 2)


class ReleaseContractTests(unittest.TestCase):
    def test_all_documented_configs_exist_and_parse(self):
        for name in CONFIG_PATHS:
            with self.subTest(config=name):
                obj = json.loads((ROOT / name).read_text(encoding="utf-8"))
                self.assertIsInstance(obj, dict)

    def test_prompt_references_exist(self):
        self.assertEqual(check_release(ROOT), [])

    def test_nested_role_prompts_are_checked(self):
        from spar_hate.release_check import _prompt_references
        refs = _prompt_references({
            "role_prompt_map": {"victim": {"en": "SPAR/prompts/example.txt"}},
        })
        self.assertEqual(refs, {"SPAR/prompts/example.txt"})

    def test_missing_config_is_detected(self):
        with tempfile.TemporaryDirectory() as d:
            errors = check_release(Path(d))
        self.assertEqual(len(errors), len(CONFIG_PATHS))


if __name__ == "__main__":
    unittest.main()
