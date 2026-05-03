#!/usr/bin/env python3
from pathlib import Path
import importlib.util

module_path = Path(__file__).resolve().parent / "phase2_role_generator.py"
spec = importlib.util.spec_from_file_location("phase2_role_generator", str(module_path))
phase2 = importlib.util.module_from_spec(spec)
import sys
sys.modules[spec.name] = phase2
spec.loader.exec_module(phase2)

cases = {
    "simple": '[{"rationale":"ok","target":"t","argument":"a","harmful":1}]',
    "tail_after_think": '<think>I am reasoning...</think>\nHere is the result:\n[{"rationale":"ok","target":"t","argument":"a","harmful":1}]',
    "json_inside_think": '<think>[{"rationale":"ok","target":"t","argument":"a","harmful":1}]</think>',
    "code_fence": 'Some preface\n```json\n[{"rationale":"ok","target":"t","argument":"a","harmful":1}]\n```',
    "messy": 'assistant: <think>hm [not JSON] </think> final output: [{"rationale":"ok","target":"t","argument":"[weird] bracket","harmful":1}]',
}

for name, txt in cases.items():
    print("---", name)
    try:
        parsed = phase2.extract_json_array_candidate(txt)
        print("OK ->", parsed)
    except Exception as e:
        print("ERR ->", repr(e))
