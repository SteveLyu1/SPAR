#!/usr/bin/env python3
import json
import random
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[2]
BASE_CONFIG = WORKSPACE / "DAR" / "configs" / "phase2_template.json"
SAMPLE_OUT = WORKSPACE / "DAR" / "outputs" / "phase2" / "phase1_random_sample_for_phase2.jsonl"
TEMP_CONFIG_OUT = WORKSPACE / "DAR" / "configs" / "phase2_temp_random.json"
SAMPLE_PER_SPLIT = 4

def load_jsonl(p: Path):
    rows = []
    if not p.exists():
        return rows
    with p.open('r', encoding='utf-8') as f:
        for line in f:
            line=line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            rows.append(obj)
    return rows


def main():
    random.seed()
    if not BASE_CONFIG.exists():
        print(f"Base config not found: {BASE_CONFIG}")
        raise SystemExit(1)
    cfg = json.loads(BASE_CONFIG.read_text(encoding='utf-8'))
    runtime_map = cfg.get('runtime', {}).get('phase1_language_split_inputs', {})

    collected = []
    for key in ['zh_train', 'zh_test', 'en_train', 'en_test']:
        path_str = runtime_map.get(key)
        if not path_str:
            print(f"No input path for {key} in config; skipping")
            continue
        p = (WORKSPACE / Path(path_str)).resolve()
        rows = load_jsonl(p)
        if not rows:
            print(f"No rows found for {key} at {p}; skipping")
            continue
        if len(rows) <= SAMPLE_PER_SPLIT:
            picks = rows
        else:
            picks = random.sample(rows, SAMPLE_PER_SPLIT)
        collected.extend(picks)
        print(f"Selected {len(picks)} from {key} ({p})")

    # Ensure output dir exists
    SAMPLE_OUT.parent.mkdir(parents=True, exist_ok=True)
    with SAMPLE_OUT.open('w', encoding='utf-8') as outf:
        for r in collected:
            outf.write(json.dumps(r, ensure_ascii=False) + '\n')

    # Create temp config based on base config but point to SAMPLE_OUT and disable requiring phase1 files
    cfg['paths']['input_jsonl'] = str(SAMPLE_OUT.relative_to(WORKSPACE))
    if 'runtime' not in cfg:
        cfg['runtime'] = {}
    cfg['runtime']['require_phase1_language_split_inputs'] = False

    if 'inference' not in cfg:
        cfg['inference'] = {}
    cfg['inference']['no_think_max_new_tokens'] = 512
    cfg['inference']['no_think_temperature'] = 0.7
    cfg['inference']['no_think_top_p'] = 0.8
    cfg['inference']['think_max_new_tokens'] = 1024
    cfg['inference']['think_temperature'] = 0.6
    cfg['inference']['think_top_p'] = 0.95

    TEMP_CONFIG_OUT.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f"Wrote sample input to: {SAMPLE_OUT}")
    print(f"Wrote temp config to: {TEMP_CONFIG_OUT}")

if __name__ == '__main__':
    main()
