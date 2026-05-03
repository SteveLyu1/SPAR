#!/usr/bin/env python3
import json
import time
from pathlib import Path
from datetime import datetime

EVENTS_PATH = Path("DAR/outputs/phase1/phase1_full_dualgpu_bs8_20260416/logs/events.jsonl")
REPORT_DIR = EVENTS_PATH.parents[1] / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

POLL_INTERVAL = 10


def parse_iso(ts: str):
    try:
        return datetime.fromisoformat(ts)
    except Exception:
        # best-effort fallback
        try:
            return datetime.strptime(ts, "%Y-%m-%dT%H:%M:%S.%f%z")
        except Exception:
            return None


def load_events():
    events = []
    if not EVENTS_PATH.exists():
        return events
    with EVENTS_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except Exception:
                continue
    return events


def compute_metrics(events):
    run_started_ts = None
    init_ts = None
    total_samples = None
    last_progress_index = 0
    last_progress_ts = None
    parse_fail_count = 0

    for ev in events:
        ts = ev.get("ts")
        ev_time = None
        if isinstance(ts, str):
            ev_time = parse_iso(ts)
        stage = ev.get("stage")
        event = ev.get("event")
        payload = ev.get("payload") or {}

        if stage == "bootstrap" and event == "run_started":
            run_started_ts = ev_time
        if stage == "segmenter" and event == "qwen_local_initialized":
            init_ts = ev_time
        if stage == "load" and event == "input_loaded":
            total_samples = payload.get("samples")
        if stage == "segmenter" and event == "progress":
            try:
                idx = int(payload.get("index", 0))
            except Exception:
                idx = 0
            if idx > last_progress_index:
                last_progress_index = idx
                last_progress_ts = ev_time
        if stage == "segmenter" and event == "parse_fail":
            parse_fail_count += 1

    throughput = None
    eta = None
    remaining = None
    elapsed = None
    if init_ts and last_progress_ts and last_progress_index > 0:
        elapsed = (last_progress_ts - init_ts).total_seconds()
        if elapsed > 0:
            throughput = last_progress_index / elapsed
            if total_samples:
                remaining = max(int(total_samples) - last_progress_index, 0)
                if throughput and throughput > 0:
                    eta = remaining / throughput

    return {
        "run_started": run_started_ts.isoformat() if run_started_ts else None,
        "initialized_at": init_ts.isoformat() if init_ts else None,
        "last_progress_index": last_progress_index,
        "last_progress_ts": last_progress_ts.isoformat() if last_progress_ts else None,
        "parse_fail_count": parse_fail_count,
        "total_samples": total_samples,
        "throughput_samples_per_sec": throughput,
        "estimated_seconds_remaining": eta,
        "remaining_samples": remaining,
    }


def write_report(report, name):
    path = REPORT_DIR / name
    with path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"WROTE_REPORT: {path}")


def main():
    thresholds = [200, 400]
    seen = set()
    while thresholds:
        events = load_events()
        if not events:
            time.sleep(POLL_INTERVAL)
            continue
        metrics = compute_metrics(events)
        idx = int(metrics.get("last_progress_index") or 0)
        for t in list(thresholds):
            if idx >= t and t not in seen:
                report_name = f"midrun_health_report_{t}.json"
                write_report(metrics, report_name)
                seen.add(t)
                thresholds.remove(t)
        if thresholds:
            time.sleep(POLL_INTERVAL)
    print("Watcher finished for thresholds")


if __name__ == "__main__":
    main()
