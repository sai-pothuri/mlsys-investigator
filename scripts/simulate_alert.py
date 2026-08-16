#!/usr/bin/env python3
"""Alert simulator for the MLSys Investigator HTTP service.

Sends a structured alert payload, polls until the investigation completes,
and prints the diagnosis.

Usage:
    python scripts/simulate_alert.py                        # random preset
    python scripts/simulate_alert.py --preset feature_drift
    python scripts/simulate_alert.py --alert "Custom alert text"
    python scripts/simulate_alert.py --url http://localhost:8080 --preset bad_deployment
    python scripts/simulate_alert.py --list
"""

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone

# ── Preset alerts drawn from the three real evaluation scenarios ───────────────

PRESETS: dict[str, dict] = {
    "feature_drift": {
        "label": "Feature Drift — Upstream Schema Change",
        "alert": (
            "ALERT: Model accuracy dropped from 0.91 to 0.76 over the past 6 hours. "
            "Prediction confidence also degraded. No system errors visible on dashboard."
        ),
    },
    "bad_deployment": {
        "label": "Bad Deployment — Feature Normalizer Shape Mismatch",
        "alert": (
            "ALERT: Model accuracy dropped from 0.91 to 0.73 over the past 7 hours. "
            "Error rate has elevated significantly. A model deployment occurred earlier today."
        ),
    },
    "label_corruption": {
        "label": "Label Pipeline Corruption — Join Key Misconfiguration",
        "alert": (
            "ALERT: Model accuracy dropped from 0.91 to 0.74 over the past 8 hours. "
            "No infrastructure errors reported. Engineers unsure if the model degraded "
            "or if something changed in the evaluation pipeline."
        ),
    },
    "latency_spike": {
        "label": "Infrastructure Latency Spike",
        "alert": (
            "ALERT: Inference latency p99 spiked from 120ms to 890ms over the last 30 minutes. "
            "Throughput fell by 40%. Accuracy unchanged. No deployment in the last 24 hours."
        ),
    },
    "silent_drift": {
        "label": "Gradual Concept Drift — No Sudden Event",
        "alert": (
            "ALERT: Model accuracy has drifted from 0.89 to 0.81 over the past 72 hours. "
            "The degradation is gradual with no single inflection point. "
            "No deployments, schema changes, or infra events detected."
        ),
    },
}


# ── HTTP helpers (stdlib only) ─────────────────────────────────────────────────

def _post(url: str, payload: dict) -> dict:
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read())


def _get(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=10) as resp:
        return json.loads(resp.read())


# ── Output formatting ─────────────────────────────────────────────────────────

def _bar(value: float, width: int = 20) -> str:
    filled = round(value * width)
    return "█" * filled + "░" * (width - filled)


def _print_result(job: dict) -> None:
    result = job.get("result", {})
    diagnosis = result.get("diagnosis")
    top_suspect = result.get("top_suspect")

    print(f"\n{'═' * 62}")
    print(f"  INVESTIGATION COMPLETE")
    print(f"{'═' * 62}")
    print(f"  Alert        : {job['alert'][:80]}{'…' if len(job['alert']) > 80 else ''}")
    print(f"  Job ID       : {job['job_id']}")
    print(f"  Terminated   : {result.get('termination_reason', '?')}")
    print(f"  Tool calls   : {result.get('tool_calls_used', '?')}")
    print(f"  Hypotheses   : {result.get('hypothesis_count', '?')}")

    if diagnosis:
        conf = diagnosis["confidence"]
        print(f"\n  {'─' * 60}")
        print(f"  ROOT CAUSE")
        print(f"  {'─' * 60}")
        print(f"  Category  : {diagnosis['root_cause'].upper().replace('_', ' ')}")
        print(f"  Confidence: [{_bar(conf)}] {conf:.0%}")
        print(f"  Diagnosis : {diagnosis['diagnosis']}")
        print(f"  Action    : {diagnosis['recommended_action']}")
        if diagnosis.get("alternative_categories"):
            alts = ", ".join(diagnosis["alternative_categories"])
            print(f"  Also ruled in: {alts}")
    elif top_suspect:
        conf = top_suspect.get("likelihood", 0)
        cat = (top_suspect.get("category") or "unknown").upper().replace("_", " ")
        print(f"\n  {'─' * 60}")
        print(f"  TOP SUSPECT (budget exhausted — no final diagnosis)")
        print(f"  {'─' * 60}")
        print(f"  Category  : {cat}")
        print(f"  Likelihood: [{_bar(conf)}] {conf:.0%}")
        print(f"  Hypothesis: {top_suspect.get('description', '')}")
    else:
        print("\n  [no diagnosis or active hypotheses]")

    facts = result.get("established_facts", [])
    if facts:
        print(f"\n  Established facts ({len(facts)}):")
        for f in facts:
            print(f"    • {f}")

    print(f"{'═' * 62}\n")


def _print_error(job: dict) -> None:
    print(f"\n  [FAILED] {job.get('error', 'unknown error')}\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Send an alert to the MLSys Investigator and print the diagnosis."
    )
    parser.add_argument(
        "--url", default="http://localhost:8080",
        help="Base URL of the investigator service (default: http://localhost:8080)",
    )
    parser.add_argument(
        "--preset", choices=list(PRESETS), metavar="NAME",
        help="Use a preset alert scenario. See --list for options.",
    )
    parser.add_argument(
        "--alert", metavar="TEXT",
        help="Custom alert text (overrides --preset)",
    )
    parser.add_argument(
        "--budget", type=int, default=8,
        help="Max tool calls for the investigation (default: 8)",
    )
    parser.add_argument(
        "--poll", type=float, default=3.0,
        help="Poll interval in seconds (default: 3.0)",
    )
    parser.add_argument(
        "--list", action="store_true",
        help="Print available presets and exit",
    )
    args = parser.parse_args()

    if args.list:
        print("\nAvailable presets (--preset <name>):\n")
        for name, meta in PRESETS.items():
            print(f"  {name:<20} {meta['label']}")
            print(f"  {' ' * 20} {meta['alert'][:72]}…\n")
        return

    # Resolve alert text
    if args.alert:
        alert_text = args.alert
        label = "custom"
    elif args.preset:
        alert_text = PRESETS[args.preset]["alert"]
        label = PRESETS[args.preset]["label"]
    else:
        import random
        chosen = random.choice(list(PRESETS.values()))
        alert_text = chosen["alert"]
        label = chosen["label"]

    base = args.url.rstrip("/")
    print(f"\n  Scenario : {label}")
    print(f"  Alert    : {alert_text[:80]}{'…' if len(alert_text) > 80 else ''}")
    print(f"  Endpoint : {base}/investigate")
    print(f"  Budget   : {args.budget} tool calls\n")

    # POST the investigation
    try:
        resp = _post(f"{base}/investigate", {"alert": alert_text, "budget": args.budget})
    except urllib.error.URLError as exc:
        print(f"  [ERROR] Could not reach {base}: {exc.reason}")
        print("  Is the server running? Start with:")
        print("    PYTHONPATH=src uvicorn server:app --app-dir src --port 8080")
        sys.exit(1)

    job_id = resp["job_id"]
    print(f"  Job queued: {job_id}")
    print(f"  Polling every {args.poll:.0f}s …\n", flush=True)

    # Poll until terminal state
    start = time.monotonic()
    spinner = ["|", "/", "─", "\\"]
    tick = 0

    while True:
        try:
            job = _get(f"{base}/jobs/{job_id}")
        except urllib.error.URLError as exc:
            print(f"\n  [ERROR] Poll failed: {exc}")
            sys.exit(1)

        status = job["status"]
        elapsed = time.monotonic() - start
        frame = spinner[tick % len(spinner)]
        print(f"\r  {frame}  status={status:<10} elapsed={elapsed:.0f}s", end="", flush=True)
        tick += 1

        if status == "completed":
            print()
            _print_result(job)
            break
        elif status == "failed":
            print()
            _print_error(job)
            sys.exit(1)

        time.sleep(args.poll)


if __name__ == "__main__":
    main()
