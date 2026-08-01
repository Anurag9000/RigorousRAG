#!/usr/bin/env python3
"""Run a strict offline adaptive-route fixture and emit a query-free JSON report."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tools.adaptive_route_experiments import run_route_benchmark
from tools.adaptive_route_fixture import fixture_adapters, load_route_fixtures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fixture", help="UTF-8 JSONL route fixture")
    parser.add_argument("--cost-weight", type=float, default=0.05)
    parser.add_argument("--latency-weight", type=float, default=0.05)
    args = parser.parse_args()
    fixtures = load_route_fixtures(args.fixture)
    report = run_route_benchmark(
        [fixture.case for fixture in fixtures],
        adapters=fixture_adapters(fixtures),
        cost_weight=args.cost_weight,
        latency_weight=args.latency_weight,
    )
    print(json.dumps(asdict(report), sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
