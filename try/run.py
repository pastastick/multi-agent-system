"""
Entry point untuk batch test prompt per-agent.

Pemakaian (dari folder `ai-agent/`):

    # 1 case spesifik
    python -m try.run --group proposal_feedback --case propose

    # semua case di 1 group
    python -m try.run --group external

    # semua group / case
    python -m try.run --all

    # dry-run (print prompt saja, tidak load LLM)
    python -m try.run --all --dry-run

Override via env var:
    TEST_MODEL=Qwen/Qwen3-14B TEST_DEVICE=cuda python -m try.run --all
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from typing import Callable

from .config import CONFIG, TEST_REGISTRY
from . import (
    test_planning_evolution, test_proposal_feedback, test_coder_evaluator,
    test_multi_agent_kv, test_pair_propose_construct, test_pair_construct_coder,
    test_evolution_rekayasa, test_kv_probe_v2,
)


# Map group → modul test (sumber CASES dict)
_GROUP_MODULES = {
    "planning_evolution": test_planning_evolution,
    "proposal_feedback": test_proposal_feedback,
    "coder_evaluator": test_coder_evaluator,
    "multi_agent_kv": test_multi_agent_kv,
    "pair_propose_construct": test_pair_propose_construct,
    "pair_construct_coder": test_pair_construct_coder,
    "evolution_rekayasa": test_evolution_rekayasa,
    "kv_probe_v2": test_kv_probe_v2,
}


def _resolve(group: str, case: str) -> Callable[[], dict]:
    if group not in _GROUP_MODULES:
        raise SystemExit(f"Unknown group: {group}. Valid: {list(_GROUP_MODULES)}")
    cases = _GROUP_MODULES[group].CASES
    if case not in cases:
        raise SystemExit(
            f"Unknown case '{case}' in group '{group}'. Valid: {list(cases)}"
        )
    return cases[case]


def _run_group(group: str, stop_on_error: bool = False) -> list[dict]:
    results = []
    for case in TEST_REGISTRY[group]:
        fn = _resolve(group, case)
        try:
            results.append(fn())
        except Exception as e:
            print(f"── FATAL ── [{group}/{case}] {type(e).__name__}: {e}")
            if stop_on_error:
                raise
            results.append({"group": group, "case": case, "error": str(e), "ok_format": False})
    return results


def _print_summary(results: list[dict]) -> None:
    print("\n" + "═" * 78)
    print("SUMMARY")
    print("═" * 78)
    total_elapsed = 0.0
    ok = 0
    for r in results:
        tag = "✔" if r.get("ok_format") else "❌"
        elapsed = r.get("elapsed_s", 0.0) or 0.0
        total_elapsed += elapsed
        if r.get("ok_format"):
            ok += 1
        group = r.get("group", "?")
        case = r.get("case", "?")
        msg = f"{tag} [{group}/{case}] elapsed={elapsed:.2f}s"
        if "error" in r:
            msg += f"  ERROR={r['error'][:120]}"
        print(msg)
    print("─" * 78)
    print(f"passed: {ok}/{len(results)}  |  total_elapsed: {total_elapsed:.2f}s")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run per-agent prompt LLM tests.")
    parser.add_argument("--group", choices=list(TEST_REGISTRY), help="group to run")
    parser.add_argument("--case", help="specific case within the group")
    parser.add_argument("--all", action="store_true", help="run all groups & cases")
    parser.add_argument("--dry-run", action="store_true", help="print prompts without calling LLM")
    parser.add_argument(
        "--stop-on-error", action="store_true",
        help="abort batch if any case raises (default: log & continue)",
    )
    parser.add_argument("--list", action="store_true", help="list all groups & cases, then exit")
    args = parser.parse_args(argv)

    if args.list:
        for g, cases in TEST_REGISTRY.items():
            print(f"[{g}]")
            for c in cases:
                print(f"  {c}")
        return 0

    if args.dry_run:
        CONFIG.dry_run = True

    t0 = time.time()
    results: list[dict] = []

    if args.all:
        for g in TEST_REGISTRY:
            results.extend(_run_group(g, stop_on_error=args.stop_on_error))
    elif args.group and args.case:
        fn = _resolve(args.group, args.case)
        results.append(fn())
    elif args.group:
        results.extend(_run_group(args.group, stop_on_error=args.stop_on_error))
    else:
        parser.print_help()
        return 2

    _print_summary(results)
    print(f"\nTotal wall-clock: {time.time() - t0:.2f}s")

    # Dump ringkasan ke JSON
    summary_path = CONFIG.output_dir / f"_summary_{time.strftime('%Y%m%d_%H%M%S')}.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    # Buang objek non-serializable (response bisa panjang tapi tetap str)
    slim = [
        {k: v for k, v in r.items() if k != "parsed" or v is None or isinstance(v, dict)}
        for r in results
    ]
    summary_path.write_text(json.dumps(slim, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Summary JSON: {summary_path}")

    return 0 if all(r.get("ok_format") for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
