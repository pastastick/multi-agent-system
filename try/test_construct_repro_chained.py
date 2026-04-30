"""Reproducer #2: replicate production EXACTLY by loading saved prompts and
chaining propose KV → construct (mimics has_past_kv=True scenario).

Run: cd ai-agent && python try/test_construct_repro_chained.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
BACKEND_DIR = THIS_DIR.parent / "backend"
sys.path.insert(0, str(BACKEND_DIR))

import torch  # noqa: E402

from llm.client import LocalLLMBackend  # noqa: E402
from llm.guided_decoding import CONSTRUCT_FACTOR_JSON_SCHEMA  # noqa: E402


def load_snapshot(path: Path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def main():
    sess_dir = BACKEND_DIR / "debug" / "llm_outputs" / "session_20260430_102157"
    propose_snap = load_snapshot(sess_dir / "0001_propose_kv_and_text.json")
    construct_snap = load_snapshot(sess_dir / "0002_construct_kv_and_text.json")

    print("=" * 60)
    print(f"Loaded propose snapshot (call_n={propose_snap['call_n']})")
    print(f"Loaded construct snapshot (call_n={construct_snap['call_n']}, has_past_kv={construct_snap['has_past_kv']})")
    print("=" * 60)
    print("Loading Qwen3-4B...")

    backend = LocalLLMBackend(
        model_name="Qwen/Qwen3-4B",
        device="cuda",
        latent_steps=10,             # propose default
        max_new_tokens=2048,
        temperature=0.6,
        top_p=0.95,
        use_realign=True,
        enable_thinking=False,
    )

    # ── Step 1: Propose (get KV chain) ───────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 1: propose (mode=kv_and_text, no past_kv) → get KV")
    print("=" * 60)
    propose_result = backend.build_messages_and_run(
        user_prompt=propose_snap["user_prompt"],
        system_prompt=propose_snap["system_prompt"],
        json_mode=False,
        past_key_values=None,
        mode="kv_and_text",
        role="propose",
        latent_steps=10,
        temperature=0.6,
        json_schema=None,             # no guided for propose
    )
    print(f"propose text_len = {len(propose_result.text or '')}")
    print(f"propose has_kv   = {propose_result.kv_cache is not None}")
    print(f"propose text head: {(propose_result.text or '')[:200]!r}")

    propose_kv = propose_result.kv_cache

    # ── Step 2: Construct (with past_kv from propose, GUIDED) ────────────
    print("\n" + "=" * 60)
    print("STEP 2: construct (mode=kv_and_text, past_kv FROM PROPOSE, guided=True)")
    print("=" * 60)
    construct_result = backend.build_messages_and_run(
        user_prompt=construct_snap["user_prompt"],
        system_prompt=construct_snap["system_prompt"],
        json_mode=False,
        past_key_values=propose_kv,    # ← KEY: chained from propose
        mode="kv_and_text",
        role="construct",
        latent_steps=20,                # match production
        temperature=0.8,                # match production attempt #1
        json_schema=CONSTRUCT_FACTOR_JSON_SCHEMA,
    )
    print(f"\nconstruct text_len = {len(construct_result.text or '')}")
    print(f"construct text     =\n{construct_result.text!r}")
    print(f"\n--- pretty ---\n{construct_result.text}")

    # JSON parse check
    print("\n--- JSON parse check ---")
    try:
        parsed = json.loads(construct_result.text)
        print(f"keys = {list(parsed.keys())}")
        for k, v in parsed.items():
            if isinstance(v, dict):
                print(f"  {k!r}: nested dict with fields {list(v.keys())}  ✓ schema OK")
            else:
                print(f"  {k!r}: {type(v).__name__}={v!r}  ← FLAT (schema VIOLATED)")
    except Exception as e:
        print(f"parse failed: {e}")


if __name__ == "__main__":
    main()
