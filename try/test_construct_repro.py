"""Minimal reproducer for the construct-step guided decoding bug.

Tujuan: load model 1x, panggil ONE construct call dengan kv_and_text + guided
decoding. Tidak perlu run pipeline penuh. Output diagnostic akan menunjukkan:

  - Apakah _prefix_fn benar-benar terbangun (bukan None)
  - Apakah _TransformersPrefixAllowedTokensFn.__call__ benar-benar terpanggil
    selama generate()
  - Berapa allowed tokens di setiap step (5 = escape-hatch, 200-1000 = normal)
  - Output text raw

Run: cd backend && python ../try/test_construct_repro.py
"""
from __future__ import annotations

import sys
import os

# Pastikan import path konsisten dgn cli.py mine
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.abspath(os.path.join(THIS_DIR, "..", "backend"))
sys.path.insert(0, BACKEND_DIR)

import torch  # noqa: E402

from llm.client import LocalLLMBackend  # noqa: E402
from llm.guided_decoding import CONSTRUCT_FACTOR_JSON_SCHEMA  # noqa: E402


SYSTEM_PROMPT = """You are an expert quantitative researcher constructing alpha factors.

Generate factors as JSON. Each factor must follow this nested structure:
{
  "<FactorName>": {
    "description": "...",
    "variables": {"<symbol>": "<meaning>"},
    "formulation": "...",
    "expression": "..."
  }
}

The OUTER key is the factor name (camelCase). The value MUST be a nested object
with EXACTLY 4 fields: description, variables, formulation, expression.

Available features: $open, $close, $high, $low, $volume.
Available operators: TS_MEAN, TS_STD, TS_MAX, TS_MIN, RANK, LOG.

The first two characters of your response MUST be `{"` followed by the factor name.
Do NOT begin your response with `{"$`, `{"variables`, `{"description`, or whitespace."""


USER_PROMPT = """Construct ONE alpha factor that captures short-term price reversal.

Hypothesis: Stocks that experienced a sharp 5-day drop tend to revert in the
next 1-2 days. Use $close and a moving average reference.

Output the factor JSON now. ONE factor only."""


def main():
    print("=" * 60)
    print("Loading Qwen3-4B (this can take ~30s)...")
    print("=" * 60)

    backend = LocalLLMBackend(
        model_name="Qwen/Qwen3-4B",
        device="cuda",
        latent_steps=20,           # match config: steps_construct
        max_new_tokens=512,        # smaller for faster repro
        temperature=0.3,
        top_p=0.95,
        use_realign=True,
        enable_thinking=False,
    )

    print("\n" + "=" * 60)
    print("Calling build_messages_and_run with kv_and_text + guided JSON schema")
    print("=" * 60)

    result = backend.build_messages_and_run(
        user_prompt=USER_PROMPT,
        system_prompt=SYSTEM_PROMPT,
        json_mode=False,
        past_key_values=None,                       # no KV chain — just latent + generate
        mode="kv_and_text",
        role="construct",
        latent_steps=20,
        temperature=0.3,
        json_schema=CONSTRUCT_FACTOR_JSON_SCHEMA,
    )

    print("\n" + "=" * 60)
    print("RESULT")
    print("=" * 60)
    print(f"text_len = {len(result.text or '')}")
    print(f"raw text  =\n{result.text!r}")
    print("\n--- pretty ---")
    print(result.text)

    # Try to parse as JSON
    print("\n--- JSON parse check ---")
    try:
        import json
        parsed = json.loads(result.text)
        print(f"Parsed OK. keys = {list(parsed.keys())}")
        for k, v in parsed.items():
            if isinstance(v, dict):
                print(f"  {k}: object with fields {list(v.keys())}")
            else:
                print(f"  {k}: {type(v).__name__} = {v!r}  ← FLAT VALUE = SCHEMA NOT ENFORCED")
    except Exception as e:
        print(f"JSON parse failed: {e}")


if __name__ == "__main__":
    main()