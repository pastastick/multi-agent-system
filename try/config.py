"""
Konfigurasi uji prompt per-agent.

Fokus: cek apakah LLM mengikuti instruksi dan format output sesuai prompt.
Bukan untuk pipeline penuh — setiap test call dijalankan secara terisolasi
dengan input kontekstual yang dimock di fixtures.py.

Pemakaian:
    python -m try.run --group proposal_feedback --case propose
    python -m try.run --group coder_evaluator --case code_feedback
    python -m try.run --all
    python -m try.run --group planning_evolution --dry-run   # tidak call LLM; cuma print prompt

Ganti parameter global di sini (model, max_tokens, temperature, output dir, dll).
"""

from __future__ import annotations

import os
from pathlib import Path
from dataclasses import dataclass, field


@dataclass
class TestConfig:
    # ── Backend LLM ──────────────────────────────────────────────────────
    # Set sesuai environment (Qwen3-4B / Qwen3-14B / etc.)
    model_name: str = os.environ.get("TEST_MODEL", "Qwen/Qwen3-30B-A3B")
    device: str = os.environ.get("TEST_DEVICE", "cuda")

    # ── Sampling ─────────────────────────────────────────────────────────
    max_new_tokens: int = 360000
    temperature: float = 0.8
    top_p: float = 0.95

    # Beberapa test (construct, feedback, evaluator) minta JSON ketat.
    # Bila True, force json_mode=True di call yg relevan — override fixture.
    force_json_mode: bool = False

    # ── Eksekusi ─────────────────────────────────────────────────────────
    # dry_run=True: TIDAK panggil LLM. Hanya render + print system+user prompt
    # beserta skema output yang diharapkan. Berguna untuk inspeksi cepat.
    dry_run: bool = False

    # Batasi output yg di-print ke konsol; full output tetap disimpan ke file.
    console_preview_chars: int = 2000

    # ── Logging ──────────────────────────────────────────────────────────
    # Folder output: prompt rendered + LLM response + validation status.
    # Struktur: {output_dir}/{group}/{case}_{timestamp}.txt
    output_dir: Path = field(default_factory=lambda: Path(__file__).parent / "outputs")

    # Simpan system + user prompt terpisah (selain response LLM)?
    save_prompts: bool = True


# Default config (boleh diedit langsung di file ini, atau di-override via env)
CONFIG = TestConfig()


# ── Peta test case ──────────────────────────────────────────────────────
# Sumber kebenaran untuk --group / --case di run.py.
# Setiap case = satu panggilan LLM yang diuji secara terpisah.

TEST_REGISTRY: dict[str, list[str]] = {
    "planning_evolution": [
        "planning",             # pipeline/planning.py:148
        "mutation_detailed",    # pipeline/evolution/mutation.py:190 (user prompt)
        "mutation_simple",      # pipeline/evolution/mutation.py:190 (simple_user)
        "crossover_detailed",   # pipeline/evolution/crossover.py:225
        "crossover_simple",     # pipeline/evolution/crossover.py:225
    ],
    "proposal_feedback": [
        "propose",              # factors/proposal.py:380 (hypothesis_gen)
        "construct",            # factors/proposal.py:648 (hypothesis2experiment)
        "feedback",             # factors/feedback.py:365 (factor_feedback_generation)
        "construct_guided_vs_free",    # guided JSON (lm-format-enforcer) vs free-form
        "compare_latent_vs_text",      # propose→construct: text_only vs kv_only+kv_and_text
    ],
    "coder_evaluator": [
        "coder_retry",          # factors/coder/evolving_strategy.py:522 (LLM fix expression)
        "coder_error_summary",  # factors/coder/evolving_strategy.py:354
        "code_feedback",        # factors/coder/eva_utils.py:186
        "output_format",        # factors/coder/eva_utils.py:273
        "final_decision",       # factors/coder/eva_utils.py:615
    ],
}
