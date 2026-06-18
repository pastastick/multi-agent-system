"""
promptbench/artifacts.py
========================
Penataan output benchmark — penamaan file & subfolder yang MUDAH DIBACA.

Masalah lama (Phase A awal):
    results/phaseA/<agent>/<variant>__ls<N>__rep<R>.txt   ← 120–210 file datar
                                                            per agent, semua
                                                            varian+ls+rep tercampur.

Struktur baru (dipakai Phase A nested + Phase B):
    results/phaseA/<agent>/<variant_short>/ls<N>/rep<R>.txt
    results/phaseB/<chain>/<config>/rep<R>/<NN>_<agent>.txt   (+ chain.json)

`variant_short` = bagian source_id dari variant_id (tanpa prefix agent & hash),
mis. `proposal__working__258abdbbccea` → `working`. Jadi folder langsung
terbaca: "working", "git_optimalisasi", "authored_claude_latentpaper", ...

Modul ini SATU-SATUNYA sumber kebenaran untuk path & format artifact agar
Phase A dan Phase B konsisten dan tidak ada lagi folder datar yang sulit dibaca.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import os as _os

_THIS = Path(__file__).resolve()
PROMPTBENCH = _THIS.parent
RESULTS = PROMPTBENCH / "results"
PHASE_A = RESULTS / "phaseA"
PHASE_B = Path(_os.environ["PHASE_B_OVERRIDE"]) if _os.environ.get("PHASE_B_OVERRIDE") else RESULTS / "phaseB"

_SEP = "=" * 78


# ════════════════════════════════════════════════════════════════════════════
# slug helpers
# ════════════════════════════════════════════════════════════════════════════

def variant_short(variant_id: str, agent: Optional[str] = None) -> str:
    """`<agent>__<source_id>__<hash>` → `<source_id>` (folder yang mudah dibaca).

    Contoh:
        proposal__working__258abdbbccea            → working
        judger__git_gate_deterministik__6ce003675 → git_gate_deterministik
        consistency__authored_claude_latentpaper__88133558d50d
                                                   → authored_claude_latentpaper

    Fail-safe: bila format tak terduga (bukan 3 bagian), kembalikan variant_id
    apa adanya supaya tidak pernah menabrak/menghapus info.
    """
    parts = variant_id.split("__")
    if len(parts) >= 3:
        # tengah bisa saja mengandung '__'? source_id pakai single underscore,
        # jadi ambil semua bagian di antara prefix-agent dan hash-terakhir.
        return "__".join(parts[1:-1])
    if len(parts) == 2 and agent and parts[0] == agent:
        return parts[1]
    return variant_id


_AGENT_ABBREV: Dict[str, str] = {
    "construct": "c", "consistency": "co", "judger": "j", "proposal": "p",
}


def config_slug(latent_steps: int, *, seed: Optional[int] = None,
                extra: Optional[str] = None,
                overrides: Optional[Dict[str, str]] = None) -> str:
    """Slug konfigurasi run, contoh: `ls60__c-git_stepwise_final__j-working`.

    `overrides` adalah dict {agent: variant_short} dari --pick CLI. Dikodekan
    ke slug agar run dengan pick berbeda tidak saling menimpa folder.
    """
    s = f"ls{latent_steps}"
    if seed is not None:
        s += f"__seed{seed}"
    if extra:
        s += f"__{extra}"
    if overrides:
        for agent in sorted(overrides):
            abbr = _AGENT_ABBREV.get(agent, agent[:2])
            s += f"__{abbr}-{overrides[agent]}"
    return s


# ════════════════════════════════════════════════════════════════════════════
# Phase A paths (nested)
# ════════════════════════════════════════════════════════════════════════════

def phaseA_dir(agent: str, variant_id: str, latent_steps: int) -> Path:
    return PHASE_A / agent / variant_short(variant_id, agent) / f"ls{latent_steps}"


def phaseA_artifact(agent: str, variant_id: str, latent_steps: int, rep: int) -> Path:
    return phaseA_dir(agent, variant_id, latent_steps) / f"rep{rep}.txt"


# ════════════════════════════════════════════════════════════════════════════
# Phase B paths (nested)
# ════════════════════════════════════════════════════════════════════════════

def phaseB_run_dir(chain_name: str, config: str, rep: int) -> Path:
    return PHASE_B / chain_name / config / f"rep{rep}"


def phaseB_step_artifact(chain_name: str, config: str, rep: int,
                         step_idx: int, agent: str) -> Path:
    return phaseB_run_dir(chain_name, config, rep) / f"{step_idx:02d}_{agent}.txt"


def phaseB_chain_json(chain_name: str, config: str, rep: int) -> Path:
    return phaseB_run_dir(chain_name, config, rep) / "chain.json"


# ════════════════════════════════════════════════════════════════════════════
# writers
# ════════════════════════════════════════════════════════════════════════════

def _block(title: str, body: str) -> list[str]:
    return [_SEP, title, _SEP, body if body else "(empty)"]


def write_step_artifact(
    path: Path,
    *,
    header: Dict[str, Any],
    system: str,
    user: str,
    response: Optional[str],
    kv_report: Optional[Dict[str, Any]] = None,
    score_detail: Optional[Dict[str, Any]] = None,
    error: Optional[str] = None,
) -> None:
    """Tulis satu artifact langkah (prompt + response + KV report + skor).

    Format teks rapi & deterministik agar mudah di-`grep` lintas run.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = ["# " + "  ".join(f"{k}={v}" for k, v in header.items()), ""]
    if kv_report is not None:
        lines += [f"# kv: {json.dumps(kv_report, ensure_ascii=False)}"]
    lines += _block("SYSTEM", system)
    lines += [""] + _block("USER", user)
    lines += [""] + _block("RESPONSE", response or "")
    if score_detail is not None:
        lines += [""] + _block(
            "SCORE DETAIL", json.dumps(score_detail, indent=2, ensure_ascii=False)
        )
    if error:
        lines += ["", _SEP, "ERROR", _SEP, error]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
