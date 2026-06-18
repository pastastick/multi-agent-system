"""
promptbench/chains/registry.py
==============================
Muat `chain_manifest.yaml` dan resolusi tiap langkah → varian KONKRET.

Resolusi varian per langkah:
  - "winner"      : varian dengan score_mean tertinggi di Phase A scoreboard.csv
                    (lintas semua latent_steps untuk agen itu).
  - "<source_id>" : varian spesifik (mis. "working", "git_optimalisasi") —
                    dicocokkan via variant_short() atau source.source_id di
                    variants_manifest.yaml.

Tanpa GPU. Murni baca YAML/CSV → cocok dipakai di dry-run maupun runpod.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import yaml

from ..artifacts import variant_short

_THIS = Path(__file__).resolve()
CHAINS_DIR = _THIS.parent
PROMPTBENCH = CHAINS_DIR.parent
VARIANTS_DIR = PROMPTBENCH / "variants"
SCOREBOARD_CSV = PROMPTBENCH / "results" / "phaseA" / "scoreboard.csv"
MANIFEST = CHAINS_DIR / "chain_manifest.yaml"


@dataclass
class StepSpec:
    idx: int
    agent: str
    variant_id: str
    variant_path: str           # absolut
    native_mode: str            # kv_only | kv_and_text
    transfer: str               # none | chain | concat
    latent_steps: Optional[int] # None = pakai global CLI
    is_terminal: bool

    @property
    def variant_short(self) -> str:
        return variant_short(self.variant_id, self.agent)


@dataclass
class ChainSpec:
    name: str
    desc: str
    steps: List[StepSpec]


# ════════════════════════════════════════════════════════════════════════════
# loaders
# ════════════════════════════════════════════════════════════════════════════

def load_chain_manifest() -> dict:
    return yaml.safe_load(MANIFEST.read_text(encoding="utf-8")) or {}


def load_variants_manifest() -> dict:
    p = VARIANTS_DIR / "variants_manifest.yaml"
    return yaml.safe_load(p.read_text(encoding="utf-8")) or {}


def _scoreboard_winner(agent: str) -> Optional[str]:
    """variant_id dengan score_mean tertinggi untuk `agent` (None bila tak ada)."""
    if not SCOREBOARD_CSV.exists():
        return None
    best_id, best_score = None, float("-inf")
    with SCOREBOARD_CSV.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("agent") != agent:
                continue
            try:
                sc = float(row.get("score_mean") or 0.0)
            except ValueError:
                continue
            if sc > best_score:
                best_score, best_id = sc, row.get("variant_id")
    return best_id


def _resolve_variant(agent: str, pick: str, vman: dict) -> dict:
    """Kembalikan entri variant {variant_id, path, mode, ...} untuk agen+pick.

    pick: "winner" | "<source_id>" (mis. "working").
    """
    entries = vman.get(agent) or []
    if not entries:
        raise KeyError(f"Tidak ada varian untuk agent '{agent}' di variants_manifest.")

    if pick == "winner":
        wid = _scoreboard_winner(agent)
        if wid:
            for e in entries:
                if e["variant_id"] == wid:
                    return e
        # fallback: varian 'working' bila scoreboard belum ada
        pick = "working"

    # cocokkan via variant_short ATAU source.source_id ATAU alias source_id
    for e in entries:
        if variant_short(e["variant_id"], agent) == pick:
            return e
        if (e.get("source") or {}).get("source_id") == pick:
            return e
        for al in e.get("aliases") or []:
            if al.get("source_id") == pick:
                return e

    avail = sorted({variant_short(e["variant_id"], agent) for e in entries})
    raise KeyError(
        f"pick '{pick}' tidak cocok untuk agent '{agent}'. Tersedia: {avail}"
    )


# ════════════════════════════════════════════════════════════════════════════
# resolve chain
# ════════════════════════════════════════════════════════════════════════════

def resolve_chain(
    name: str,
    *,
    overrides: Optional[Dict[str, str]] = None,
) -> ChainSpec:
    """Bangun ChainSpec konkret untuk `name`.

    overrides: {agent: pick} memaksa pemilihan varian (mis. {"judger": "working"}).
    Berlaku global lintas chain — menimpa `variant` di tiap langkah & defaults.
    """
    man = load_chain_manifest()
    vman = load_variants_manifest()
    overrides = overrides or {}
    default_picks = (man.get("defaults") or {}).get("picks") or {}

    chains = man.get("chains") or {}
    if name not in chains:
        raise KeyError(f"chain '{name}' tidak ada. Tersedia: {sorted(chains)}")
    cdef = chains[name]
    raw_steps = cdef.get("steps") or []

    steps: List[StepSpec] = []
    for i, st in enumerate(raw_steps):
        agent = st["agent"]
        # prioritas pick: override CLI > variant di langkah > default manifest
        pick = overrides.get(agent) or st.get("variant") or default_picks.get(agent) or "winner"
        if pick == "winner" and agent in default_picks and agent not in overrides \
                and st.get("variant") in (None, "winner"):
            pick = default_picks[agent]
        entry = _resolve_variant(agent, pick, vman)
        steps.append(StepSpec(
            idx=i,
            agent=agent,
            variant_id=entry["variant_id"],
            variant_path=str(PROMPTBENCH / entry["path"]),
            native_mode=entry.get("mode", "kv_only"),
            transfer=st.get("transfer", "chain" if i > 0 else "none"),
            latent_steps=st.get("latent_steps"),
            is_terminal=(i == len(raw_steps) - 1),
        ))
    return ChainSpec(name=name, desc=cdef.get("desc", ""), steps=steps)


def all_chain_names() -> List[str]:
    return list((load_chain_manifest().get("chains") or {}).keys())


if __name__ == "__main__":
    # smoke test (tanpa GPU): cetak resolusi tiap chain.
    for nm in all_chain_names():
        ch = resolve_chain(nm)
        print(f"\n### {nm} — {ch.desc}")
        for s in ch.steps:
            term = " [TERMINAL→decode]" if s.is_terminal else ""
            print(f"  {s.idx}. {s.agent:12} variant={s.variant_short:28} "
                  f"mode={s.native_mode:12} transfer={s.transfer}{term}")
