"""
promptbench/runners/collect_variants.py
=======================================
Phase 0c — kumpulkan VARIAN prompt per-agent dari berbagai sumber menjadi
file single-agent YAML yang kompatibel dengan `load_agent(name, backend, path=)`.

Sumber (lihat GUIDE.md §5):
  - working file  : backend/latent_mas/prompts.yaml (HEAD working tree)
  - git sha/branch: `git show <ref>:backend/latent_mas/prompts.yaml`
  - hand-authored : file YAML buatan tangan di variants/_authored/ (mis. varian
                    baru bergaya paper LatentMAS). Di-MERGE apa adanya.

Output:
  variants/<agent>/<variant_id>.yaml      # { agents: { <agent>: <spec> } }
  variants/variants_manifest.yaml         # peta agent -> daftar varian + metadata

Dedup: spec identik (hash system+user+mode) disimpan SEKALI; sumber lain dicatat
sebagai alias di manifest agar tidak membenchmark prompt yang sama berkali-kali.

Jalankan (tanpa GPU):
    cd quantalatent && python -m try.promptbench.runners.collect_variants
"""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

# ── lokasi ────────────────────────────────────────────────────────────────
_THIS = Path(__file__).resolve()
PROMPTBENCH = _THIS.parent.parent                 # try/promptbench
REPO = PROMPTBENCH.parent.parent                  # quantalatent (submodule root)
VARIANTS_DIR = PROMPTBENCH / "variants"
AUTHORED_DIR = VARIANTS_DIR / "_authored"
PROMPTS_RELPATH = "backend/latent_mas/prompts.yaml"
WORKING_PROMPTS = REPO / PROMPTS_RELPATH

# ── alias nama agent -> bucket kanonik ──────────────────────────────────────
ALIAS = {
    "mutation_judger": "mutation",
    "mutation_reflection": "mutation",
    "crossover_judger": "crossover",
}
CANON_AGENTS = ["proposal", "construct", "consistency", "judger",
                "repair", "feedback", "introspect", "mutation", "crossover"]

# ── sumber git (ref, label) ─────────────────────────────────────────────────
GIT_SOURCES = [
    ("218c665", "current_newEvol"),       # = HEAD newEvol baseline (redesign aktif)
    ("e63bd08", "optimalisasi"),
    ("5ac82f7", "judger_only"),
    ("b817b70", "guidance_reentry"),
    ("117540a", "latentmas_foundation"),
    ("82cf925", "qa_style_redesign"),
    ("c5fb7fe", "judger_retry_fix"),
    ("f4b53b1", "gate_deterministik"),
    ("66b3768", "negative_memory"),
    ("feat/latentmas-rework", "branch_rework"),
]


def _git_show(ref: str, relpath: str) -> Optional[str]:
    proc = subprocess.run(
        ["git", "show", f"{ref}:{relpath}"],
        cwd=REPO, capture_output=True, text=True, check=False,
    )
    return proc.stdout if proc.returncode == 0 else None


def _load_agents_block(text: str) -> Dict[str, Any]:
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError:
        return {}
    return (data or {}).get("agents") or {}


def _spec_hash(spec: Dict[str, Any]) -> str:
    key = " ".join([
        str(spec.get("mode", "")),
        str(spec.get("system", "")).strip(),
        str(spec.get("user", "")).strip(),
    ])
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:12]


def _collect_from_block(
    agents_block: Dict[str, Any], source_id: str, source_meta: dict,
    registry: Dict[str, Dict[str, dict]],
) -> None:
    """Masukkan tiap agent dari satu sumber ke registry[canon][hash]."""
    for raw_name, spec in agents_block.items():
        if not isinstance(spec, dict):
            continue
        canon = ALIAS.get(raw_name, raw_name)
        if canon not in CANON_AGENTS:
            continue
        h = _spec_hash(spec)
        bucket = registry.setdefault(canon, {})
        if h in bucket:
            bucket[h]["aliases"].append({"source_id": source_id, "orig_name": raw_name, **source_meta})
            continue
        bucket[h] = {
            "spec": spec,
            "orig_name": raw_name,
            "primary": {"source_id": source_id, **source_meta},
            "aliases": [],
        }


def collect() -> dict:
    registry: Dict[str, Dict[str, dict]] = {}

    # 1) working file
    if WORKING_PROMPTS.exists():
        _collect_from_block(
            _load_agents_block(WORKING_PROMPTS.read_text(encoding="utf-8")),
            "working", {"kind": "working_file", "ref": PROMPTS_RELPATH},
            registry,
        )

    # 2) git sources
    for ref, label in GIT_SOURCES:
        text = _git_show(ref, PROMPTS_RELPATH)
        if text is None:
            print(f"[collect] skip {ref} ({label}): file tidak ada")
            continue
        _collect_from_block(
            _load_agents_block(text), f"git_{label}",
            {"kind": "git", "ref": ref, "label": label}, registry,
        )

    # 3) hand-authored (variants/_authored/*.yaml — boleh multi-agent)
    if AUTHORED_DIR.exists():
        for f in sorted(AUTHORED_DIR.glob("*.yaml")):
            _collect_from_block(
                _load_agents_block(f.read_text(encoding="utf-8")),
                f"authored_{f.stem}",
                {"kind": "authored", "ref": str(f.relative_to(PROMPTBENCH))},
                registry,
            )

    # 4) tulis file varian + manifest
    manifest: Dict[str, List[dict]] = {}
    for canon, bucket in registry.items():
        out_dir = VARIANTS_DIR / canon
        out_dir.mkdir(parents=True, exist_ok=True)
        entries = []
        for i, (h, item) in enumerate(sorted(bucket.items())):
            src = item["primary"]
            variant_id = f"{canon}__{src['source_id']}__{h}"
            fpath = out_dir / f"{variant_id}.yaml"
            # tulis sebagai single-agent block bernama kanonik
            payload = {"agents": {canon: item["spec"]}}
            fpath.write_text(
                yaml.safe_dump(payload, sort_keys=False, allow_unicode=True,
                               default_flow_style=False, width=100),
                encoding="utf-8",
            )
            entries.append({
                "variant_id": variant_id,
                "path": str(fpath.relative_to(PROMPTBENCH)),
                "hash": h,
                "orig_name": item["orig_name"],
                "source": src,
                "aliases": item["aliases"],
                "mode": item["spec"].get("mode"),
            })
        manifest[canon] = entries

    (VARIANTS_DIR / "variants_manifest.yaml").write_text(
        yaml.safe_dump(manifest, sort_keys=True, allow_unicode=True, width=100),
        encoding="utf-8",
    )
    return manifest


if __name__ == "__main__":
    m = collect()
    print("\n=== VARIAN TERKUMPUL (unik per-hash) ===")
    for agent in CANON_AGENTS:
        if agent in m:
            print(f"  {agent:12s}: {len(m[agent])} varian unik")
    total = sum(len(v) for v in m.values())
    print(f"  {'TOTAL':12s}: {total} file varian")
    print(f"\nManifest: {VARIANTS_DIR / 'variants_manifest.yaml'}")
