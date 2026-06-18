"""
promptbench/runners/reorg_phaseA.py
===================================
Migrasi artifact Phase A LAMA (datar) → layout nested yang mudah dibaca.

  LAMA : results/phaseA/<agent>/<variant_id>__ls<N>__rep<R>.txt   (120–210 file/agent)
  BARU : results/phaseA/<agent>/<variant_short>/ls<N>/rep<R>.txt

Idempoten & aman: hanya memindah file di TOP-LEVEL folder agent yang cocok pola
datar; file yang sudah nested diabaikan. Default DRY-RUN (cetak rencana saja);
beri `--apply` untuk benar-benar memindah.

Jalankan:
  python -m try.promptbench.runners.reorg_phaseA            # dry-run (lihat rencana)
  python -m try.promptbench.runners.reorg_phaseA --apply    # eksekusi pindah
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from ..artifacts import PHASE_A, variant_short

_FLAT_RE = re.compile(r"^(?P<vid>.+)__ls(?P<ls>\d+)__rep(?P<rep>\d+)\.txt$")


def plan_moves() -> list[tuple[Path, Path]]:
    moves: list[tuple[Path, Path]] = []
    if not PHASE_A.exists():
        return moves
    for agent_dir in sorted(p for p in PHASE_A.iterdir() if p.is_dir()):
        agent = agent_dir.name
        for f in sorted(agent_dir.glob("*.txt")):   # hanya top-level (bukan nested)
            m = _FLAT_RE.match(f.name)
            if not m:
                continue
            vid, ls, rep = m["vid"], m["ls"], m["rep"]
            dst = agent_dir / variant_short(vid, agent) / f"ls{ls}" / f"rep{rep}.txt"
            moves.append((f, dst))
    return moves


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="eksekusi (default: dry-run)")
    args = ap.parse_args()

    moves = plan_moves()
    if not moves:
        print("[reorg] tidak ada file datar untuk dipindah (sudah rapi?).")
        return

    print(f"[reorg] {len(moves)} file akan dipindah ke layout nested.")
    for src, dst in moves[:8]:
        print(f"  {src.relative_to(PHASE_A)}  →  {dst.relative_to(PHASE_A)}")
    if len(moves) > 8:
        print(f"  … (+{len(moves) - 8} lagi)")

    if not args.apply:
        print("[reorg] DRY-RUN. Tambahkan --apply untuk benar-benar memindah.")
        return

    for src, dst in moves:
        dst.parent.mkdir(parents=True, exist_ok=True)
        src.rename(dst)
    print(f"[reorg] selesai: {len(moves)} file dipindah.")


if __name__ == "__main__":
    main()
