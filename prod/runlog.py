"""prod/runlog.py — logging terminal ringan & konsisten untuk pipeline produksi.

Menyatukan format log: durasi agen (berpikir+generate), transfer KV, jumlah token,
alasan gate menolak faktor, durasi backtest. Default ke stderr agar artifacts/stdout
tetap bersih.
"""
from __future__ import annotations

import sys
import time
from typing import Optional


def fmt_dur(s: Optional[float]) -> str:
    if s is None:
        return "  -  "
    return f"{s:6.2f}s"


def fmt_int(n: Optional[int]) -> str:
    return "-" if n is None else f"{int(n):,}"


class RunLog:
    """Logger berprefiks dengan timestamp relatif sejak start."""

    def __init__(self, enabled: bool = True, stream=None) -> None:
        self.enabled = enabled
        self.stream = stream or sys.stderr
        self.t0 = time.time()

    def line(self, msg: str) -> None:
        if not self.enabled:
            return
        dt = time.time() - self.t0
        print(f"[prod +{dt:7.2f}s] {msg}", file=self.stream, flush=True)

    def section(self, title: str) -> None:
        if not self.enabled:
            return
        print(f"[prod] {'─' * 4} {title} {'─' * (60 - len(title))}",
              file=self.stream, flush=True)

    def agent(self, *, gen: int, node: str, agent: str, transfer: str,
              kv_xfer_s: float, latent_s: float, gen_s: float, total_s: float,
              in_tok: int, out_tok: int, kv_len: int, latent_steps: int) -> None:
        """Satu baris ringkas per node agent."""
        self.line(
            f"g{gen} {node:<14} {agent:<9} xfer={transfer:<11} "
            f"kv_xfer={fmt_dur(kv_xfer_s)} think={fmt_dur(latent_s)}(ls{latent_steps}) "
            f"gen={fmt_dur(gen_s)} tot={fmt_dur(total_s)} "
            f"in={fmt_int(in_tok)} out={fmt_int(out_tok)} kv_len={fmt_int(kv_len)}"
        )
