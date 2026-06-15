"""
Negative-hypothesis memory — memori kegagalan lintas-generasi.

MASALAH: agent proposal memulai tiap generasi "bersih". Tanpa memori, ia bisa
mengusulkan ULANG kerangka MEKANISME yang sudah terbukti gagal di ronde
sebelumnya (akar pengulangan/anchoring level-mekanisme). diversity_hint hanya
melawan monokultur OPERATOR; ini melawan monokultur MEKANISME.

EVALUASI MENYELURUH (sesuai alur paper hypothesis→experiment→feedback): sebuah
trajectory dinilai gagal dari BEBERAPA lapisan, bukan hanya satu angka —
  · hypothesis  : mekanisme yang diusulkan (teks) → kunci dedup + tampilan AVOID
  · expression  : struktur/ekspresi (tak hasilkan faktor / semua di-drop korelasi)
  · metrics     : FactorIC_mean (kekuatan) + FactorICIR_mean (stabilitas)
Reason code merangkum lapisan mana yang gagal, lalu di-inject ke prompt proposal
sebagai daftar "AVOID" agar generasi berikutnya menghindari mekanisme itu.

Persistensi: JSON di samping trajectory pool. Kompak (≤ max_items entri yang
dirender), di-dedup per mekanisme (kunci prefix-ternormalisasi hipotesis).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from log import logger

try:  # operator-family util (sudah dipakai controller.get_best_trajectories)
    from latent_mas.operator_families import trajectory_families
except Exception:  # pragma: no cover — jaga modul tetap import tanpa latent_mas
    def trajectory_families(_factors):  # type: ignore
        return set()


# Reason code — lapisan kegagalan (urut dari paling informatif untuk AVOID).
REASON_WEAK_IC = "weak_ic"            # metrik: IC ≤ ambang (sinyal lemah/anti-prediktif)
REASON_UNSTABLE_ICIR = "unstable_icir"  # metrik: IC ok tapi ICIR ≤ ambang (noise)
REASON_REDUNDANT = "redundant"        # ekspresi: semua faktor di-drop correlation gate
REASON_NO_FACTOR = "no_factor"        # ekspresi: tak ada faktor lolos (gate/empty)
REASON_NO_METRIC = "no_metric"        # metrik: ada faktor tapi IC tak terhitung (backtest gagal)

_REASON_LABEL = {
    REASON_WEAK_IC: "weak signal",
    REASON_UNSTABLE_ICIR: "unstable",
    REASON_REDUNDANT: "redundant",
    REASON_NO_FACTOR: "no valid factor",
    REASON_NO_METRIC: "no metric",
}


def _norm_key(text: str, n: int = 120) -> str:
    """Kunci dedup mekanisme: huruf-kecil alfanumerik, n karakter pertama.
    Kasar tapi cukup untuk menggabungkan hipotesis yang pada dasarnya sama."""
    return re.sub(r"[^a-z0-9]+", "", (text or "").lower())[:n]


@dataclass
class NegativeEntry:
    """Satu mekanisme yang gagal (hasil evaluasi menyeluruh sebuah trajectory)."""
    hypothesis: str                      # ringkasan mekanisme (untuk tampilan AVOID)
    reason: str                          # reason code lapisan kegagalan
    ic: Optional[float] = None           # FactorIC_mean saat gagal
    icir: Optional[float] = None         # FactorICIR_mean saat gagal
    families: list[str] = field(default_factory=list)  # family operator yang dipakai
    round_idx: int = -1
    phase: str = ""
    count: int = 1                       # berapa kali mekanisme serupa gagal
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "NegativeEntry":
        d = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**d)


class NegativeMemory:
    """Kumpulan mekanisme gagal lintas-generasi + render hint AVOID untuk proposal."""

    def __init__(
        self,
        save_path: Optional[Path] = None,
        *,
        max_items: int = 8,
        enabled: bool = True,
        fresh_start: bool = True,
    ) -> None:
        self.save_path = Path(save_path) if save_path else None
        self.max_items = max_items
        self.enabled = enabled
        self._entries: dict[str, NegativeEntry] = {}  # norm_key → entry

        if not enabled:
            return
        if not fresh_start and self.save_path and self.save_path.exists():
            self._load()
        elif fresh_start and self.save_path and self.save_path.exists():
            logger.info(f"Fresh start: ignoring existing negative memory at {self.save_path}")

    # ── klasifikasi kegagalan (evaluasi menyeluruh) ──────────────────────────
    @staticmethod
    def classify(trajectory: Any, thr_ic: float = 0.0, thr_icir: float = 0.0) -> Optional[str]:
        """Tentukan reason code kegagalan trajectory; None bila SUKSES (bukan
        kandidat negatif). Memeriksa lapisan ekspresi DULU (tak ada faktor / semua
        redundan) lalu lapisan metrik (IC lemah / ICIR tak stabil / metrik hilang).
        """
        # is_successful() = gate dua-syarat (IC>thr AND ICIR>thr). Sukses → bukan negatif.
        try:
            if trajectory.is_successful(thr_ic, thr_icir):
                return None
        except Exception:
            pass

        metrics = getattr(trajectory, "backtest_metrics", {}) or {}
        extra = getattr(trajectory, "extra_info", {}) or {}
        factors = getattr(trajectory, "factors", None) or []

        # Lapisan EKSPRESI: tak ada faktor sama sekali → gate/empty.
        if not factors:
            return REASON_NO_FACTOR
        # Semua faktor yang ada justru di-drop correlation gate → redundan.
        dropped = extra.get("correlation_dropped") or []
        if dropped and len(dropped) >= len(factors):
            return REASON_REDUNDANT

        # Lapisan METRIK.
        ic = metrics.get("FactorIC_mean")
        if ic is None:
            ic = metrics.get("RankIC")  # backward-compat trajectory lama
        if ic is None:
            return REASON_NO_METRIC      # ada faktor tapi backtest tak hasilkan IC
        if ic <= thr_ic:
            return REASON_WEAK_IC
        icir = metrics.get("FactorICIR_mean")
        if icir is not None and icir <= thr_icir:
            return REASON_UNSTABLE_ICIR
        # Lolos semua pemeriksaan negatif tapi is_successful() False (mis. metrik
        # hilang sebagian) → jangan rekam sebagai negatif yang menyesatkan.
        return None

    # ── rekam ────────────────────────────────────────────────────────────────
    def record(self, trajectory: Any, thr_ic: float = 0.0, thr_icir: float = 0.0) -> Optional[str]:
        """Rekam trajectory bila gagal. Mengembalikan reason code (atau None bila
        sukses/dinonaktifkan). Mekanisme serupa di-merge (count++)."""
        if not self.enabled:
            return None
        reason = self.classify(trajectory, thr_ic, thr_icir)
        if reason is None:
            return None

        hyp = (getattr(trajectory, "hypothesis", "") or "").strip()
        key = _norm_key(hyp) or f"_{getattr(trajectory, 'trajectory_id', id(trajectory))}"
        metrics = getattr(trajectory, "backtest_metrics", {}) or {}
        ic = metrics.get("FactorIC_mean", metrics.get("RankIC"))
        icir = metrics.get("FactorICIR_mean")
        fams = sorted(trajectory_families(getattr(trajectory, "factors", []) or []))

        existing = self._entries.get(key)
        if existing is not None:
            # mekanisme yang sama gagal lagi → kuatkan, segarkan metrik terbaru
            existing.count += 1
            existing.reason = reason
            existing.ic = ic
            existing.icir = icir
            if fams:
                existing.families = fams
        else:
            self._entries[key] = NegativeEntry(
                hypothesis=hyp[:240],
                reason=reason,
                ic=float(ic) if isinstance(ic, (int, float)) else None,
                icir=float(icir) if isinstance(icir, (int, float)) else None,
                families=fams,
                round_idx=int(getattr(trajectory, "round_idx", -1) or -1),
                phase=str(getattr(trajectory, "phase", "")),
            )
        if self.save_path:
            self._save()
        return reason

    # ── render hint untuk prompt proposal ─────────────────────────────────────
    def render_hint(self, max_items: Optional[int] = None) -> str:
        """Blok teks "AVOID" untuk disuntik ke prompt proposal. Kosong bila tak
        ada kegagalan terekam atau memori dinonaktifkan. Diurut: paling sering
        gagal dulu, lalu paling baru. Setiap baris ringkas (altitude mekanisme)."""
        if not self.enabled or not self._entries:
            return ""
        n = max_items if max_items is not None else self.max_items
        entries = sorted(
            self._entries.values(),
            key=lambda e: (e.count, e.created_at),
            reverse=True,
        )[:n]

        lines = [
            "Avoid these mechanism frameworks — they ALREADY FAILED in earlier "
            "rounds. Do NOT re-propose them or minor variants:",
        ]
        for e in entries:
            tag = _REASON_LABEL.get(e.reason, e.reason)
            metric_bits = []
            if e.ic is not None:
                metric_bits.append(f"IC={e.ic:.3f}")
            if e.icir is not None:
                metric_bits.append(f"ICIR={e.icir:.2f}")
            metric_str = f", {', '.join(metric_bits)}" if metric_bits else ""
            rep = f" x{e.count}" if e.count > 1 else ""
            hyp = e.hypothesis.strip().replace("\n", " ")
            if len(hyp) > 140:
                hyp = hyp[:140].rstrip() + "…"
            lines.append(f"  - [{tag}{metric_str}{rep}] {hyp}")
        lines.append(
            "Propose a mechanism that is structurally DIFFERENT from every item above."
        )
        return "\n".join(lines)

    # ── persistensi ────────────────────────────────────────────────────────────
    def _save(self) -> None:
        if not self.save_path:
            return
        try:
            self.save_path.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "entries": {k: e.to_dict() for k, e in self._entries.items()},
                "saved_at": datetime.now().isoformat(),
            }
            with open(self.save_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Failed to save negative memory: {e}")

    def _load(self) -> None:
        if not self.save_path or not self.save_path.exists():
            return
        try:
            with open(self.save_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self._entries = {
                k: NegativeEntry.from_dict(v)
                for k, v in data.get("entries", {}).items()
            }
            logger.info(f"Loaded {len(self._entries)} negative-memory entries from {self.save_path}")
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Failed to load negative memory: {e}")

    def get_statistics(self) -> dict[str, Any]:
        by_reason: dict[str, int] = {}
        for e in self._entries.values():
            by_reason[e.reason] = by_reason.get(e.reason, 0) + 1
        return {"total_entries": len(self._entries), "by_reason": by_reason}

    def clear(self) -> None:
        self._entries.clear()
