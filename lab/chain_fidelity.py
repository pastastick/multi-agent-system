"""Sumbu A7 — KESETIAAN RANTAI: apakah agen benar-benar saling berkomunikasi?

Tiga uji deterministik (tanpa LLM tambahan) atas artefak `lab/out/frontend_*.json`:

  1. fidelity hipotesis→ekspresi — variabel & horizon yang DISEBUT hipotesis vs
     yang DIPAKAI ekspresi. AUDIT §2.1 menemukan hipotesis "small-cap … high
     volume" yang berakhir jadi `-RANK($volume)`; sumbu ini menangkapnya otomatis.
  2. kepatuhan palette — fraksi fungsi di ekspresi yang berasal dari palette yang
     dipilih agen `design`. Kalau rendah, `design` adalah hiasan. Butuh teks
     design tersimpan di `agent_trace` (lihat frontend_probe.instrument).
  3. rank-equivalence ke kolom mentah — |Spearman| lintas-penampang harian faktor
     terhadap $volume/$close/($high−$low)/... Kalau ≥ 0,99 faktor itu kolom mentah
     yang disamarkan (AUDIT §2.5 menjadi gate rutin, bukan temuan sekali jalan).

Dipakai sebagai library (`annotate_runs`) oleh `lab/analyze_gpu.py`, atau CLI:

    python lab/chain_fidelity.py --glob 'frontend_g4_*.json' --by comm_mode
    python lab/chain_fidelity.py --glob 'frontend_a8_*.json' --by chain --no-rank-equiv
"""
from __future__ import annotations

import argparse
import json
import re
import statistics as st
import sys
from pathlib import Path

QL = Path(__file__).resolve().parent.parent
if str(QL) not in sys.path:
    sys.path.insert(0, str(QL))

OUT = QL / "lab" / "out"

# ── kosakata ────────────────────────────────────────────────────────────────
VARS = ("open", "high", "low", "close", "volume", "return")

# Kata yang menandai satu variabel di PROSA hipotesis. Sengaja konservatif:
# hanya kata yang tak punya makna lain di kalimat keuangan. "return" muncul di
# hampir semua hipotesis sebagai objek prediksi ("predicts next-period returns"),
# jadi ia hanya dihitung bila muncul sebagai $return atau sebagai driver eksplisit
# ("return series", "past returns", "daily return").
#
# PENTING — kenapa pola untuk high/low/close/open sesempit ini. Versi pertama
# memakai `\bhigh\b` / `\blow\b` dan salah menghitung "LOW-volatility stocks" dan
# "abnormally HIGH-volume days" sebagai penyebutan kolom $low / $high. Di sana
# "high"/"low" adalah kata sifat untuk besaran LAIN, bukan harga tertinggi/
# terendah hari itu. Positif-palsu semacam ini akan menurunkan var_recall secara
# artifisial dan membuat sistem tampak lebih tidak setia daripada kenyataannya —
# persis jenis kesalahan yang tak boleh masuk skripsi. Karena itu kolom harga
# hanya dihitung bila prosa benar-benar menunjuk harga.
_VAR_PROSE = {
    "open":   (r"\$open\b", r"\bopen(?:ing)?\s+price\b", r"\bthe\s+open\b"),
    "high":   (r"\$high\b", r"\b(?:daily|intraday|session)\s+high\b",
               r"\bhigh\s+price\b", r"\bhighs\b", r"\bhigh[-\s]?low\b"),
    "low":    (r"\$low\b", r"\b(?:daily|intraday|session)\s+low\b",
               r"\blow\s+price\b", r"\blows\b", r"\bhigh[-\s]?low\b"),
    "close":  (r"\$close\b", r"\bclos(?:e|ing)\s+price\b", r"\bclosing\b",
               r"\bclose[-\s]to[-\s]close\b"),
    "volume": (r"\$volume\b", r"\bvolume\b", r"\bturnover\b"),
    "return": (r"\$return\b", r"\b(?:daily|past|lagged|trailing)\s+returns?\b",
               r"\breturns?\s+series\b"),
}

# Pita horizon dari prompt proposal (LEVEL OF DETAIL): short 1–10, medium 10–30,
# long 30–60 hari. Dipakai untuk menilai apakah window di ekspresi cocok dengan
# horizon yang dinyatakan hipotesis.
HORIZON_BANDS = {"short": (1, 10), "medium": (10, 30), "long": (30, 60)}
_HORIZON_WORDS = {
    "short": (r"\bshort[- ]?(?:term|horizon|window)\b", r"\bintraday\b",
              r"\bnext[- ]day\b", r"\bshort\b"),
    "medium": (r"\bmedium[- ]?(?:term|horizon|window)\b", r"\bintermediate\b"),
    "long": (r"\blong[- ]?(?:term|horizon|window)\b", r"\bextended\b"),
}

_FUNC_RE = re.compile(r"\b([A-Z][A-Z0-9_]{1,})\s*\(")
_VAR_RE = re.compile(r"\$([a-zA-Z_]+)")


# ── 1. fidelity hipotesis → ekspresi ────────────────────────────────────────

def vars_in_expression(expr: str) -> set[str]:
    return {v.lower() for v in _VAR_RE.findall(expr or "")}


def funcs_in_expression(expr: str) -> set[str]:
    return set(_FUNC_RE.findall(expr or ""))


def windows_in_expression(expr: str) -> list[int]:
    """Argumen numerik bulat yang berperan sebagai window (argumen ke-2+ dari
    sebuah pemanggilan fungsi). Angka di posisi pertama (mis. POW(A, 2)) ikut
    terbawa — itu diterima: yang diukur adalah SKALA WAKTU yang disentuh."""
    out: list[int] = []
    for m in re.finditer(r",\s*(\d+)\s*[,)]", expr or ""):
        out.append(int(m.group(1)))
    return out


def looks_degenerate(text: str) -> bool:
    """Teks yang KEHILANGAN SPASI — gejala nyata di comm_mode kv/kv_and_text:
    hipotesis keluar sebagai "Short-term reversalfollowingunusuallyhighvolume…".
    Ini bukan sekadar gangguan pengukuran; ia harus dilaporkan sebagai temuan,
    karena hipotesis yang cacat begini tak bisa dibaca agen hilir MAUPUN penguji."""
    t = (text or "").strip()
    if len(t) < 40:
        return False
    return t.count(" ") / len(t) < 0.05


def vars_in_hypothesis(hyp: str) -> set[str]:
    """Variabel yang DISEBUT hipotesis. Pada teks yang kehilangan spasi, batas
    kata (\\b) tak lagi ada, jadi pencocokan turun ke substring — kalau tidak,
    seluruh lengan kv akan tampak "tanpa variabel" padahal itu artefak alat."""
    h = (hyp or "").lower()
    found = set()
    for v, pats in _VAR_PROSE.items():
        if any(re.search(p, h) for p in pats):
            found.add(v)
    # Mekanisme yang SECARA DEFINISI berjalan di atas return masa lalu. Tanpa
    # aturan ini, "short-term reversal" dinilai tak menyebut $return, lalu setiap
    # ekspresi reversal dihukum var_precision rendah padahal ia setia. Ini
    # perbaikan substantif, bukan pelonggaran: reversal/momentum TIDAK bisa
    # dinyatakan tanpa deret return.
    if re.search(r"\b(revers\w*|momentum|mean[-\s]?revert\w*|overreact\w*|"
                 r"drift|continuation|trend[-\s]?follow\w*)\b", h):
        found.add("return")
    if looks_degenerate(hyp):
        for v in VARS:
            if v in ("return", "high", "low", "open", "close"):
                # Tanpa spasi, "highvolume" tak bisa dibedakan dari "dailyhigh",
                # dan "return" selalu muncul sebagai objek prediksi. Hanya
                # $volume yang tetap tak ambigu sebagai substring.
                continue
            if v in h:
                found.add(v)
        if "turnover" in h:
            found.add("volume")
    return found


def horizon_of_hypothesis(hyp: str) -> tuple[str | None, tuple[int, int] | None]:
    """Pita horizon yang dinyatakan hipotesis. Angka eksplisit ("over 5 days")
    menang atas kata sifat ("short-term"), karena lebih spesifik."""
    h = (hyp or "").lower()
    nums = [int(n) for n in re.findall(r"\b(\d{1,3})[- ]?(?:day|days|d)\b", h)]
    nums = [n for n in nums if 1 <= n <= 250]
    if nums:
        return "explicit", (min(nums), max(nums))
    for band, pats in _HORIZON_WORDS.items():
        if any(re.search(p, h) for p in pats):
            return band, HORIZON_BANDS[band]
    return None, None


def hypothesis_fidelity(hyp: str, expr: str) -> dict:
    """Kesetiaan satu ekspresi terhadap hipotesisnya.

    recall    = variabel hipotesis yang benar-benar dipakai ÷ variabel hipotesis.
                Rendah = ekspresi MENGABAIKAN sebagian mekanisme.
    precision = variabel ekspresi yang disebut hipotesis ÷ variabel ekspresi.
                Rendah = ekspresi MENAMBAH driver yang tak pernah dihipotesiskan.
    horizon_ok= ada ≥1 window di dalam pita horizon yang dinyatakan.
    """
    hv, ev = vars_in_hypothesis(hyp), vars_in_expression(expr)
    wins = windows_in_expression(expr)
    band, rng = horizon_of_hypothesis(hyp)
    horizon_ok = None
    if rng is not None and wins:
        lo, hi = rng
        # pita eksplisit diberi toleransi ±50% — hipotesis menyebut horizon, bukan
        # window persis; yang diuji adalah apakah SKALA-nya sama, bukan angkanya.
        if band == "explicit":
            lo, hi = max(1, int(lo * 0.5)), int(hi * 1.5) + 1
        horizon_ok = any(lo <= w <= hi for w in wins)
    return {
        "hyp_vars": sorted(hv), "expr_vars": sorted(ev),
        "hyp_degenerate": looks_degenerate(hyp),
        "hyp_empty": not (hyp or "").strip(),
        "var_recall": (len(hv & ev) / len(hv)) if hv else None,
        "var_precision": (len(hv & ev) / len(ev)) if ev else None,
        "horizon_band": band, "horizon_range": list(rng) if rng else None,
        "windows": wins, "horizon_ok": horizon_ok,
    }


# ── 2. kepatuhan palette ────────────────────────────────────────────────────

def palette_of_design(design_text: str) -> set[str] | None:
    """Nama fungsi yang di-shortlist agen design. Dibaca dari blok JSON
    ("palette": [{"function": ...}]); bila JSON tak lengkap (teks terpotong),
    jatuh ke regex atas field "function"."""
    if not design_text:
        return None
    m = re.search(r"\{[\s\S]*\"palette\"[\s\S]*\}", design_text)
    if m:
        try:
            doc = json.loads(m.group(0))
            names = {str(p.get("function", "")).strip().upper()
                     for p in (doc.get("palette") or []) if isinstance(p, dict)}
            names = {re.sub(r"\(.*$", "", n).strip() for n in names if n}
            if names:
                return names
        except Exception:  # noqa: BLE001 — teks LLM terpotong itu normal
            pass
    names = {n.strip().upper() for n in
             re.findall(r"\"function\"\s*:\s*\"([^\"]+)\"", design_text)}
    names = {re.sub(r"\(.*$", "", n).strip() for n in names if n}
    return names or None


def palette_compliance(palette: set[str] | None, expr: str) -> float | None:
    """Fraksi PEMANGGILAN fungsi di ekspresi yang namanya ada di palette.
    None bila palette tak tersedia (mis. mode kv: design tak menulis teks)."""
    if not palette:
        return None
    calls = _FUNC_RE.findall(expr or "")
    if not calls:
        return None
    return sum(1 for c in calls if c.upper() in palette) / len(calls)


# ── 3. rank-equivalence ke kolom mentah ─────────────────────────────────────

RAW_REFS = {
    "$volume": "$volume",
    "$close": "$close",
    "$return": "$return",
    "range": "($high - $low)",
    "$high": "$high",
    "$low": "$low",
    "$open": "$open",
}


class RankEquivalence:
    """|Spearman| lintas-penampang harian (dirata-rata) faktor vs kolom mentah.

    Perbandingan dilakukan PER HARI lintas saham — itulah ruang tempat faktor
    dipakai (ranking harian), jadi korelasi panel gabungan akan menyesatkan.
    """

    def __init__(self, lab=None):
        from lab.core import Lab
        self.lab = lab or Lab(mode="fast")
        self._refs: dict[str, object] = {}
        self._cache: dict[str, dict] = {}

    def _ref(self, expr: str):
        if expr not in self._refs:
            self._refs[expr] = self.lab.values(expr)
        return self._refs[expr]

    def of(self, expr: str) -> dict:
        if expr in self._cache:
            return self._cache[expr]
        import numpy as np
        import pandas as pd

        try:
            vals = self.lab.values(expr)
        except Exception as e:  # noqa: BLE001
            res = {"raw_equiv_max": None, "raw_equiv_col": None,
                   "raw_equiv_error": f"{type(e).__name__}: {e}"}
            self._cache[expr] = res
            return res

        best, best_col = None, None
        for name, ref_expr in RAW_REFS.items():
            try:
                ref = self._ref(ref_expr)
            except Exception:  # noqa: BLE001
                continue
            d = pd.DataFrame({"f": vals, "r": ref}).replace(
                [np.inf, -np.inf], np.nan).dropna()
            if d.empty:
                continue
            dts = d.index.get_level_values("datetime")
            d = d[(dts >= self.lab.oos_start) & (dts <= self.lab.oos_end)]
            if d.empty:
                continue
            per_day = d.groupby(level="datetime").apply(
                lambda x: x["f"].corr(x["r"], method="spearman") if len(x) > 2 else np.nan
            )
            v = per_day.abs().mean()
            if v == v and (best is None or v > best):
                best, best_col = float(v), name
        res = {"raw_equiv_max": best, "raw_equiv_col": best_col,
               "raw_equiv_error": None}
        self._cache[expr] = res
        return res


# ── anotasi artefak ─────────────────────────────────────────────────────────

def _design_text_of(run: dict) -> str:
    for t in (run.get("agent_trace") or []):
        if t.get("agent") == "design" and t.get("text"):
            return t["text"]
    return ""


def annotate_runs(runs: list[dict], rank_equiv: bool = True) -> None:
    """Tambahkan field A7 ke tiap faktor (in-place)."""
    re_calc = RankEquivalence() if rank_equiv else None
    for r in runs:
        hyp = r.get("hypothesis") or ""
        palette = palette_of_design(_design_text_of(r))
        for f in (r.get("factors") or []):
            e = f.get("expression", "")
            if not e:
                continue
            f.update(hypothesis_fidelity(hyp, e))
            f["palette_compliance"] = palette_compliance(palette, e)
            f["palette_size"] = len(palette) if palette else None
            if re_calc is not None:
                f.update(re_calc.of(e))


# ── CLI ─────────────────────────────────────────────────────────────────────

def _mean(vals) -> float:
    vals = [v for v in vals if v is not None]
    return st.mean(vals) if vals else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", nargs="+", default=["frontend_*.json"])
    ap.add_argument("--by", nargs="+", default=["comm_mode"])
    ap.add_argument("--no-rank-equiv", action="store_true",
                    help="lewati uji 3 (butuh evaluasi ekspresi di CPU, ~1 detik/ekspresi)")
    ap.add_argument("--write", action="store_true",
                    help="tulis balik anotasi ke file frontend_*.json")
    a = ap.parse_args()

    docs: list[tuple[Path, dict]] = []
    for pat in a.glob:
        for p in sorted(OUT.glob(pat)):
            docs.append((p, json.loads(p.read_text())))
    if not docs:
        sys.exit(f"tak ada file cocok: {a.glob} di {OUT}")

    runs: list[dict] = []
    for p, doc in docs:
        for r in doc["runs"]:
            r.setdefault("tag", doc["args"].get("tag"))
            runs.append(r)
    annotate_runs(runs, rank_equiv=not a.no_rank_equiv)
    if a.write:
        for p, doc in docs:
            p.write_text(json.dumps(doc, indent=2, default=str))
        print(f"anotasi ditulis ke {len(docs)} file")

    arms: dict[str, list[dict]] = {}
    for r in runs:
        key = " | ".join(f"{k}={r.get(k)}" for k in a.by)
        arms.setdefault(key, []).append(r)

    hdr = (f"{'lengan':<40s} {'expr':>5s} {'var_rec':>8s} {'var_prec':>9s} "
           f"{'horizon':>8s} {'palette':>8s} {'raw≈':>7s} {'raw≥.99':>8s}")
    print(f"\nA7 — kesetiaan rantai ({len(runs)} run)\n")
    print(hdr)
    print("-" * len(hdr))
    for name, rs in sorted(arms.items()):
        facs = [f for r in rs for f in (r.get("factors") or []) if f.get("expression")]
        hz = [f["horizon_ok"] for f in facs if f.get("horizon_ok") is not None]
        eq = [f["raw_equiv_max"] for f in facs if f.get("raw_equiv_max") is not None]
        print(f"{name:<40s} {len(facs):>5d} "
              f"{_mean(f.get('var_recall') for f in facs):>8.2f} "
              f"{_mean(f.get('var_precision') for f in facs):>9.2f} "
              f"{(sum(hz)/len(hz) if hz else float('nan')):>8.2f} "
              f"{_mean(f.get('palette_compliance') for f in facs):>8.2f} "
              f"{(_mean(eq) if eq else float('nan')):>7.2f} "
              f"{sum(1 for v in eq if v >= 0.99):>4d}/{len(eq):<3d}")

    print("\nvar_recall  = variabel hipotesis yang dipakai ekspresi (1,0 = setia)")
    print("var_prec    = variabel ekspresi yang disebut hipotesis (rendah = mengarang driver)")
    print("horizon     = fraksi ekspresi yang windownya di dalam pita horizon hipotesis")
    print("palette     = fraksi pemanggilan fungsi yang berasal dari palette design")
    print("            (nan = design tak menulis teks, mis. comm_mode=kv)")
    print("raw≈        = rata-rata |Spearman| harian ke kolom mentah TERDEKAT")
    print("raw≥.99     = jumlah faktor yang pada dasarnya kolom mentah menyamar")


if __name__ == "__main__":
    main()
