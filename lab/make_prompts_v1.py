"""Bangun `prompts_v1.yaml` = prompts produksi + koreksi cacat DOKUMENTASI DSL.

Dipakai untuk menjawab pertanyaan: mutu ekspresi jelek karena MODELNYA lemah,
atau karena PROMPT-nya sendiri menyesatkan? Supaya jawabannya bisa
diatribusikan, v1 HANYA menyentuh kalimat yang oleh AUDIT_KRITIS §2.2/§S6
terbukti menanam kesalahan pada model — bukan perombakan prompt.

Tiap patch di bawah menyebut cacat empiris yang disasarnya. Semuanya di-assert
supaya perubahan diam-diam pada prompts.yaml langsung ketahuan.

    python lab/make_prompts_v1.py
"""
from __future__ import annotations

from pathlib import Path

BACKEND = Path(__file__).resolve().parent.parent / "backend"
SRC = BACKEND / "latent_mas" / "prompts.yaml"
DST = BACKEND / "latent_mas" / "prompts_v1.yaml"

# (deskripsi, lama, baru, jumlah kemunculan yang diharapkan)
PATCHES: list[tuple[str, str, str, int]] = [
    (
        # AUDIT §2.2: `TS_RANK(...) < 50` muncul 7x di batch. Prompt tak pernah
        # menyebut keluarannya persentil → modelnya menebak skala 0..100.
        "TS_RANK: sebut persentil [0,1] eksplisit",
        "        TS_RANK(A, n) time-series rank of the last value of A in the past n days.",
        "        TS_RANK(A, n) time-series percentile rank of the last value of A within\n"
        "          the past n days; the result is a fraction between 0 and 1, so compare it\n"
        "          with 0.8, never with 50.",
        3,
    ),
    (
        "RANK: sebut persentil [0,1] eksplisit",
        "        RANK(A) rank of A across all stocks today.",
        "        RANK(A) cross-sectional percentile rank of A across all stocks today;\n"
        "          the result is a fraction between 0 and 1.",
        3,
    ),
    (
        # AUDIT §2.2: 25 dari 86 ekspresi memakai skor kontinu sebagai syarat
        # ternary → cabang lain mati. Prompt lama hanya memberi CONTOH, tak
        # melarang bentuk yang salah.
        "ternary: syarat wajib perbandingan eksplisit",
        "        (C) ? (A) : (B)  if condition C holds then A, otherwise B. C is a logical\n"
        "          expression such as $close > $open.",
        "        (C) ? (A) : (B)  if condition C holds then A, otherwise B. C MUST be an\n"
        "          explicit comparison or logical test such as $close > $open. A bare score\n"
        "          is NOT a condition: in TS_ZSCORE($volume, 10) ? A : B every non-zero\n"
        "          value counts as true, so B is never taken. Write\n"
        "          (TS_ZSCORE($volume, 10) > 1) ? A : B instead.",
        3,
    ),
    (
        # AUDIT §2.2: 25 window degenerate (TS_ZSCORE(.,1) → 100% NaN,
        # TS_RANK(.,1) → konstan 1.0). Prompt construct malah MENGIZINKAN
        # "windows 1..60".
        "construct Step 4: larang window 1",
        "        Step 4 — Check: only the six variables; argument counts exact; windows 1..60;",
        "        Step 4 — Check: only the six variables; argument counts exact; every window\n"
        "                 is at least 2 and at most 60, and at least 5 for statistics that\n"
        "                 need spread (TS_STD, TS_VAR, TS_ZSCORE, TS_CORR, TS_COVARIANCE,\n"
        "                 TS_MAD) — a window of 1 gives a constant or NaN column;",
        1,
    ),
    (
        # AUDIT §2.2: 5 ekspresi memakai ambang absolut pada $volume
        # (`TS_MEAN($volume,10) < 500000`) → tak sebanding lintas saham.
        # Sekaligus tempat menaruh larangan window-1 di sisi design & construct.
        "RULES: larang ambang absolut $volume + window 1 + markdown",
        "        - Every expression contains at least one variable.",
        "        - Every expression contains at least one variable.\n"
        "        - Never compare a raw $volume quantity, or a rolling mean of it, with an\n"
        "          absolute number such as 500000: volume levels are not comparable across\n"
        "          stocks or across time. Normalise first, for example\n"
        "          TS_ZSCORE($volume, 20) > 2 or RANK($volume) > 0.8.\n"
        "        - A window of 1 is illegal: TS_ZSCORE(A, 1) is NaN and TS_RANK(A, 1) is the\n"
        "          constant 1. Use at least 2, and at least 5 where a spread is needed.\n"
        "        - Write plain ASCII. Do not use markdown emphasis (no ** **), headings, or\n"
        "          code fences anywhere in your answer.",
        2,
    ),
    (
        # catatan.txt: satu simbol = satu makna. "/" di prosa berbenturan dengan
        # "/" sebagai pembagian di dalam ekspresi.
        "hilangkan '/' sebagai kata hubung di prosa",
        "the conditional question-mark / colon pair",
        "the conditional question-mark and colon pair",
        2,
    ),
]


def main() -> None:
    text = SRC.read_text()
    for desc, old, new, n_expected in PATCHES:
        n = text.count(old)
        if n != n_expected:
            raise SystemExit(
                f"PATCH GAGAL [{desc}]: menemukan {n} kemunculan, "
                f"diharapkan {n_expected}. prompts.yaml sudah berubah?"
            )
        text = text.replace(old, new)
        print(f"  ok  ({n}x)  {desc}")

    header = (
        "# ===========================================================================\n"
        "# prompts_v1.yaml — DIBANGKITKAN oleh lab/make_prompts_v1.py. JANGAN diedit\n"
        "# tangan; ubah daftar PATCHES di skrip itu lalu jalankan ulang.\n"
        "#\n"
        "# = prompts.yaml + koreksi cacat dokumentasi DSL yang terbukti menanam\n"
        "#   kesalahan pada model (AUDIT_KRITIS §2.2 dan §S6):\n"
        "#     RANK/TS_RANK persentil [0,1] · syarat ternary harus perbandingan ·\n"
        "#     window >= 2 (>= 5 utk statistik sebaran) · larangan ambang absolut\n"
        "#     $volume · satu simbol satu makna · tanpa markdown.\n"
        "# Tak ada perubahan lain: peran agen, format keluaran, dan alurnya identik,\n"
        "# supaya selisih hasil bisa diatribusikan ke koreksi itu saja.\n"
        "# ===========================================================================\n"
    )
    DST.write_text(header + text)
    import yaml
    spec = yaml.safe_load(DST.read_text())
    assert set(spec["agents"]) == set(yaml.safe_load(SRC.read_text())["agents"])
    print(f"\ntersimpan → {DST}  ({len(spec['agents'])} agen, YAML valid)")


if __name__ == "__main__":
    main()
