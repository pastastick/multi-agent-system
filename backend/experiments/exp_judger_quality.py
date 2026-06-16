#!/usr/bin/env python3
"""
experiments/exp_judger_quality.py
=================================
Eksperimen TERISOLASI untuk KUALITAS OUTPUT JUDGER per-skenario — bukan full
factor_mining/backtest. Tujuan: menangkap bug-bug kecil format/ekspresi yang
hanya muncul intermiten, dengan menjalankan tiap skenario BERULANG (N reps) lalu
menilai output judger pada kriteria yang disepakati:

  - parse        : parser hypothesis_exprs berhasil (≥1 ekspresi)
  - no-markdown  : judger tidak menulis **bold**/```fence```/bullet/blok prosa
  - no-repeat    : tidak ada n-gram berulang (repetition-collapse Qwen3)
  - complex      : ekspresi cukup kaya (symbol-length & jumlah node/operator)
  - gate         : LOLOS FactorRegulator PENUH (parsable+arity+var+complexity+dedup)
  - usable       : ≥1 ekspresi lolos gate → bisa dipakai backtest

Skenario (semua berakhir di JUDGER; tanpa backtest):
  original : proposal → construct → consistency → judger          (FrontEndPipeline.run)
  mutation : agent mutation(guidance) → re-entry → … → judger     (run_evolution, kind=mutation)
  crossover: agent crossover(guidance,k parent) → re-entry → judger(run_evolution, kind=crossover)

mutation & crossover butuh input FEEDBACK parent. Default: bangun parent nyata
via front-end original lalu lampirkan feedback+metrics SINTETIS (rekayasa). Bisa
override --feedback / --backtest-summary, atau --parents-json untuk memuat parent
(hypothesis/expression/feedback) dari sesi sebelumnya.

Catatan kondisi: default latent_steps=10 (PRODUKSI, pipeline/settings.py), BUKAN 0
seperti run_evolution.py — supaya reproduksi sesi yang gagal. Override --latent-steps.

Contoh
------
  V=/workspace/project/multi-agent-system/.venv/bin/python
  $V experiments/exp_judger_quality.py --reps 3 \
     --direction "high-volume reversal" \
     --scenarios original,mutation,crossover
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ── penilai kualitas (torch-free; bisa diuji terpisah) ───────────────────────

def _tokens(s: str) -> list[str]:
    """Tokenisasi kasar untuk DSL/teks: nama, $variabel, angka, simbol tunggal."""
    return re.findall(r"\$[A-Za-z_]+|[A-Za-z_]{2,}|\d+|[(),<>+\-*/]", s)


def _has_content(gram: tuple[str, ...]) -> bool:
    """Gram dianggap 'berisi' bila memuat ≥1 operator (UPPER≥2) atau $variabel.
    Tanpa filter ini, gram tanda-baca+angka seperti '( , 5 )' dari window yang
    SAH dipakai berulang (TS_ZSCORE($volume,5) … TS_ZSCORE($return,5) …) memicu
    false-positive — itu reuse window normal, BUKAN repetition-collapse."""
    return any(re.fullmatch(r"[A-Z][A-Z_]{1,}|\$[A-Za-z_]+", t) for t in gram)


def detect_repetition(text: str, n: int = 4, thresh: int = 3) -> tuple[bool, str]:
    """True bila ADA n-gram BERISI (memuat operator/var) yang muncul ≥thresh kali,
    ATAU sebuah gram-berisi yang langsung berdempetan (i dan i+n identik) — pola
    repetition-collapse 'DELTA($return,7) < DELTA($return,7) > DELTA($return,7) <'.
    Window literal yang dipakai ulang (', 5 )') TIDAK dihitung (lihat _has_content)."""
    toks = _tokens(text)
    if len(toks) < n:
        return False, ""
    gram_list = [tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)]
    grams = Counter(g for g in gram_list if _has_content(g))
    if grams:
        gram, cnt = grams.most_common(1)[0]
        if cnt >= thresh:
            return True, f"{' '.join(gram)} ×{cnt}"
    # duplikasi berdempetan (back-to-back) → pathology kuat walau cuma 2×
    for i in range(len(gram_list) - n):
        if gram_list[i] == gram_list[i + n] and _has_content(gram_list[i]):
            return True, f"{' '.join(gram_list[i])} (consecutive)"
    return False, ""


def detect_markdown(text: str) -> list[str]:
    """Tanda model 'menghias' alih-alih emit baris polos (akar bug format)."""
    flags = []
    if "**" in text:
        flags.append("bold")
    if "```" in text or "`" in text:
        flags.append("fence/backtick")
    if re.search(r"(?m)^\s*[-*•]\s+\w", text):
        flags.append("bullet")
    if re.search(r"(?mi)^\s*\**\s*(reasoning|mechanism|signal|horizon|note)\s*\**\s*:", text):
        flags.append("prose-section")
    return flags


def operators_in(expr: str) -> list[str]:
    """Operator (token UPPER ≥2 huruf) yang dipakai — proksi kekayaan ekspresi."""
    ops = set(re.findall(r"\b([A-Z][A-Z_]{1,})\b", expr))
    ops.discard("AND")
    ops.discard("OR")
    return sorted(ops)


@dataclass
class RepResult:
    scenario: str
    rep: int
    ok: bool = False                 # lulus SEMUA kriteria
    parse_ok: bool = False
    n_candidates: int = 0
    n_passing: int = 0
    usable: bool = False
    repaired: bool = False
    repetition: str = ""
    markdown: list[str] = field(default_factory=list)
    min_symbol_len: Optional[int] = None
    max_nodes: Optional[int] = None
    distinct_ops: int = 0
    gate_error: str = ""
    expressions: list[str] = field(default_factory=list)
    hypothesis: str = ""
    duration_s: float = 0.0
    error: str = ""

    def verdict(self) -> str:
        bits = []
        bits.append("parse" if self.parse_ok else "PARSE✗")
        bits.append("gate" if self.usable else "GATE✗")
        if self.repetition:
            bits.append("REPEAT✗")
        if self.markdown:
            bits.append("MD✗(" + ",".join(self.markdown) + ")")
        if self.repaired:
            bits.append("repaired")
        return " ".join(bits)


# kriteria 'complex' minimal — di bawah ini dianggap terlalu sepele
MIN_SYMBOL_LEN = 6      # < ini: ekspresi terlalu pendek/trivial
MIN_DISTINCT_OPS = 1    # minimal 1 operator nyata (bukan cuma $var perbandingan)


def score_output(scenario: str, rep: int, out: Any, regulator: Any,
                 t: float) -> RepResult:
    """Skor FrontEndOutput pada kriteria kualitas judger."""
    r = RepResult(scenario=scenario, rep=rep, duration_s=t)
    raw = out.judger_text or ""
    r.hypothesis = (out.hypothesis or "").strip()
    r.repaired = bool(out.repaired)
    r.gate_error = out.gate_error or ""
    r.expressions = list(out.expressions or [])
    r.n_passing = len(r.expressions)
    r.usable = r.n_passing >= 1

    # format judger MENTAH (sebelum parser membersihkan) → tangkap drift format
    rep_flag, rep_what = detect_repetition(raw)
    # juga cek repetisi DI DALAM ekspresi lolos (kalau gate kebobolan)
    for e in r.expressions:
        f, w = detect_repetition(e)
        if f:
            rep_flag, rep_what = True, w
            break
    r.repetition = rep_what if rep_flag else ""
    r.markdown = detect_markdown(raw)

    # parse_ok: judger menghasilkan hipotesis + ≥1 kandidat yang bisa diparse.
    # n_candidates di-recompute via parser agar terlihat berapa yang DIHASILKAN
    # (sebelum gate), bukan cuma yang lolos.
    try:
        from latent_mas.parsers import parse_hypothesis_exprs
        he = parse_hypothesis_exprs(raw)
        r.parse_ok = he is not None and len(he.expressions) >= 1
        r.n_candidates = len(he.expressions) if he else 0
    except Exception as e:  # noqa: BLE001
        r.error = f"parse:{e!r}"

    # kompleksitas via regulator (symbolic, tanpa data) untuk ekspresi lolos
    sls, nodes, ops_all = [], [], set()
    if regulator is not None:
        for e in r.expressions:
            try:
                ok, ev = regulator.evaluate(e)
                if ok and ev:
                    sls.append(int(ev.get("symbol_length") or 0))
                    nodes.append(int(ev.get("num_all_nodes") or 0))
            except Exception:
                pass
            ops_all.update(operators_in(e))
    else:
        for e in r.expressions:
            ops_all.update(operators_in(e))
    r.min_symbol_len = min(sls) if sls else None
    r.max_nodes = max(nodes) if nodes else None
    r.distinct_ops = len(ops_all)

    complex_ok = (
        r.n_passing >= 1
        and (r.min_symbol_len is None or r.min_symbol_len >= MIN_SYMBOL_LEN)
        and r.distinct_ops >= MIN_DISTINCT_OPS
    )
    r.ok = bool(
        r.parse_ok and r.usable and not r.repetition
        and not r.markdown and complex_ok
    )
    return r


# ── runner skenario ──────────────────────────────────────────────────────────

def _parent_block(out, *, feedback: str, backtest: str, label: str = "") -> str:
    """FrontEndOutput → TEKS parent (mirror loop._format_parents_text)."""
    exprs = out.expressions or ([out.expression] if out.expression else [])
    parts = []
    if out.hypothesis:
        parts.append(f"Hypothesis: {out.hypothesis}")
    if exprs:
        parts.append("Expression(s):\n" + "\n".join(f"  - {e}" for e in exprs))
    if backtest:
        parts.append(f"Backtest: {backtest}")
    if feedback:
        parts.append(f"Feedback: {feedback}")
    block = "\n".join(parts) or "(empty)"
    return f"[{label}]\n{block}" if label else block


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenarios", default="original,mutation,crossover",
                    help="comma: original,mutation,crossover")
    ap.add_argument("--reps", type=int, default=3, help="pengulangan per skenario")
    ap.add_argument("--direction", default="high-volume days precede short-horizon reversal")
    ap.add_argument("--direction2", default="overnight gap mean-reversion",
                    help="arah parent-2 untuk crossover")
    ap.add_argument("--feedback", default="Low RankIC (~0.01). Mechanism too "
                    "volume-dominated and fragile under noise; needs a distinct signal family.",
                    help="feedback parent SINTETIS untuk mutation/crossover")
    ap.add_argument("--backtest-summary",
                    default="IC=0.010 RankIC=0.024 annualized_return=3.5% max_drawdown=-9.5%")
    ap.add_argument("--parents-json", default=None,
                    help="JSON [{hypothesis,expression,feedback}] dari sesi lampau "
                         "(override pembuatan parent via front-end)")
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--latent-steps", type=int, default=10,
                    help="DEFAULT 10 (produksi). 0 = tanpa latent.")
    ap.add_argument("--use-realign", action="store_true")
    ap.add_argument("--knn", action="store_true")
    ap.add_argument("--console", default="WARNING")
    ap.add_argument("--out-dir", default=None,
                    help="dir hasil (raw judger + summary.json). Default latent_runs/<ts>")
    args = ap.parse_args()

    scenarios = [s.strip() for s in args.scenarios.split(",") if s.strip()]

    from llm.client import LocalLLMBackend
    from latent_mas.runlog import get_run_logger
    from latent_mas.pipeline import FrontEndPipeline

    rl = get_run_logger(run_name="exp_judger_quality", console_level=args.console)
    out_dir = Path(args.out_dir) if args.out_dir else Path(rl.dir) / "judger_quality"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[exp] loading {args.model} on {args.device} "
          f"(latent_steps={args.latent_steps}) …", flush=True)
    backend = LocalLLMBackend(
        model_name=args.model, device=args.device,
        latent_steps=args.latent_steps, use_realign=args.use_realign,
        knn_enabled=args.knn,
    )
    # SATU pipeline → model + regulator dibangun sekali, reuse antar rep. Catatan:
    # diversity-hint & alpha-zoo dedup AKTIF lintas-rep (faithful ke sesi produksi);
    # ekspresi identik berulang akan ditolak gate sebagai 'dup' — itu sinyal
    # mode-collapse yang memang ingin terlihat, BUKAN bug penilai.
    pipe = FrontEndPipeline(backend, runlog=rl)
    regulator = pipe._regulator
    if regulator is None:
        print("[exp] WARNING: FactorRegulator tidak aktif → gate = syntax-only, "
              "metrik kompleksitas terbatas.", flush=True)

    # ── parent untuk mutation/crossover ──────────────────────────────────────
    parents_text_1 = parents_text_2 = ""
    need_evo = any(s in ("mutation", "crossover") for s in scenarios)
    if need_evo:
        if args.parents_json:
            data = json.loads(Path(args.parents_json).read_text())

            class _P:  # adapter ringan → _parent_block
                def __init__(s, d):
                    s.hypothesis = d.get("hypothesis", "")
                    s.expressions = ([d["expression"]] if d.get("expression")
                                     else d.get("expressions", []))

                @property
                def expression(s):
                    return s.expressions[0] if s.expressions else ""

            p1 = _P(data[0])
            p2 = _P(data[1]) if len(data) > 1 else p1
            fb1 = data[0].get("feedback", args.feedback)
            fb2 = data[1].get("feedback", args.feedback) if len(data) > 1 else fb1
        else:
            print("[exp] building parent-1 via front-end (original) …", flush=True)
            p1 = pipe.run(direction=args.direction)
            p2 = p1
            fb1 = fb2 = args.feedback
            if "crossover" in scenarios:
                print("[exp] building parent-2 via front-end (original) …", flush=True)
                p2 = pipe.run(direction=args.direction2)
        parents_text_1 = _parent_block(p1, feedback=fb1, backtest=args.backtest_summary)
        parents_text_2 = (
            _parent_block(p1, feedback=fb1, backtest=args.backtest_summary, label="Parent 1")
            + "\n\n" +
            _parent_block(p2, feedback=fb2, backtest=args.backtest_summary, label="Parent 2")
        )

    # ── jalankan reps ────────────────────────────────────────────────────────
    results: list[RepResult] = []
    for scen in scenarios:
        for rep in range(1, args.reps + 1):
            tag = f"{scen}#{rep}"
            print(f"\n[exp] ── {tag} ──────────────────────────────", flush=True)
            t0 = time.time()
            out = None
            try:
                if scen == "original":
                    out = pipe.run(direction=args.direction)
                elif scen == "mutation":
                    out = pipe.run_evolution(kind="mutation",
                                             parent_text=parents_text_1,
                                             direction=args.direction)
                elif scen == "crossover":
                    out = pipe.run_evolution(kind="crossover",
                                             parent_text=parents_text_2, n_parents=2,
                                             direction=args.direction)
                else:
                    raise SystemExit(f"skenario tak dikenal: {scen}")
                dt = time.time() - t0
                r = score_output(scen, rep, out, regulator, dt)
            except Exception as e:  # noqa: BLE001 — satu rep gagal tak menghentikan
                import traceback
                traceback.print_exc()
                r = RepResult(scenario=scen, rep=rep, duration_s=time.time() - t0,
                              error=repr(e))
            results.append(r)
            # simpan judger mentah untuk eyeball
            (out_dir / f"{scen}_{rep:02d}.txt").write_text(
                f"# {tag}  verdict: {r.verdict()}\n"
                f"# hypothesis: {r.hypothesis}\n"
                f"# expressions ({r.n_passing} pass / {r.n_candidates} cand): "
                f"{r.expressions}\n"
                f"# gate_error: {r.gate_error}\n\n"
                + (out.judger_text if (out is not None and not r.error)
                   else f"ERROR {r.error}")
            )
            print(f"  {r.verdict()}  | dur={r.duration_s:.1f}s  hyp={r.hypothesis[:70]!r}",
                  flush=True)
            for e in r.expressions:
                print(f"      ✓ {e}", flush=True)
            if r.repetition:
                print(f"      ⚠ repetition: {r.repetition}", flush=True)
            if r.gate_error:
                print(f"      ⚠ gate: {r.gate_error}", flush=True)

    # ── ringkasan ────────────────────────────────────────────────────────────
    print(f"\n{'='*78}\nSUMMARY ({args.model}, latent_steps={args.latent_steps}, "
          f"reps={args.reps})\n{'='*78}")
    hdr = f"{'scenario':<10} {'ok':>5} {'parse':>6} {'usable':>7} {'noRep':>6} {'noMD':>5} {'avg_s':>6}"
    print(hdr)
    print("-" * len(hdr))
    summary: dict[str, Any] = {"config": vars(args), "scenarios": {}}
    for scen in scenarios:
        rs = [r for r in results if r.scenario == scen]
        n = len(rs)
        def rate(pred):
            return sum(1 for r in rs if pred(r)) / n if n else 0.0
        ok = rate(lambda r: r.ok)
        parse = rate(lambda r: r.parse_ok)
        usable = rate(lambda r: r.usable)
        norep = rate(lambda r: not r.repetition)
        nomd = rate(lambda r: not r.markdown)
        avg = sum(r.duration_s for r in rs) / n if n else 0.0
        print(f"{scen:<10} {ok:>5.0%} {parse:>6.0%} {usable:>7.0%} "
              f"{norep:>6.0%} {nomd:>5.0%} {avg:>6.1f}")
        summary["scenarios"][scen] = {
            "n": n, "ok_rate": ok, "parse_rate": parse, "usable_rate": usable,
            "no_repetition_rate": norep, "no_markdown_rate": nomd, "avg_s": avg,
            "reps": [asdict(r) for r in rs],
        }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"\n[exp] raw outputs + summary.json → {out_dir}")


if __name__ == "__main__":
    main()
