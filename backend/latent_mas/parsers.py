"""
latent_mas/parsers.py
====================
Parser output teks agent. Sengaja permisif — model 4B sering menambah
penjelasan/markdown walau diminta satu baris. Tiap parser mengembalikan
struktur kecil yang gampang dicek, atau None bila benar-benar gagal.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional


# ── Judger / Mutation / Crossover: HYPOTHESIS + EXPRESSION ───────────────────

@dataclass
class HypothesisExpr:
    hypothesis: str
    expression: str


def parse_hypothesis_expr(raw: str) -> Optional[HypothesisExpr]:
    """Ambil 'HYPOTHESIS: ...' dan 'EXPRESSION: ...' dari output judger.

    Toleran terhadap:
      - label case-insensitive + terpotong (HYPOTHESIS/Hypothesis/hypo/HYPOTH/
        HYPOTHS, EXPRESSION/EXPR) — model 4B sering memenggal label,
      - hypothesis multi-baris sampai ketemu baris EXPRESSION,
      - expression dibungkus backtick/quote,
      - markdown fence,
      - tag <think>/</think> yatim yang lolos dari strip.
    """
    if not raw or not raw.strip():
        return None
    text = raw.strip()
    # buang markdown fence global + tag think yatim
    text = re.sub(r"```[a-zA-Z]*\n?", "", text).replace("```", "")
    text = re.sub(r"</?think>", "", text)

    # Label match longgar: 'hypo' diikuti word-char apa pun (hypo, hypoth,
    # hypoths, hypothesis) lalu ':'. Idem 'expr' (expr, expression).
    hyp_m = re.search(
        r"hypo\w*\s*:\s*(.+?)(?=\n\s*expr\w*\s*:|\Z)",
        text, flags=re.IGNORECASE | re.DOTALL,
    )
    expr_m = re.search(
        r"expr\w*\s*:\s*(.+?)\s*\Z",
        text, flags=re.IGNORECASE | re.DOTALL,
    )
    if not expr_m:
        return None
    expression = expr_m.group(1).strip()
    # expression: ambil baris non-kosong pertama (model sering menambah catatan)
    for line in expression.splitlines():
        if line.strip():
            expression = line.strip()
            break
    expression = _extract_code_span(expression)
    expression = _strip_wrappers(expression)
    expression = _balance_parens(expression)

    hypothesis = hyp_m.group(1).strip() if hyp_m else ""
    if not expression:
        return None
    return HypothesisExpr(hypothesis=hypothesis, expression=expression)


# ── Repair-or-pass: PASS | FIXED: <expr> ─────────────────────────────────────

PASS_SENTINEL = "__PASS__"


def parse_repair(raw: str) -> Optional[str]:
    """Kembalikan PASS_SENTINEL, ekspresi (string), atau None.

    Kontrak: satu baris — 'PASS' atau 'FIXED: <expression>'.
    """
    if not raw or not raw.strip():
        return None
    text = re.sub(r"</?think>", "", raw).strip()
    first = text.splitlines()[0].strip() if text.splitlines() else ""
    if re.fullmatch(r"pass[.!]?", first, flags=re.IGNORECASE):
        return PASS_SENTINEL
    kw = re.compile(r"^\s*(?:fixed|expr\w*|result)\s*:\s*(.+?)\s*$",
                    flags=re.IGNORECASE)
    for line in text.splitlines():
        if not line.strip():
            continue
        m = kw.match(line)
        if m:
            expr = _extract_code_span(m.group(1).strip())
            expr = _balance_parens(_strip_wrappers(expr))
            if expr:
                return expr
    # fallback JSON {"expr": "..."}
    m = re.search(r'"(?:expr|fixed|expression)"\s*:\s*"((?:[^"\\]|\\.)*)"', text)
    if m:
        return m.group(1).strip()
    return None


# ── Mutation reflection: FAILURE_STEP + REASON ───────────────────────────────

@dataclass
class MutationDiagnosis:
    failure_step: str          # propose | construct | consistency | expression | unknown
    reason: str


_VALID_STEPS = {"propose", "construct", "consistency", "expression", "unknown"}


def parse_mutation_diagnosis(raw: str) -> MutationDiagnosis:
    """Ambil 'FAILURE_STEP: ...' + 'REASON: ...'. Tidak pernah None —
    fallback ke ('construct', <teks mentah>) bila gagal parse, karena
    construct adalah titik revisi paling umum."""
    text = (raw or "").strip()
    step_m = re.search(r"failure_step\s*:\s*(\w+)", text, flags=re.IGNORECASE)
    reason_m = re.search(r"reason\s*:\s*(.+?)\s*\Z", text,
                         flags=re.IGNORECASE | re.DOTALL)
    step = (step_m.group(1).lower() if step_m else "construct")
    if step not in _VALID_STEPS:
        step = "construct"
    reason = reason_m.group(1).strip() if reason_m else text
    return MutationDiagnosis(failure_step=step, reason=reason)


# ── Introspect: bebas (teks apa adanya) ──────────────────────────────────────

def parse_passthrough(raw: str) -> str:
    return (raw or "").strip()


# ── helpers ──────────────────────────────────────────────────────────────────

def _extract_code_span(expr: str) -> str:
    """Bila ada span ber-backtick (`...`), ambil isinya — model 4B sering
    membungkus ekspresi dalam backtick lalu menambah catatan setelahnya."""
    m = re.search(r"`([^`]+)`", expr)
    return m.group(1).strip() if m else expr


def _strip_wrappers(expr: str) -> str:
    expr = expr.strip()
    for q in ("`", '"', "'"):
        if len(expr) >= 2 and expr.startswith(q) and expr.endswith(q):
            expr = expr[1:-1].strip()
    return expr


def _balance_parens(expr: str) -> str:
    """Potong di titik kelebihan ')' (model 4B sering menambah junk di akhir)."""
    depth = 0
    for i, ch in enumerate(expr):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth < 0:
                return expr[:i].rstrip()
    return expr


# registry untuk lookup by name dari YAML
PARSERS = {
    "hypothesis_expr": parse_hypothesis_expr,
    "repair": parse_repair,
    "mutation_diagnosis": parse_mutation_diagnosis,
    "passthrough": parse_passthrough,
    "none": None,
}
