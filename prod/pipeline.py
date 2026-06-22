"""prod/pipeline.py — orkestrator loop evolusi produksi (segmented KV).

Topologi (DESIGN.md §3). KV di-RESTART di feedback tiap generasi:
  gen 0      : proposal(seed) -> design -> construct            [KV segment A]
  gen g>=1   : feedback(text dari construct_{g-1}, KV FRESH)
               -> director(mutation|crossover) -> proposal -> design -> construct
  construct (terminal) -> runner{gate,repair,backtest} -> feedback generasi berikut.

NO-CROP: jawaban tiap agent tetap di KV (keep_answer_in_kv di prompts.yaml) sehingga
agent berikut membaca output ASLI dari KV. Mode 'text' = baseline A/B (kv none,
latent_steps 0, output upstream disuntik sebagai teks).
"""
from __future__ import annotations

import json
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml
from jinja2 import Environment, Undefined

from .config import FUNCTION_LIB, PROMPTS_YAML, RunConfig
from . import transfer as T
from . import runner as R
from .runlog import RunLog


class _Vis(Undefined):
    def __str__(self) -> str:
        return f"[[MISSING:{self._undefined_name}]]"


# ── node graph ───────────────────────────────────────────────────────────────

def build_nodes(cfg: RunConfig) -> List[dict]:
    """Daftar node berurutan-topologis lintas generasi (lihat docstring modul)."""
    nodes: List[dict] = []
    for g in range(cfg.generations):
        if g == 0:
            nodes += [
                {"id": "proposal_g0", "agent": "proposal", "stage": "proposal",
                 "gen": 0, "parents": [], "kv_root": True, "seed": True},
                {"id": "design_g0", "agent": "design", "stage": "design",
                 "gen": 0, "parents": ["proposal_g0"]},
                {"id": "construct_g0", "agent": "construct", "stage": "construct",
                 "gen": 0, "parents": ["design_g0"], "terminal": True},
            ]
        else:
            director = cfg.director
            nodes += [
                {"id": f"feedback_g{g}", "agent": "feedback", "stage": "feedback",
                 "gen": g, "parents": [], "kv_root": True,
                 "source_construct": f"construct_g{g - 1}"},
                {"id": f"{director}_g{g}", "agent": director, "stage": director,
                 "gen": g, "parents": [f"feedback_g{g}"]},
                {"id": f"proposal_g{g}", "agent": "proposal", "stage": "proposal",
                 "gen": g, "parents": [f"{director}_g{g}"]},
                {"id": f"design_g{g}", "agent": "design", "stage": "design",
                 "gen": g, "parents": [f"proposal_g{g}"]},
                {"id": f"construct_g{g}", "agent": "construct", "stage": "construct",
                 "gen": g, "parents": [f"design_g{g}"], "terminal": True},
            ]
    for o, n in enumerate(nodes):
        n["order"] = o
    return nodes


def transfer_label(node: dict, cfg: RunConfig) -> str:
    if cfg.transfer_mode == "text":
        return "none(text)"
    if node.get("kv_root"):
        return "none(fresh)"
    if not node["parents"]:
        return "none"
    return "concat" if len(node["parents"]) > 1 else "chain"


# ── vars per node ────────────────────────────────────────────────────────────

def _dry_feedback_vars() -> Dict[str, str]:
    return {
        "hypothesis": "(dry) HYPOTHESIS placeholder for previous construct",
        "factor_block": "FACTORS (dry):\n- f1: RANK(TS_MEAN($return, 5))",
        "backtest_results": "Block A (dry): f1 RankIC=0.03 ICIR=0.2\nBlock B (dry): RankIC=0.06 MaxDD=0.31",
        "sota_block": "none yet",
    }


def node_vars(node: dict, cfg: RunConfig, text_by_id: Dict[str, str],
              feedback_inputs: Dict[str, Dict[str, str]], *, dry: bool) -> Dict[str, str]:
    v: Dict[str, str] = {"handoff": cfg.handoff, "market_context": cfg.market_context}
    stage = node["stage"]

    if stage == "feedback":
        if dry:
            v.update(_dry_feedback_vars())
        else:
            v.update(feedback_inputs.get(node["id"], _dry_feedback_vars()))
        return v

    if node.get("seed"):  # gen-0 proposal: seed direction selalu sebagai teks
        v["handoff"] = "text"
        v["direction"] = cfg.seed_direction
        return v

    if cfg.transfer_mode != "text":
        return v  # kv: upstream via latent memory, tak ada var teks

    # ── mode text: suntik teks upstream hasil decode ──
    parent0 = node["parents"][0] if node["parents"] else ""
    if stage == "mutation":
        v["target_text"] = text_by_id.get(parent0, "")
    elif stage == "crossover":
        v["parents_text"] = "\n---\n".join(text_by_id.get(p, "") for p in node["parents"])
    elif stage == "proposal":
        v["direction"] = text_by_id.get(parent0, "")
    elif stage == "design":
        v["hypothesis_text"] = text_by_id.get(parent0, "")
    elif stage == "construct":
        v["prior_factors"] = text_by_id.get(parent0, "")
    return v


# ── rendering & artifacts ────────────────────────────────────────────────────

def _load_spec(agent: str) -> dict:
    raw = yaml.safe_load(Path(PROMPTS_YAML).read_text(encoding="utf-8"))
    return (raw.get("agents") or {})[agent]


def _render(spec: dict, vars_: dict) -> Tuple[str, str]:
    env = Environment(undefined=_Vis)
    system = env.from_string(spec.get("system", "")).render(**vars_).strip()
    user = env.from_string(spec.get("user", "")).render(**vars_).strip()
    return system, user


def _write_artifact(path: Path, header: dict, system: str, user: str,
                    response: str, extra: Optional[dict] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    hdr = "  ".join(f"{k}={v}" for k, v in header.items())
    parts = [f"# {hdr}", ""]
    if extra:
        parts += [f"# {json.dumps(extra, ensure_ascii=False)}", ""]
    bar = "=" * 78
    parts += [bar, "SYSTEM", bar, system, "",
              bar, "USER", bar, user, "",
              bar, "RESPONSE", bar, response]
    path.write_text("\n".join(parts), encoding="utf-8")


# ── pipeline ─────────────────────────────────────────────────────────────────

class EvolutionPipeline:
    def __init__(self, cfg: RunConfig, backend: Any = None) -> None:
        self.cfg = cfg
        self.backend = backend
        self.run_dir = cfg.out_dir / cfg.transfer_mode / f"ls{cfg.effective_latent_steps}"
        self.log = RunLog(enabled=cfg.verbose)
        self.sota_rankic: Optional[float] = None  # best standalone RankIC sejauh ini
        self._repair_agent = None                 # agent repair (lazy; butuh backend)

    # -- dry-run: render + cek wiring tanpa GPU --------------------------------
    def dry_run(self) -> dict:
        nodes = build_nodes(self.cfg)
        text_by_id: Dict[str, str] = {}
        out = {"mode": "dry", "transfer": self.cfg.transfer_mode,
               "latent_steps": self.cfg.effective_latent_steps, "nodes": [], "ok": True}
        for n in nodes:
            spec = _load_spec(n["agent"])
            vars_ = node_vars(n, self.cfg, text_by_id, {}, dry=True)
            system, user = _render(spec, vars_)
            missing = "[[MISSING:" in (system + user)
            out["ok"] = out["ok"] and not missing
            tlabel = transfer_label(n, self.cfg)
            rec = {"id": n["id"], "order": n["order"], "agent": n["agent"],
                   "gen": n["gen"], "transfer": tlabel,
                   "no_crop": bool(spec.get("keep_answer_in_kv", True)),
                   "missing_vars": missing, "parents": n["parents"]}
            out["nodes"].append(rec)
            _write_artifact(
                self.run_dir / f"{n['order']:02d}_{n['id']}.txt",
                {"mode": "dry", "transfer": tlabel, "node": n["id"],
                 "agent": n["agent"], "gen": n["gen"],
                 "no_crop": rec["no_crop"]},
                system, user, "(dry_run — tidak call LLM)", extra=rec,
            )
            text_by_id[n["id"]] = f"(dry decoded {n['id']})"
        (self.run_dir).mkdir(parents=True, exist_ok=True)
        (self.run_dir / "run.json").write_text(
            json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        return out

    # -- real run (GPU) -------------------------------------------------------
    def run(self) -> dict:
        if self.backend is None:
            raise RuntimeError("run() butuh backend; pakai dry_run() tanpa GPU.")
        from latent_mas.kv_ops import kv_seq_len
        from .agents import load_prod_agent

        nodes = build_nodes(self.cfg)
        kv_by_id: Dict[str, Any] = {}
        text_by_id: Dict[str, str] = {}
        feedback_inputs: Dict[str, Dict[str, str]] = {}
        out = {"mode": "real", "transfer": self.cfg.transfer_mode,
               "latent_steps": self.cfg.effective_latent_steps,
               "backtest": self.cfg.backtest_mode, "nodes": [], "ok": True, "err": None}
        self.log.section(f"RUN transfer={self.cfg.transfer_mode} "
                         f"ls={self.cfg.effective_latent_steps} gens={self.cfg.generations} "
                         f"backtest={self.cfg.backtest_mode} gate={R.gate_kind()}")
        t0 = time.time()
        try:
            for n in nodes:
                tlabel = transfer_label(n, self.cfg)
                # transfer KV (kv mode, non-root) — diukur terpisah dari "berpikir"
                t_xfer = 0.0
                input_kv = None
                if self.cfg.transfer_mode == "kv" and not n.get("kv_root") and n["parents"]:
                    _tx = time.time()
                    input_kv = T.transfer_kv([kv_by_id.get(p) for p in n["parents"]])
                    t_xfer = time.time() - _tx

                vars_ = node_vars(n, self.cfg, text_by_id, feedback_inputs, dry=False)
                ag = load_prod_agent(n["agent"], self.backend, self.cfg)
                try:
                    res = ag.run(past_kv=input_kv, **vars_)
                except RuntimeError as e:  # GPU OOM/collapse → warn + skip (DESIGN.md §6.1)
                    if "out of memory" in str(e).lower() or "cuda" in str(e).lower():
                        out["ok"] = False
                        out["err"] = f"GPU collapse di {n['id']}: {e}"
                        self.log.line(f"[WARN] GPU collapse di {n['id']} — skip sisa run.")
                        try:
                            import torch; torch.cuda.empty_cache()
                        except Exception:
                            pass
                        break
                    raise

                if self.cfg.transfer_mode == "kv":
                    kv_by_id[n["id"]] = res.kv_cache
                if res.text is not None:
                    text_by_id[n["id"]] = res.text

                kvlen = kv_seq_len(getattr(res, "kv_cache", None))
                self.log.agent(
                    gen=n["gen"], node=n["id"], agent=n["agent"], transfer=tlabel,
                    kv_xfer_s=t_xfer, latent_s=getattr(res, "latent_s", 0.0),
                    gen_s=getattr(res, "gen_s", 0.0), total_s=res.duration_s,
                    in_tok=res.n_input_tokens, out_tok=res.n_output_tokens,
                    kv_len=kvlen, latent_steps=getattr(res, "latent_steps", 0))

                # terminal construct → gate+repair+backtest → input feedback gen berikut
                score_detail = None
                if n.get("terminal"):
                    score_detail = self._score_and_prepare_feedback(n, res.text or "", feedback_inputs)

                spec = _load_spec(n["agent"])
                system, user = _render(spec, vars_)
                _write_artifact(
                    self.run_dir / f"{n['order']:02d}_{n['id']}.txt",
                    {"mode": "real", "transfer": tlabel, "node": n["id"],
                     "agent": n["agent"], "gen": n["gen"],
                     "no_crop": bool(spec.get("keep_answer_in_kv", True))},
                    system, user, res.text or "(kv_only)",
                    extra={"kv_seq_len": kvlen, "kv_xfer_s": round(t_xfer, 3),
                           "latent_s": getattr(res, "latent_s", 0.0),
                           "gen_s": getattr(res, "gen_s", 0.0),
                           "in_tok": res.n_input_tokens, "out_tok": res.n_output_tokens,
                           "score": score_detail},
                )
                out["nodes"].append({"id": n["id"], "agent": n["agent"],
                                     "transfer": tlabel, "kv_len": kvlen,
                                     "latent_s": getattr(res, "latent_s", 0.0),
                                     "gen_s": getattr(res, "gen_s", 0.0),
                                     "out_tok": res.n_output_tokens,
                                     "score": (score_detail or {}).get("score")})
            out["elapsed_s"] = round(time.time() - t0, 2)
            out["sota_rankic"] = self.sota_rankic
            self.log.section(f"DONE elapsed={out['elapsed_s']}s sota_rankic={self.sota_rankic}")
        except Exception as e:  # noqa: BLE001
            out["ok"] = False
            out["err"] = f"{type(e).__name__}: {e}"
            out["traceback"] = traceback.format_exc()[-1500:]
            self.log.line(f"[ERR] {out['err']}")

        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "run.json").write_text(
            json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        return out

    def _make_repair_fn(self):
        """Closure ke AGENT repair (standalone, NON-chained). Memperbaiki SATU
        ekspresi ilegal agar lolos gate tanpa mengubah intent (explanation Builder).
        Di-cache; mengembalikan None bila tak ada backend."""
        if self.backend is None:
            return None
        from .agents import load_prod_agent
        if self._repair_agent is None:
            self._repair_agent = load_prod_agent("repair", self.backend, self.cfg)

        def _repair(*, name, broken_expr, reason, explanation, hypothesis):
            res = self._repair_agent.run(
                past_kv=None, function_lib=FUNCTION_LIB, factor_name=name,
                broken_expr=broken_expr, reason=reason,
                explanation=explanation or "", hypothesis=hypothesis or "")
            fixed = R.parse_repair_output(res.text or "")
            self.log.line(
                f"  REPAIR-AGENT {name}: think={res.latent_s:.2f}s gen={res.gen_s:.2f}s "
                f"out={res.n_output_tokens} -> {fixed or '(no fix)'}")
            return fixed

        return _repair

    def _score_and_prepare_feedback(self, node: dict, construct_text: str,
                                    feedback_inputs: Dict[str, Dict[str, str]]) -> dict:
        """Gate+repair+backtest construct (runner), simpan var feedback gen berikut.

        SOTA RankIC di-track lintas generasi (replace-best deterministik)."""
        fb_vars = R.run_construct(construct_text, sota_rankic=self.sota_rankic,
                                  mode=self.cfg.backtest_mode,
                                  repair_fn=self._make_repair_fn(), log=self.log)
        best = fb_vars.get("_best_rankic")
        if best is not None and (self.sota_rankic is None or best > self.sota_rankic):
            self.sota_rankic = best
        next_fb = f"feedback_g{node['gen'] + 1}"
        feedback_inputs[next_fb] = fb_vars
        return fb_vars.get("_score", {"score": None})
