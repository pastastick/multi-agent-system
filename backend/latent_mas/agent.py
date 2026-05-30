"""
latent_mas/agent.py
====================
`LatentAgent` — unit agent modular yang bisa dijalankan SENDIRI-SENDIRI.

Filosofi desain
---------------
Tiap agent = (spec prompt + mode KV + parser). Tidak ada agent yang "tahu"
tentang pipeline. Akibatnya kamu bisa:

    backend = LocalLLMBackend(...)
    judger  = load_agent("judger", backend)
    res     = judger.run(past_kv=some_kv, hypothesis="...", function_lib="...")
    print(res.text)              # output teks
    kv_describe(res.kv_cache)    # cek isi KV

…tanpa menjalankan seluruh pipeline. Ini yang bikin debugging prompt &
inspeksi KV jauh lebih cepat.

Mode KV (diteruskan ke LocalLLMBackend.run)
-------------------------------------------
  kv_only      : latent reasoning saja, tidak generate teks → cuma membentuk KV.
                 Untuk proposal/construct/consistency (front-end sequential).
  kv_and_text  : latent reasoning lalu generate teks dari KV.
                 Untuk judger/repair/feedback/mutation/crossover.
  text_only    : generate teks tanpa menyimpan KV (jarang dipakai di sini).

Spec dimuat dari `prompts.yaml` (lihat `load_agent` / `load_all_agents`).
"""

from __future__ import annotations

import time
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from jinja2 import Environment, StrictUndefined, Undefined

from llm.client import LocalLLMBackend, KVCache
from latent_mas.kv_ops import kv_describe, kv_seq_len
from latent_mas.parsers import PARSERS

_PROMPTS_PATH = Path(__file__).parent / "prompts.yaml"


class _VisibleUndefined(Undefined):
    """Untuk run standalone: variabel hilang dirender sebagai penanda terlihat
    alih-alih crash — supaya gampang lihat var mana yang belum diisi."""
    def __str__(self) -> str:  # noqa: D401
        return f"[[MISSING:{self._undefined_name}]]"


# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class AgentSpec:
    role: str
    mode: str = "kv_only"              # kv_only | kv_and_text | text_only
    system: str = ""                   # jinja template
    user: str = ""                     # jinja template
    latent_steps: Optional[int] = None
    temperature: Optional[float] = None
    max_new_tokens: Optional[int] = None
    parser: Optional[Callable[[str], Any]] = None
    json_mode: bool = False


@dataclass
class AgentResult:
    role: str
    mode: str
    text: Optional[str]
    kv_cache: Optional[KVCache]
    duration_s: float
    n_input_tokens: int = 0
    n_output_tokens: int = 0
    kv_seq_len: int = 0
    parsed: Any = None
    hidden_last: Any = None
    latent_vecs: Any = None
    input_ids: Any = None
    output_ids: Any = None

    @property
    def ok(self) -> bool:
        if self.mode == "kv_only":
            return self.kv_cache is not None
        return bool(self.text and self.text.strip())

    def describe(self) -> Dict[str, Any]:
        return {
            "role": self.role, "mode": self.mode, "ok": self.ok,
            "duration_s": round(self.duration_s, 3),
            "text_len": len(self.text) if self.text else 0,
            "n_out_tok": self.n_output_tokens,
            "kv": kv_describe(self.kv_cache),
            "parsed": self.parsed,
        }


# ─────────────────────────────────────────────────────────────────────────────

class LatentAgent:
    """Satu agent latent yang berdiri sendiri."""

    def __init__(
        self,
        spec: AgentSpec,
        backend: LocalLLMBackend,
        *,
        strict_vars: bool = True,
        runlog: Any = None,
    ) -> None:
        self.spec = spec
        self.backend = backend
        self.runlog = runlog
        self._env = Environment(
            undefined=StrictUndefined if strict_vars else _VisibleUndefined
        )

    # ── prompt rendering ─────────────────────────────────────────────────────

    def render(self, **vars: Any) -> tuple[str, str]:
        system = self._env.from_string(self.spec.system).render(**vars).strip()
        user = self._env.from_string(self.spec.user).render(**vars).strip()
        return system, user

    # ── eksekusi ─────────────────────────────────────────────────────────────

    def run(
        self,
        *,
        past_kv: Optional[KVCache] = None,
        runlog: Any = None,
        **vars: Any,
    ) -> AgentResult:
        """Render prompt → panggil backend → parse → AgentResult.

        `past_kv` adalah KV dari agent sebelumnya (sudah di-clone oleh
        orkestrator bila perlu — agent TIDAK meng-clone sendiri).
        """
        rl = runlog or self.runlog
        system, user = self.render(**vars)

        step_cm = rl.step(self.spec.role) if rl is not None else nullcontext()
        t0 = time.time()
        with step_cm:
            res = self.backend.build_messages_and_run(
                user_prompt=user,
                system_prompt=system,
                past_key_values=past_kv,
                mode=self.spec.mode,
                role=self.spec.role,
                latent_steps=self.spec.latent_steps,
                temperature=self.spec.temperature,
                max_new_tokens=self.spec.max_new_tokens,
                json_mode=self.spec.json_mode,
            )
        dur = time.time() - t0

        parsed = None
        if self.spec.parser is not None and res.text:
            try:
                parsed = self.spec.parser(res.text)
            except Exception as e:  # noqa: BLE001
                if rl is not None:
                    rl.warn(f"parser failed for {self.spec.role}", err=repr(e))

        out = AgentResult(
            role=self.spec.role,
            mode=self.spec.mode,
            text=res.text,
            kv_cache=res.kv_cache,
            duration_s=dur,
            n_input_tokens=int(res.input_ids.shape[-1]) if res.input_ids is not None else 0,
            n_output_tokens=int(res.output_ids.shape[-1]) if res.output_ids is not None else 0,
            kv_seq_len=kv_seq_len(res.kv_cache),
            parsed=parsed,
            hidden_last=res.hidden_last,
            latent_vecs=res.latent_vecs,
            input_ids=res.input_ids,
            output_ids=res.output_ids,
        )
        if rl is not None:
            rl.event("agent_done", **out.describe())
        return out


# ─────────────────────────────────────────────────────────────────────────────
# Loader dari prompts.yaml
# ─────────────────────────────────────────────────────────────────────────────

def _load_specs(path: Path = _PROMPTS_PATH) -> Dict[str, AgentSpec]:
    import yaml
    raw = yaml.safe_load(path.read_text())
    specs: Dict[str, AgentSpec] = {}
    for name, cfg in (raw.get("agents") or {}).items():
        parser_name = cfg.get("parser", "none")
        specs[name] = AgentSpec(
            role=cfg.get("role", name),
            mode=cfg.get("mode", "kv_only"),
            system=cfg.get("system", ""),
            user=cfg.get("user", ""),
            latent_steps=cfg.get("latent_steps"),
            temperature=cfg.get("temperature"),
            max_new_tokens=cfg.get("max_new_tokens"),
            parser=PARSERS.get(parser_name),
            json_mode=cfg.get("json_mode", False),
        )
    return specs


def load_agent(
    name: str,
    backend: LocalLLMBackend,
    *,
    strict_vars: bool = True,
    runlog: Any = None,
    path: Path = _PROMPTS_PATH,
) -> LatentAgent:
    """Muat satu agent by name dari prompts.yaml."""
    specs = _load_specs(path)
    if name not in specs:
        raise KeyError(f"agent '{name}' tidak ada di {path}. "
                       f"Tersedia: {sorted(specs)}")
    return LatentAgent(specs[name], backend, strict_vars=strict_vars, runlog=runlog)


def load_all_agents(
    backend: LocalLLMBackend,
    *,
    strict_vars: bool = True,
    runlog: Any = None,
    path: Path = _PROMPTS_PATH,
) -> Dict[str, LatentAgent]:
    """Muat semua agent sekaligus (untuk pipeline)."""
    return {
        name: LatentAgent(spec, backend, strict_vars=strict_vars, runlog=runlog)
        for name, spec in _load_specs(path).items()
    }
