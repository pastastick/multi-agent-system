"""G7 — apa yang terjadi pada RUANG LATEN saat ia BERTUMBUH lintas hop agen.

Kenapa ini perlu, dan kenapa G1 tidak cukup. `lab/latent_dynamics.py` (dan
AUDIT_KRITIS §3.2) mengukur SATU agen yang memulai dari prompt segar: satu
rollout, KV kosong di awal. Pipeline sesungguhnya tidak begitu — ia menumpuk:

    proposal : KV = [prompt_p]                       + L vektor laten
    design   : KV = [                    ^^^ di atas ] + [prompt_d] + L laten
    construct: KV = [                    ^^^ di atas ] + [prompt_c] + L laten
               lalu MENG-EMIT teks dari KV gabungan itu.

Jadi yang menulis ekspresi adalah agen dengan konteks TERPANJANG dan paling
tercemar, bukan konteks segar. Dua sumber pertumbuhan yang tak pernah diukur:

  (1) DUPLIKASI PROMPT. Pustaka fungsi DSL (~1.5k token) ditulis ULANG di dalam
      system prompt design DAN construct. Di comm_mode=text tiap agen mulai dari
      konteks kosong sehingga pustaka itu muncul SEKALI. Di comm_mode kv /
      kv_and_text semuanya menumpuk di satu KV → blok hampir identik muncul 2-3x
      dalam satu konteks. Konteks dengan blok berulang adalah pemicu klasik
      repetition collapse.
  (2) MASSA LATEN DUPLIKAT. G1 menunjukkan rollout laten mencapai TITIK TETAP
      (Qwen3-4B: langkah ~12; Qwen3-8B: ~34). Dengan latent_steps=60 sisanya
      adalah salinan vektor yang sama. Softmax attention menjumlahkan massa atas
      SEMUA kunci: k salinan kunci k* menyumbang massa ~ k*exp(q.k*/sqrt(d))/Z,
      jadi pengaruhnya tumbuh LINEAR terhadap jumlah salinan — dan menumpuk tiap
      hop.

Skrip ini mengukur, pada hop yang benar-benar meng-emit teks:
  - komposisi KV: token prompt vs token laten, per hop dan kumulatif
  - fraksi prompt hop-k yang SUDAH ada verbatim di KV (duplikasi)
  - MASSA ATTENTION pada blok laten vs blok prompt duplikat vs sisanya,
    diukur langsung dari attention weight token pertama yang di-emit
  - self-similarity vektor laten di dalam & antar hop

    python lab/latent_growth.py --model Qwen/Qwen3-8B --steps 60 --comm-mode kv
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

QL = Path(__file__).resolve().parent.parent
_HERE = str(Path(__file__).resolve().parent)
sys.path[:] = [p for p in sys.path if p not in ("", ".", _HERE)]
for p in (str(QL), str(QL / "backend")):
    if p not in sys.path:
        sys.path.insert(0, p)

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

OUT = QL / "lab" / "out"
DIRECTION = "short-term reversal after abnormally high-volume days in small-cap stocks"


def render(agent, **vars):
    return agent.render(**vars)


def ngram_overlap(new_ids: list[int], old_ids: list[int], n: int = 8) -> float:
    """Fraksi n-gram prompt baru yang SUDAH ada di konteks — ukuran duplikasi
    verbatim (bukan kemiripan semantik)."""
    if len(new_ids) < n:
        return 0.0
    old = {tuple(old_ids[i:i + n]) for i in range(len(old_ids) - n + 1)}
    hits = sum(tuple(new_ids[i:i + n]) in old for i in range(len(new_ids) - n + 1))
    return hits / (len(new_ids) - n + 1)


@torch.no_grad()
def attention_mass(engine, kv, segments: list[dict]) -> dict:
    """Massa attention token yang AKAN di-emit terhadap tiap segmen KV.

    Satu forward pass atas prefiks asisten dengan output_attentions=True; ambil
    baris query TERAKHIR di tiap layer/head, lalu jumlahkan bobot pada rentang
    indeks tiap segmen. Ini pengukuran langsung, bukan proksi.
    """
    from latent_mas.kv_ops import kv_deepcopy

    kv = kv_deepcopy(kv)
    prefix = "<|im_start|>assistant\n"
    ids, _ = engine.tokenize(prefix)
    past_len = _past_len(kv)
    mask = torch.ones((ids.shape[0], past_len + ids.shape[-1]),
                      dtype=torch.long, device=engine.device)
    out = engine.model(input_ids=ids, attention_mask=mask, past_key_values=kv,
                       use_cache=True, output_attentions=True, return_dict=True)
    # attentions: tuple per-layer [B, H, q_len, kv_len]
    per_layer = []
    for att in out.attentions:
        if att is None:
            continue
        row = att[0, :, -1, :].float()             # [H, kv_len]
        per_layer.append(row.mean(0))              # rata-rata head → [kv_len]
    if not per_layer:
        return {"error": "model tidak mengembalikan attention weights "
                         "(attn_implementation bukan eager)"}
    A = torch.stack(per_layer)                     # [L, kv_len]
    res = {"n_layers": int(A.shape[0]), "kv_len": int(A.shape[1]), "segments": []}
    for s in segments:
        lo, hi = s["lo"], min(s["hi"], A.shape[1])
        if hi <= lo:
            continue
        m = A[:, lo:hi].sum(-1)                    # massa per layer
        res["segments"].append({
            "name": s["name"], "lo": lo, "hi": hi, "n_tok": hi - lo,
            "share_of_context": round((hi - lo) / A.shape[1], 4),
            "attn_mass_mean": round(float(m.mean()), 4),
            "attn_mass_max_layer": round(float(m.max()), 4),
            # >1 berarti segmen menarik attention LEBIH dari porsi panjangnya
            "enrichment": round(float(m.mean()) / max((hi - lo) / A.shape[1], 1e-9), 2),
        })
    return res


def _past_len(kv) -> int:
    from llm._shared import _past_length
    return _past_length(kv)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--comm-mode", default="kv", choices=["kv", "kv_and_text"])
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    from llm.client import LocalLLMBackend
    from latent_mas.agent import load_all_agents
    from latent_mas.operator_families import diversity_hint

    OUT.mkdir(parents=True, exist_ok=True)
    backend = LocalLLMBackend(
        model_name=args.model, device="cuda", latent_steps=args.steps,
        use_realign=True, enable_thinking=False, log_tensors=False,
        store_kv=False, output_log_dir=str(OUT / "llm_outputs" / "growth"),
        max_new_tokens=2048, temperature=0.8, knn_enabled=False,
    )
    eng = backend._engine
    # attention weights hanya keluar pada implementasi eager (SDPA/flash tidak
    # memateralisasi matriks bobot). API-nya berpindah antar versi transformers.
    for setter in (
        lambda: eng.model.set_attn_implementation("eager"),
        lambda: eng.model.config.__setattr__("_attn_implementation", "eager"),
    ):
        try:
            setter()
            break
        except Exception as e:  # noqa: BLE001
            print(f"[growth] set eager gagal ({e!r}), coba cara lain")
    agents = load_all_agents(backend)

    kv_only = "kv_only" if args.comm_mode == "kv" else "kv_and_text"
    hops = [
        ("proposal", kv_only, dict(handoff="text", direction=DIRECTION,
                                   market_context="", prior_feedback="",
                                   negative_hint="")),
        ("design", kv_only, dict(handoff="kv")),
        ("construct", "kv_and_text", dict(handoff="kv",
                                          diversity_hint=diversity_hint([]))),
    ]

    kv = None
    seen_ids: list[int] = []
    segments: list[dict] = []
    report = {"model": args.model, "steps": args.steps, "comm_mode": args.comm_mode,
              "hops": []}
    latent_blocks = {}

    for name, mode, kw in hops:
        agent = agents[name]
        system, user = agent.render(**kw)
        msgs = backend.build_messages(user_prompt=user, system_prompt=system)
        prompt_text = eng.format_messages(msgs, add_generation_prompt=False)
        ids, _ = eng.tokenize(prompt_text)
        n_prompt = int(ids.shape[-1])
        lo_prompt = _past_len(kv) if kv is not None else 0
        dup = ngram_overlap(ids[0].tolist(), seen_ids) if seen_ids else 0.0

        res = backend.run(messages=msgs, mode=mode, role=name,
                          past_key_values=kv, record_latent_vecs=True,
                          crop_after_generate=False)
        kv = res.kv_cache
        vecs = res.latent_vecs                     # [steps, d]
        lo_lat = lo_prompt + n_prompt
        segments.append({"name": f"{name}:prompt", "lo": lo_prompt, "hi": lo_lat})
        segments.append({"name": f"{name}:latent", "lo": lo_lat,
                         "hi": lo_lat + args.steps})
        seen_ids.extend(ids[0].tolist())

        # self-similarity vektor laten hop ini (berapa yang praktis salinan?)
        stat = {}
        if vecs is not None and vecs.shape[0] > 1:
            v = F.normalize(vecs.float(), dim=-1)
            cons = (v[1:] * v[:-1]).sum(-1)        # cos antar-langkah berurutan
            first_frozen = next((i for i, c in enumerate(cons.tolist()) if c > 0.999), None)
            stat = {
                "cos_consecutive_mean": round(float(cons.mean()), 5),
                "cos_consecutive_last": round(float(cons[-1]), 5),
                "first_frozen_step": first_frozen,
                "n_effectively_duplicate": (args.steps - first_frozen
                                            if first_frozen is not None else 0),
                "pairwise_cos_mean": round(float((v @ v.T).mean()), 4),
            }
            latent_blocks[name] = v.cpu()

        hop = {"agent": name, "mode": mode, "n_prompt_tokens": n_prompt,
               "prompt_dup_frac_8gram": round(dup, 4),
               "kv_len_after": _past_len(kv),
               "latent_block": [lo_lat, lo_lat + args.steps],
               "text_len": len(res.text or ""), "latent": stat}
        report["hops"].append(hop)
        print(f"[growth] {name:<10s} prompt={n_prompt:>5d} tok "
              f"(dup8={dup:>5.1%})  kv_after={hop['kv_len_after']:>6d}  "
              f"beku@={stat.get('first_frozen_step')}  text={hop['text_len']}",
              flush=True)

    # kemiripan blok laten ANTAR hop: apakah "pikiran" tiap agen berbeda?
    names = list(latent_blocks)
    cross = {}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            cross[f"{a}~{b}"] = round(
                float((latent_blocks[a] @ latent_blocks[b].T).mean()), 4)
    report["latent_cross_hop_cos"] = cross

    report["attention_at_emission"] = attention_mass(eng, kv, segments)
    report["kv_total"] = _past_len(kv)
    report["latent_total"] = args.steps * len(hops)
    report["latent_share"] = round(args.steps * len(hops) / max(_past_len(kv), 1), 4)

    tag = args.tag or f"{args.model.split('/')[-1]}_{args.comm_mode}_ls{args.steps}"
    path = OUT / f"latent_growth_{tag}.json"
    path.write_text(json.dumps(report, indent=2))

    print(f"\nKV total {report['kv_total']} token; laten {report['latent_total']} "
          f"({report['latent_share']:.1%} panjang konteks)")
    print(f"cos antar-hop blok laten: {cross}")
    am = report["attention_at_emission"]
    if "segments" in am:
        print(f"\nMassa attention token pertama yang di-emit ({am['n_layers']} layer):")
        print(f"  {'segmen':<22s} {'tok':>6s} {'%konteks':>9s} {'massa':>8s} "
              f"{'maxlayer':>9s} {'enrich':>7s}")
        for s in am["segments"]:
            print(f"  {s['name']:<22s} {s['n_tok']:>6d} {s['share_of_context']:>8.1%} "
                  f"{s['attn_mass_mean']:>8.4f} {s['attn_mass_max_layer']:>9.4f} "
                  f"{s['enrichment']:>7.2f}x")
    print(f"\ntersimpan → {path}")


if __name__ == "__main__":
    main()
