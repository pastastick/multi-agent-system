"""Dinamika rollout laten: apa yang sebenarnya terjadi selama N latent steps.

Mereplikasi persis langkah laten produksi (backend/llm/client.py::latent_pass):

    h   = hidden_states[-1][:, -1, :]          # post final-norm
    z   = realign(h)  = h @ M                  # M = I pada backbone tied-embedding
    z   = z / ||z|| * target_norm              # target_norm = mean ||W_in[i]||
    out = model(inputs_embeds=z.unsqueeze(1), past_key_values=past)

dan mengukur, per langkah:
  - cos(h_k, h_{k-1})        : konvergensi ke titik tetap
  - max_v cos(z_k, W_in[v])  : seberapa "asing" vektor laten terhadap embedding nyata
  - H(softmax(W_out h_k))    : entropi distribusi token yang diwakili state laten
  - argmax token             : token yang akan diemisikan bila di-decode

Membandingkan dua varian langkah laten:
  raw   — seperti produksi (hidden state dinormalisasi ulang)
  soft  — kombinasi konveks embedding: z = softmax(W_out h / T) @ W_in   (Soft
          Thinking). Selalu berada di dalam convex hull embedding nyata, dan
          T memberi kendali ENTROPI pada jalur laten.

CPU (demonstrasi mekanisme):
    .venv/bin/python lab/latent_dynamics.py --model gpt2 --steps 40
GPU (klaim utama skripsi, backbone sebenarnya):
    python lab/latent_dynamics.py --model Qwen/Qwen3-4B --steps 80 --device cuda
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

OUT = Path(__file__).resolve().parent / "out"

PROMPTS = {
    "alpha_momentum": "Propose a market hypothesis about price momentum in Chinese A-share stocks, then name the data columns it needs.",
    "alpha_liquidity": "Propose a market hypothesis about trading volume and liquidity in Chinese A-share stocks, then name the data columns it needs.",
    "alpha_volatility": "Propose a market hypothesis about volatility clustering in Chinese A-share stocks, then name the data columns it needs.",
}


@torch.no_grad()
def rollout(model, tok, prompt: str, steps: int, mode: str,
            temperature: float, device: str) -> dict:
    W_in = model.get_input_embeddings().weight            # [V, d]
    out_emb = model.get_output_embeddings()
    W_out = out_emb.weight if out_emb is not None else W_in
    target_norm = W_in.norm(dim=1).mean()
    W_in_n = F.normalize(W_in.float(), dim=1)

    ids = tok(prompt, return_tensors="pt").input_ids.to(device)
    out = model(input_ids=ids, use_cache=True, output_hidden_states=True, return_dict=True)
    past = out.past_key_values
    h = out.hidden_states[-1][:, -1, :].float()

    rec = {"cos_prev": [], "max_cos_embed": [], "entropy": [], "top1": [],
           "norm_h": [], "top1_prob": []}
    prev = None
    for _ in range(steps):
        logits = h @ W_out.float().T
        p = F.softmax(logits, dim=-1)
        ent = float(-(p * torch.log(p.clamp_min(1e-12))).sum())
        top1 = int(p.argmax())
        rec["entropy"].append(ent)
        rec["top1"].append(tok.decode([top1]))
        rec["top1_prob"].append(float(p.max()))
        rec["norm_h"].append(float(h.norm()))

        if mode == "raw":
            z = h  # M = I pada backbone tied-embedding (lihat lab/realign_probe.py)
        elif mode == "raw_noise":
            z = h + torch.randn_like(h) * temperature * h.norm() / (h.numel() ** 0.5)
        elif mode == "soft":
            z = F.softmax(logits / temperature, dim=-1) @ W_in.float()
        elif mode == "gumbel":
            # USULAN: kombinasi konveks embedding dengan noise Gumbel →
            # (i) kontinu (argumen ekspresivitas LatentMAS tetap berlaku),
            # (ii) di dalam convex hull embedding (in-distribution),
            # (iii) stokastik dengan knob temperature (sumber entropi utk pencarian).
            u = torch.rand_like(logits).clamp_(1e-9, 1.0 - 1e-9)
            gum = -torch.log(-torch.log(u))
            z = F.softmax((logits + gum) / temperature, dim=-1) @ W_in.float()
        elif mode == "sample":
            # entropi TANPA meninggalkan manifold embedding: token di-sample lalu
            # embedding-nya dipakai sebagai token laten (tidak pernah di-emit).
            idx = torch.multinomial(F.softmax(logits / temperature, dim=-1), 1)
            z = W_in.float()[idx.squeeze(-1)]
        else:
            raise ValueError(mode)
        z = z / z.norm(dim=-1, keepdim=True) * target_norm

        rec["max_cos_embed"].append(
            float((F.normalize(z, dim=-1) @ W_in_n.T).max())
        )
        rec["cos_prev"].append(
            float(F.cosine_similarity(h, prev, dim=-1)) if prev is not None else float("nan")
        )
        prev = h.clone()

        step_out = model(inputs_embeds=z.unsqueeze(1).to(W_in.dtype),
                         past_key_values=past, use_cache=True,
                         output_hidden_states=True, return_dict=True)
        past = step_out.past_key_values
        h = step_out.hidden_states[-1][:, -1, :].float()
    return rec


def summarize(rec: dict) -> dict:
    n = len(rec["entropy"])
    tail = slice(max(0, n - 10), n)
    uniq = len(set(rec["top1"]))
    return {
        "entropy_first": round(rec["entropy"][0], 3),
        "entropy_last": round(rec["entropy"][-1], 3),
        "cos_prev_last10_mean": round(
            sum(rec["cos_prev"][tail]) / max(1, len(rec["cos_prev"][tail])), 5),
        "max_cos_embed_mean": round(sum(rec["max_cos_embed"]) / n, 4),
        "unique_top1_tokens": uniq,
        "unique_frac": round(uniq / n, 3),
        "top1_prob_last": round(rec["top1_prob"][-1], 4),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--temperature", type=float, default=1.0)
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    OUT.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(args.model)
    dtype = torch.float32 if args.device == "cpu" else torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype).to(args.device).eval()
    print(f"[dyn] {args.model} d={model.config.hidden_size} "
          f"tied={getattr(model.config,'tie_word_embeddings',None)} steps={args.steps}")

    # (mode, temperature) yang diuji
    VARIANTS = [("raw", 0.0), ("raw_noise", 0.1), ("soft", 1.0), ("soft", 2.0),
                ("gumbel", 0.7), ("gumbel", 1.0), ("sample", 1.0)]
    N_SEED = 3          # replikasi utk mengukur varians lintas-jalankan

    results, summary_rows = {}, []
    for mode, temp in VARIANTS:
        tag = f"{mode}@T{temp}"
        per_prompt_runs = {}
        for name, prompt in PROMPTS.items():
            runs = []
            for seed in range(N_SEED):
                torch.manual_seed(seed)
                runs.append(rollout(model, tok, prompt, args.steps, mode, max(temp, 1e-6),
                                    args.device))
            per_prompt_runs[name] = runs
            results[f"{name}|{tag}"] = {"summary": summarize(runs[0]), "trace": runs[0]}

        s0 = summarize(per_prompt_runs["alpha_momentum"][0])
        # varians LINTAS-SEED: prompt sama, seed beda → berapa jalur yang identik?
        same_seed = tot_seed = 0
        for runs in per_prompt_runs.values():
            for i in range(N_SEED):
                for j in range(i + 1, N_SEED):
                    tot_seed += 1
                    same_seed += tuple(runs[i]["top1"]) == tuple(runs[j]["top1"])
        # varians LINTAS-PROMPT: arah beda, seed sama → jalur berbeda?
        names = list(per_prompt_runs)
        same_pr = tot_pr = 0
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                tot_pr += 1
                same_pr += (tuple(per_prompt_runs[names[i]][0]["top1"])
                            == tuple(per_prompt_runs[names[j]][0]["top1"]))
        row = {
            "variant": tag,
            "entropy_last": s0["entropy_last"],
            "cos_prev_last10": s0["cos_prev_last10_mean"],
            "max_cos_embed": s0["max_cos_embed_mean"],
            "uniq_top1": f"{s0['unique_top1_tokens']}/{args.steps}",
            "identical_across_seed": f"{same_seed}/{tot_seed}",
            "identical_across_prompt": f"{same_pr}/{tot_pr}",
        }
        summary_rows.append(row)
        print(f"  {tag:14s} H_akhir={row['entropy_last']:6.2f} "
              f"cos_prev={row['cos_prev_last10']:+.5f} "
              f"cos_emb={row['max_cos_embed']:+.3f} "
              f"uniq={row['uniq_top1']:>7s} "
              f"sama-lintas-seed={row['identical_across_seed']} "
              f"sama-lintas-arah={row['identical_across_prompt']}")

    results["_summary"] = summary_rows
    path = OUT / f"latent_dynamics_{args.model.replace('/', '_')}.json"
    path.write_text(json.dumps(results, indent=2))
    print(f"tersimpan → {path}")


if __name__ == "__main__":
    main()
