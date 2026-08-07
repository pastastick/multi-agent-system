"""Apakah matriks realignment ridge benar-benar melakukan sesuatu di Qwen3-4B?

LatentRealigner (backend/llm/_shared.py:696) menyelesaikan
    M = (W_out^T W_out + λI)^{-1} W_out^T W_in
Qwen3-4B punya `tie_word_embeddings: true` → W_out IS W_in → secara aljabar
    M = (W^T W + λI)^{-1} W^T W = V diag(σ²/(σ²+λ)) V^T  ≈  I.

Skrip ini memuat HANYA matriks embedding dari safetensors (tanpa memuat model
4B penuh — muat di RAM 5 GB) dan mengukur seberapa jauh M dari I.

    .venv/bin/python lab/realign_probe.py
"""
from __future__ import annotations

import json
from pathlib import Path

import torch

HUB = Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen3-4B/snapshots"
OUT = Path(__file__).resolve().parent / "out"
LAMBDA = 1e-5  # backend/llm/config default reg_lambda


def load_embed() -> torch.Tensor:
    from safetensors import safe_open

    snap = next(HUB.iterdir())
    idx = json.loads((snap / "model.safetensors.index.json").read_text())["weight_map"]
    key = "model.embed_tokens.weight"
    shard = snap / idx[key]
    with safe_open(shard, framework="pt") as f:
        w = f.get_tensor(key)
    print(f"[probe] {key} {tuple(w.shape)} {w.dtype} dari {shard.name}")
    has_lm_head = any(k.startswith("lm_head") for k in idx)
    print(f"[probe] lm_head.weight ada di checkpoint? {has_lm_head} "
          f"(tie_word_embeddings=true → lm_head memakai bobot embed)")
    return w


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    W = load_embed().to(torch.float32)      # [V, d]
    V, d = W.shape

    gram = W.T @ W                          # [d, d]
    rhs = W.T @ W                           # W_out = W_in (tied)
    M = torch.linalg.solve(gram + LAMBDA * torch.eye(d), rhs)

    eye = torch.eye(d)
    dev_f = (M - eye).norm().item()
    rel = dev_f / eye.norm().item()
    # sudut yang dibuat M pada vektor acak berdistribusi seperti hidden state
    torch.manual_seed(0)
    h = torch.randn(512, d)
    h = h / h.norm(dim=1, keepdim=True)
    hm = h @ M
    cos = torch.nn.functional.cosine_similarity(h, hm, dim=1)
    scale = hm.norm(dim=1) / h.norm(dim=1)

    evals = torch.linalg.eigvalsh(gram)     # σ² dari W
    shrink = evals / (evals + LAMBDA)

    target_norm = W.norm(dim=1).mean().item()

    res = {
        "vocab": V, "d_h": d, "reg_lambda": LAMBDA,
        "frobenius_M_minus_I": dev_f,
        "relative_deviation": rel,
        "cos(h, hM)_mean": cos.mean().item(),
        "cos(h, hM)_min": cos.min().item(),
        "norm_ratio_mean": scale.mean().item(),
        "gram_eig_min": evals.min().item(),
        "gram_eig_max": evals.max().item(),
        "shrink_min": shrink.min().item(),
        "shrink_mean": shrink.mean().item(),
        "target_norm(mean ||W_in[i]||)": target_norm,
        "embed_norm_std": W.norm(dim=1).std().item(),
    }
    print(json.dumps(res, indent=2))
    (OUT / "realign_probe.json").write_text(json.dumps(res, indent=2))
    print("\nKESIMPULAN:")
    print(f"  M menyimpang dari identitas sebesar {rel*100:.4f}% (relatif Frobenius).")
    print(f"  Vektor acak diputar rata-rata cos={cos.mean():.6f} (1.0 = tidak diputar).")
    print("  → Pada Qwen3-4B (embedding tertaut), realignment ridge = identitas + "
          "penskalaan norma.")


if __name__ == "__main__":
    main()
