"""B7 — persamaan realignment mana yang benar-benar berlaku, dan apa akibatnya.

Dua pertanyaan, keduanya diukur pada JALUR PRODUKSI (`_CoreEngine.latent_pass`),
bukan pada replika:

  (1) INERTNESS. Apakah `use_realign` masih berpengaruh setelah mode langkah
      laten produksi bukan lagi "raw"? Diuji secara deterministik: mode `soft`
      (tanpa noise), prompt & seed sama, `use_realign` True vs False. Bila
      hidden state akhirnya IDENTIK bit-per-bit, maka matriks ridge M memang
      tidak pernah diterapkan — dan ablasi G6 hanya berlaku untuk mode "raw".

  (2) GEOMETRI. Untuk vektor laten yang BENAR-BENAR diumpankan sebagai virtual
      token, seberapa dekat ia ke embedding token nyata? `max_v cos(z, W_in[v])`
      mengukur apakah vektor itu berada di dalam (≈1) atau di luar (≈0) manifold
      embedding. Inilah beda antara "proyeksi" dan "ekstrapolasi".

Angka ridge M sendiri (cos(h, hM), simpangan dari identitas) datang dari
`eval/realign_probe.py` — skrip itu tidak perlu GPU karena hanya membaca
safetensors.

    PYTHONPATH=backend python backend/eval/b7_probe.py --model Qwen/Qwen3-8B --steps 10
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paths import QL_ROOT as QL, bootstrap, ensure_out, OUT_PROBE as _OUT
bootstrap()

OUT = ensure_out(_OUT)


PROMPT = ("Propose a market hypothesis about price momentum in Chinese A-share "
          "stocks, then name the data columns it needs.")

# Mode deterministik saja untuk uji (1): gumbel/sample menyuntik noise, jadi dua
# jalankan tak akan identik apa pun jawabannya soal M — uji itu jadi tak bisa
# menyimpulkan apa-apa.
DETERMINISTIC = ("raw", "soft")
ALL_MODES = ("raw", "soft", "gumbel", "sample")


def _build(model: str, device: str, steps: int, realign: bool, mode: str,
           temp: float):
    from llm.client import LocalLLMBackend
    return LocalLLMBackend(
        model_name=model, device=device, latent_steps=steps,
        use_realign=realign, enable_thinking=False, log_tensors=False,
        store_kv=False, output_log_dir=str(OUT / "llm_outputs" / "b7"),
        max_new_tokens=64, knn_enabled=False,
        latent_step_mode=mode, latent_step_temp=temp,
        latent_early_stop_cos=1.0,     # early-stop DIMATIKAN: uji ini soal
    )                                  # persamaan, bukan soal panjang rollout


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()

    import torch
    import torch.nn.functional as F

    OUT.mkdir(parents=True, exist_ok=True)
    res: dict = {"_meta": {"model": a.model, "steps": a.steps, "temp": a.temp}}

    # ── (1) inertness use_realign per mode deterministik ────────────────────
    inert = {}
    for mode in DETERMINISTIC:
        hiddens, vecs = [], []
        for realign in (True, False):
            be = _build(a.model, a.device, a.steps, realign, mode, a.temp)
            torch.manual_seed(0)
            r = be.build_messages_and_run(user_prompt=PROMPT, mode="kv_only",
                                          role=f"b7_{mode}_{realign}")
            hiddens.append(r.hidden_last.detach().float().cpu())
            vecs.append(r.latent_vecs.detach().float().cpu())
            del be, r
            torch.cuda.empty_cache()
        identical = bool(torch.equal(hiddens[0], hiddens[1]))
        d = float((hiddens[0] - hiddens[1]).abs().max())
        cos = float(F.cosine_similarity(hiddens[0], hiddens[1], dim=-1).mean())
        inert[mode] = {"hidden_identical": identical,
                       "max_abs_diff": d, "cos": round(cos, 6),
                       "latent_vecs_identical": bool(torch.equal(vecs[0], vecs[1]))}
        verdict = ("M TIDAK dipakai → use_realign inert"
                   if identical else "M dipakai → use_realign bermakna")
        print(f"[b7/inert] {mode:5s} use_realign True vs False: "
              f"identik={identical} maxdiff={d:.3e} cos={cos:.6f}  → {verdict}",
              flush=True)
    res["inertness"] = inert

    # ── (2) geometri vektor laten produksi per mode ─────────────────────────
    geo = {}
    for mode in ALL_MODES:
        be = _build(a.model, a.device, a.steps, True, mode, a.temp)
        torch.manual_seed(0)
        r = be.build_messages_and_run(user_prompt=PROMPT, mode="kv_only",
                                      role=f"b7_geo_{mode}")
        z = r.latent_vecs.detach().float()                      # [steps, d]
        W_in = be._engine.model.get_input_embeddings().weight   # noqa: SLF001
        W_n = F.normalize(W_in.detach().float(), dim=1)
        zc = F.normalize(z.to(W_n.device), dim=1)
        # max_v cos(z_k, W_in[v]) per langkah — dihitung berkeping agar
        # matriks [steps, 151936] tak meledak di VRAM.
        mx = torch.cat([(zc[i:i + 4] @ W_n.T).max(dim=1).values
                        for i in range(0, zc.shape[0], 4)]).cpu()
        geo[mode] = {
            "max_cos_embed_mean": round(float(mx.mean()), 4),
            "max_cos_embed_min": round(float(mx.min()), 4),
            "max_cos_embed_max": round(float(mx.max()), 4),
            "n_steps": int(z.shape[0]),
        }
        print(f"[b7/geo]   {mode:6s} cos ke embedding terdekat: "
              f"rata2={geo[mode]['max_cos_embed_mean']:+.4f} "
              f"[{geo[mode]['max_cos_embed_min']:+.4f}, "
              f"{geo[mode]['max_cos_embed_max']:+.4f}]", flush=True)
        del be, r, z, zc, mx
        torch.cuda.empty_cache()
    res["geometry"] = geo

    suffix = f"_{a.tag}" if a.tag else ""
    path = OUT / f"b7_probe_{a.model.replace('/', '_')}{suffix}.json"
    path.write_text(json.dumps(res, indent=2))
    print(f"tersimpan → {path}")


if __name__ == "__main__":
    main()
