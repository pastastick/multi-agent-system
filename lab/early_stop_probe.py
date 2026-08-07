"""B6 — kapan early-stop rollout laten benar-benar menyala, dan berapa hematnya.

Bedanya dengan `lab/latent_dynamics.py`: skrip itu MEREPLIKASI langkah laten di
luar produksi (untuk bebas memilih varian). Skrip ini memanggil jalur produksi
apa adanya — `LocalLLMBackend.build_messages_and_run(mode="kv_only")` →
`_CoreEngine.latent_pass` — sehingga yang terukur adalah kode yang benar-benar
dipakai agen, termasuk seluruh plumbing (`LLMResult.n_latent_steps`).

Yang diukur per (mode langkah laten × prompt × seed):
  - n_latent_steps : langkah yang BENAR-BENAR berjalan (≤ anggaran)
  - latent_stop    : "early_stop" (titik tetap) | "budget" (anggaran habis)
  - kv_len         : panjang KV setelah latent_pass → token yang dihemat
  - dur_s          : biaya dinding

Kenapa penting: B6 mengubah makna `latent_steps` dari TARGET jadi BATAS ATAS.
Klaim itu hanya sah kalau bisa ditunjukkan (a) early-stop menyala pada mode yang
memang membeku, dan (b) TIDAK menyala pada mode produksi yang tidak membeku —
kalau menyala di mana-mana, ia diam-diam memotong "pikiran", bukan salinan.

    python lab/early_stop_probe.py --model Qwen/Qwen3-8B --budget 60
    python lab/early_stop_probe.py --budget 60 --modes raw,gumbel --seeds 0,1
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
import time
from pathlib import Path

QL = Path(__file__).resolve().parent.parent
_HERE = str(Path(__file__).resolve().parent)
sys.path[:] = [p for p in sys.path if p not in ("", ".", _HERE)]
for p in (str(QL), str(QL / "backend")):
    if p not in sys.path:
        sys.path.insert(0, p)

OUT = QL / "lab" / "out"

# Prompt yang sama dengan lab/latent_dynamics.py, supaya angka kedua skrip
# (replika vs produksi) bisa disandingkan langsung.
PROMPTS = {
    "alpha_momentum": "Propose a market hypothesis about price momentum in Chinese A-share stocks, then name the data columns it needs.",
    "alpha_liquidity": "Propose a market hypothesis about trading volume and liquidity in Chinese A-share stocks, then name the data columns it needs.",
    "alpha_volatility": "Propose a market hypothesis about volatility clustering in Chinese A-share stocks, then name the data columns it needs.",
}

# (mode, temperature) — sama dengan VARIANTS di latent_dynamics.py, minus
# varian yang tak ada di produksi (raw_noise/raw_realign hanya ada di replika:
# di produksi keduanya diatur `use_realign`, bukan `step_mode`).
ARMS = [("raw", 0.7), ("soft", 1.0), ("gumbel", 0.7), ("sample", 1.0)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--budget", type=int, default=60,
                    help="latent_steps sebagai BATAS ATAS")
    ap.add_argument("--early-stop-cos", dest="early_stop_cos", type=float,
                    default=0.999)
    ap.add_argument("--modes", default="",
                    help="subset mode, mis. 'raw,gumbel' (kosong = semua)")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--no-realign", action="store_true")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()

    import torch
    from llm.client import LocalLLMBackend

    seeds = [int(s) for s in a.seeds.split(",") if s.strip()]
    want = {m.strip() for m in a.modes.split(",") if m.strip()}
    arms = [x for x in ARMS if not want or x[0] in want]

    OUT.mkdir(parents=True, exist_ok=True)
    backend = LocalLLMBackend(
        model_name=a.model, device=a.device,
        latent_steps=a.budget, use_realign=not a.no_realign,
        enable_thinking=False, log_tensors=False, store_kv=False,
        output_log_dir=str(OUT / "llm_outputs" / "early_stop"),
        max_new_tokens=64, knn_enabled=False,
        latent_early_stop_cos=a.early_stop_cos,
    )
    eng = backend._engine  # noqa: SLF001 — probe memang menguji internal engine

    print(f"[b6] {a.model} budget={a.budget} early_stop_cos={a.early_stop_cos} "
          f"realign={not a.no_realign} arms={len(arms)} seeds={seeds}")

    rows, records = [], []
    for mode, temp in arms:
        # Mode langkah laten dibaca per-panggilan dari atribut engine, jadi
        # cukup ditukar di sini — tak perlu membangun ulang backend (bobot &
        # matriks realignment dipakai bersama).
        eng.latent_step_mode, eng.latent_step_temp = mode, temp
        per_arm = []
        for pname, prompt in PROMPTS.items():
            for seed in seeds:
                torch.manual_seed(seed)
                t0 = time.time()
                res = backend.build_messages_and_run(
                    user_prompt=prompt, mode="kv_only", role=f"b6_{mode}",
                )
                dur = time.time() - t0
                kv_len = 0
                try:
                    kv_len = int(res.kv_cache.get_seq_length())
                except Exception:  # noqa: BLE001
                    pass
                rec = {"mode": mode, "temp": temp, "prompt": pname, "seed": seed,
                       "n_latent_steps": res.n_latent_steps,
                       "latent_stop": res.latent_stop,
                       "kv_len": kv_len, "dur_s": round(dur, 2)}
                records.append(rec)
                per_arm.append(rec)
                del res
                torch.cuda.empty_cache()

        steps = [r["n_latent_steps"] for r in per_arm]
        n_early = sum(1 for r in per_arm if r["latent_stop"] == "early_stop")
        row = {
            "variant": f"{mode}@T{temp}",
            "n_run": len(per_arm),
            "early_stop_fired": f"{n_early}/{len(per_arm)}",
            "steps_mean": round(st.mean(steps), 1),
            "steps_min": min(steps), "steps_max": max(steps),
            "saved_frac": round(1 - st.mean(steps) / a.budget, 3),
            "dur_mean_s": round(st.mean(r["dur_s"] for r in per_arm), 2),
        }
        rows.append(row)
        print(f"  {row['variant']:14s} early-stop={row['early_stop_fired']:>5s} "
              f"langkah={row['steps_mean']:5.1f} "
              f"[{row['steps_min']}-{row['steps_max']}]/{a.budget} "
              f"hemat={row['saved_frac']:+.1%} "
              f"{row['dur_mean_s']:6.2f}s", flush=True)

    doc = {"_meta": {"model": a.model, "budget": a.budget,
                     "early_stop_cos": a.early_stop_cos,
                     "use_realign": not a.no_realign, "seeds": seeds},
           "_summary": rows, "records": records}
    suffix = f"_{a.tag}" if a.tag else ""
    path = OUT / f"early_stop_{a.model.replace('/', '_')}{suffix}.json"
    path.write_text(json.dumps(doc, indent=2))
    print(f"tersimpan → {path}")


if __name__ == "__main__":
    main()
