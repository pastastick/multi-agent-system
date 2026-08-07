"""B10 — *latent bottleneck*: bisakah SELURUH KV hulu diringkas jadi m ≪ L vektor?

RENCANA_PERBAIKAN §B10 mengusulkan: alih-alih mewariskan seluruh KV agen hulu,
ringkas jadi `m ≪ L` vektor (attention pooling atas blok laten + jawaban), lalu
itu saja yang diwariskan — supaya konteks emitter berhenti tumbuh linear
terhadap jumlah hop. Skrip ini menguji premisnya SEBELUM mengubah produksi.

Alat ukurnya sama dengan A9 (`lab/channel_capacity.py`): titipkan muatan yang
DIKETAHUI ke hulu, lalu ukur berapa yang selamat sampai hilir. Bedanya, di sini
konteks hulu dibuat sepanjang produksi (prompt `construct` asli ≈ 3–4k token),
karena pada L ≈ 70 token pertanyaan "kompresi 50×" tidak punya arti.

── EMPAT KELUARGA BOTTLENECK (semuanya TRAINING-FREE) ───────────────────────
  pool_uniform   L posisi dibagi jadi m segmen kontigu; K dan V dirata-rata
                 per segmen. Ini bentuk paling harfiah dari usulan B10.
  pool_vnorm     sama, tetapi rata-rata BERBOBOT ‖V_i‖₂ — proksi "attention
                 pooling" yang tak perlu forward tambahan (posisi ber-value
                 besar dianggap lebih informatif).
  select_recent  simpan m token TERAKHIR (= kv_truncate, jalur B8).
  select_knn     simpan m token paling mirip query hilir (= kv_knn_filter).
                 Dua yang terakhir bukan "pooling" melainkan SELEKSI; keduanya
                 disertakan karena keduanya sudah ada di repo, sudah teraudit,
                 dan menjadi pembanding jujur untuk pooling yang lebih rumit.

── PEMBUKUAN RoPE (kenapa pooling tak bisa naif) ────────────────────────────
Key menyimpan fase RoPE posisi ASLI-nya, jadi merata-ratakan key dari posisi
berbeda = merata-ratakan vektor yang diputar dengan sudut berbeda — hasilnya
tak berarti dan besarannya menyusut. Karena itu di sini key di-UN-rotasi ke
posisi 0 dulu, baru dipool, lalu di-re-rotasi ke posisi [0, m). Urutan ini
sama dengan yang dipakai STILL (arXiv:2606.07878) untuk pemadatan KV
ter-latih; di sini semuanya training-free.

Batas teoretis yang harus ikut dilaporkan: attention memakai softmax(q·k), dan
`mean(k)` BUKAN `k` dari token rata-rata. Jadi pooling linear pada key adalah
hampiran, dan justru itulah alasan STILL/Perceiver-resampler MELATIH modul
kompaktor alih-alih merata-rata. Skrip ini mengukur seberapa jauh hampiran
training-free itu membawa kita.

    python lab/latent_bottleneck.py --budgets 16,64,256 --trials 10
"""
from __future__ import annotations

import argparse
import json
import random
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

from lab.channel_capacity import (  # noqa: E402
    SYS_DOWN, USER_DOWN_KV, dsl_names, make_payload, parse_items, score,
)

OUT = QL / "lab" / "out"

FAMILIES = ("pool_uniform", "pool_vnorm", "select_recent", "select_knn")

SYS_UP = "You are a relay agent inside a factor-mining pipeline."
USER_UP = (
    "{head}\n\n"
    "=== RELAY TASK ===\n"
    "PAYLOAD: {payload}\n"
    "Restate all {k} payload items, comma-separated, for the next agent.\n\n"
    "{tail}"
)

# POSISI MUATAN adalah variabel, bukan detail. Kalau muatan selalu ditaruh di
# AKHIR konteks, `select_recent` (simpan m token terakhir) menang otomatis —
# bukan karena seleksi lebih baik daripada pooling, melainkan karena
# eksperimennya menaruh jawabannya persis di tempat yang ia simpan. Di produksi
# informasi yang dibutuhkan hilir tersebar di sepanjang konteks hulu, jadi
# lengan `mid` yang lebih mewakili; `end` dipertahankan untuk MENUNJUKKAN bias
# itu, bukan untuk disembunyikan.
POSITIONS = {"head": 0.0, "mid": 0.5, "end": 1.0}


def production_filler() -> str:
    """Konteks pengisi = prompt `construct` PRODUKSI apa adanya.

    Dipakai supaya L (panjang KV hulu) setara produksi (A5: KV construct 4 624
    token). Memakai lorem ipsum akan membuat rasio kompresi tampak benar tetapi
    ISI-nya tak menyerupai apa pun yang sungguh dioper antar-agen.
    """
    import yaml
    spec = yaml.safe_load((QL / "backend" / "latent_mas" / "prompts.yaml")
                          .read_text())["agents"]["construct"]
    return str(spec.get("system", ""))


def split_filler(filler: str, frac: float) -> tuple[str, str]:
    """Pecah konteks pengisi pada batas baris terdekat ke `frac`.

    Batas BARIS (bukan karakter) supaya potongannya tetap terbaca sebagai
    instruksi utuh — konteks yang terpotong di tengah kalimat akan mengubah
    perilaku model karena alasan yang tak ada hubungannya dengan bottleneck.
    """
    lines = filler.splitlines()
    cut = max(0, min(len(lines), round(len(lines) * frac)))
    return "\n".join(lines[:cut]), "\n".join(lines[cut:])


# ── operasi RoPE (memakai primitif yang sama dengan jalur produksi B8) ───────

def _rope_shift(pairs, delta, rotary, ):
    """Terapkan rotasi R(delta) pada KEY (value tak terkena RoPE).

    delta: [B, k] float — selisih posisi (baru − lama). Karena RoPE additif
    (R(a)·R(b) = R(a+b)), memutar sejauh −pos meng-UN-rotasi key ke posisi 0.
    Identik dengan matematika `_rerotate_keys_contiguous` di llm/_shared.py,
    hanya dengan posisi tujuan yang bebas, bukan dipaksa kontigu.
    """
    import torch
    from llm._shared import _rotate_half

    device = pairs[0][0].device
    ref = torch.zeros(1, dtype=torch.float32, device=device)
    cos, sin = rotary(ref, delta.to(device))       # [B, k, D]
    cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)  # broadcast atas head
    out = []
    for key, value in pairs:
        kf = key.float()
        out.append(((kf * cos + _rotate_half(kf) * sin).to(key.dtype), value))
    return out


def _segments(L: int, m: int) -> list[tuple[int, int]]:
    """Bagi [0, L) jadi m segmen kontigu sepanjang mungkin sama."""
    edges = [round(i * L / m) for i in range(m + 1)]
    return [(a, b) for a, b in zip(edges, edges[1:]) if b > a]


def compress(kv, family: str, budget: int, model, query_hidden=None):
    """Kembalikan KV baru berisi ≤ budget slot. Tidak memutasi `kv` masukan."""
    import torch
    from latent_mas import kv_ops
    from llm._shared import (_get_rotary_emb, _kv_from_pairs, _kv_pairs,
                             kv_knn_filter, kv_truncate)

    kv = kv_ops.kv_deepcopy(kv)
    L = kv_ops.kv_seq_len(kv)
    if budget >= L:
        return kv

    if family == "select_recent":
        return kv_truncate(kv, budget, model=model)
    if family == "select_knn":
        # kv_knn_filter memakai `percentage`; terjemahkan budget → fraksi.
        return kv_knn_filter(kv, query_hidden, percentage=budget / L,
                             min_keep=1, strategy="top", model=model)

    rotary = _get_rotary_emb(model)
    if rotary is None:
        raise RuntimeError("rotary_emb tak ditemukan — pooling tak bisa dibukukan")

    pairs = _kv_pairs(kv)
    B = pairs[0][0].shape[0]
    device = pairs[0][0].device
    pos = torch.arange(L, device=device, dtype=torch.float32).unsqueeze(0).expand(B, -1)

    # 1) UN-rotasi seluruh key ke posisi 0 supaya penjumlahan berarti.
    pairs = _rope_shift(pairs, -pos, rotary)

    # 2) pooling per segmen.
    segs = _segments(L, budget)
    pooled = []
    for key, value in pairs:                       # [B, H, L, D]
        ks, vs = [], []
        for a, b in segs:
            k_seg, v_seg = key[..., a:b, :].float(), value[..., a:b, :].float()
            if family == "pool_vnorm":
                w = v_seg.norm(dim=-1, keepdim=True)            # [B,H,seg,1]
                w = w / w.sum(dim=-2, keepdim=True).clamp_min(1e-6)
                ks.append((k_seg * w).sum(dim=-2, keepdim=True))
                vs.append((v_seg * w).sum(dim=-2, keepdim=True))
            else:                                               # pool_uniform
                ks.append(k_seg.mean(dim=-2, keepdim=True))
                vs.append(v_seg.mean(dim=-2, keepdim=True))
        pooled.append((torch.cat(ks, dim=-2).to(key.dtype),
                       torch.cat(vs, dim=-2).to(value.dtype)))

    # 3) re-rotasi m slot ke posisi kontigu [0, m).
    m = len(segs)
    new_pos = torch.arange(m, device=device, dtype=torch.float32).unsqueeze(0).expand(B, -1)
    pooled = _rope_shift(pooled, new_pos, rotary)
    return _kv_from_pairs(pooled, kv)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--trials", type=int, default=10)
    ap.add_argument("--payload", default="dsl")
    ap.add_argument("--budgets", default="16,64,256")
    ap.add_argument("--position", default="mid", choices=sorted(POSITIONS),
                    help="letak muatan di dalam konteks hulu (lihat POSITIONS)")
    ap.add_argument("--families", default=",".join(FAMILIES))
    ap.add_argument("--latent-steps", type=int, default=10)
    ap.add_argument("--latent-mode", default="gumbel")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()

    import torch
    from latent_mas import kv_ops
    from llm.client import LocalLLMBackend

    OUT.mkdir(parents=True, exist_ok=True)
    pool = dsl_names()
    budgets = [int(b) for b in a.budgets.split(",") if b.strip()]
    fams = [f.strip() for f in a.families.split(",") if f.strip()]
    filler = production_filler()

    backend = LocalLLMBackend(
        model_name=a.model, device=a.device, latent_steps=a.latent_steps,
        use_realign=True, enable_thinking=False, log_tensors=False,
        store_kv=False, output_log_dir=str(OUT / "llm_outputs" / "b10"),
        max_new_tokens=a.max_new_tokens, temperature=0.6, top_p=0.95,
        knn_enabled=False, latent_step_mode=a.latent_mode, latent_step_temp=0.7,
        latent_early_stop_cos=1.0,
    )
    eng = backend._engine          # noqa: SLF001
    model = eng.model

    rng = random.Random(a.seed)
    payloads = [make_payload(a.payload, a.k, rng, pool) for _ in range(a.trials)]

    # Query hilir dipakai select_knn sebagai acuan relevansi — dihitung sekali.
    down_user = USER_DOWN_KV.format(k=a.k)
    q_ids, _ = eng.tokenize(eng.format_messages(
        [{"role": "system", "content": SYS_DOWN},
         {"role": "user", "content": down_user}]))
    query_hidden = model.get_input_embeddings()(q_ids).mean(dim=1)

    # Lengan: referensi tanpa kompresi + (keluarga × anggaran) + lantai.
    arms = [("full", 0)] + [(f, b) for b in budgets for f in fams] + [("none", 0)]

    print(f"[b10] {a.model} k={a.k} trials={a.trials} m_laten={a.latent_steps} "
          f"anggaran={budgets} keluarga={fams}")

    records, rows, L_seen = [], [], []
    for family, budget in arms:
        res = []
        for i, payload in enumerate(payloads):
            torch.manual_seed(a.seed + i)
            t0 = time.time()
            past, L, kv_in = None, 0, 0
            if family != "none":
                head, tail = split_filler(filler, POSITIONS[a.position])
                r_up = backend.build_messages_and_run(
                    user_prompt=USER_UP.format(head=head, tail=tail,
                                               payload=", ".join(payload), k=a.k),
                    system_prompt=SYS_UP, mode="kv_only", role="b10_up",
                )
                L = kv_ops.kv_seq_len(r_up.kv_cache)
                L_seen.append(L)
                past = (r_up.kv_cache if family == "full"
                        else compress(r_up.kv_cache, family, budget, model,
                                      query_hidden))
                # Diukur SEBELUM panggilan hilir: `past` dimutasi in-place oleh
                # forward berikutnya (prompt hilir ter-append), jadi mengukur
                # sesudahnya akan melaporkan anggaran + panjang prompt hilir.
                kv_in = kv_ops.kv_seq_len(past)
                del r_up
            r_down = backend.build_messages_and_run(
                user_prompt=down_user, system_prompt=SYS_DOWN,
                past_key_values=past,
                mode="text_only" if past is None else "kv_and_text",
                role="b10_down", latent_steps=0, crop_after_generate=True,
            )
            pred = parse_items(r_down.text or "", a.payload, pool)
            s = score(pred, payload)
            s.update({"family": family, "budget": budget, "trial": i, "L": L,
                      "kv_in": kv_in,
                      "dur_s": round(time.time() - t0, 2), "pred": pred,
                      "truth": payload,
                      "down_text": (r_down.text or "")[:200]})
            records.append(s)
            res.append(s)
            del past, r_down
            torch.cuda.empty_cache()

        kv_in = round(st.mean(r["kv_in"] for r in res), 1)
        Lm = round(st.mean(r["L"] for r in res), 1) or 1
        row = {"family": family, "budget": budget, "n": len(res),
               "kv_in": kv_in, "L": Lm,
               "compression": round(Lm / kv_in, 1) if kv_in else None,
               "recall": round(st.mean(r["recall"] for r in res), 3),
               "exact": round(st.mean(r["exact"] for r in res), 3),
               "hallucinate": round(st.mean(r["hallucinate"] for r in res), 3),
               "dur_s": round(st.mean(r["dur_s"] for r in res), 2)}
        rows.append(row)
        label = family if family in ("full", "none") else f"{family}@{budget}"
        print(f"  {label:22s} KV {kv_in:7.1f}/{Lm:.0f} "
              f"({'—' if not row['compression'] else str(row['compression'])+'x'}) "
              f"recall={row['recall']:.3f} exact={row['exact']:.3f} "
              f"halus={row['hallucinate']:.3f} {row['dur_s']:5.2f}s", flush=True)

    doc = {"_meta": {"model": a.model, "k": a.k, "trials": a.trials,
                     "latent_steps": a.latent_steps, "payload": a.payload,
                     "position": a.position,
                     "budgets": budgets, "families": fams,
                     "L_mean": round(st.mean(L_seen), 1) if L_seen else 0,
                     "filler": "prompts.yaml::agents.construct.system"},
           "_summary": rows, "records": records}
    suffix = f"_{a.tag}" if a.tag else ""
    path = OUT / f"latent_bottleneck_{a.model.replace('/', '_')}{suffix}.json"
    path.write_text(json.dumps(doc, indent=2))
    print(f"tersimpan → {path}")


if __name__ == "__main__":
    main()
