"""Apakah `kv_truncate` merusak pembukuan posisi RoPE ketika KV bertumbuh?

Klaim yang diuji. `llm/_shared.py::kv_truncate` menyimpan k token TERAKHIR:

    K_l <- K_l[..., -k:, :] ,  V_l <- V_l[..., -k:, :]

Key yang tersisa masih membawa FASE RoPE dari posisi ABSOLUT aslinya
[d, d+1, ..., N-1] dengan d = N - k. Tetapi panjang cache yang dilaporkan
menjadi k, dan transformers menetapkan posisi token berikutnya dari
`past_key_values.get_seq_length()`, yaitu k — seolah blok yang disimpan bermula
di posisi 0. Karena attention RoPE hanya bergantung pada SELISIH posisi
(q_pos − k_pos), setiap key lama tampak `d` posisi lebih DEKAT daripada
sebenarnya.

Basis kode sudah mengakui kelas galat ini di jalur lain: `kv_knn_filter`
memanggil `_rerotate_keys_contiguous` justru untuk memperbaikinya. `kv_truncate`
tidak pernah memanggilnya.

Uji di sini membandingkan tiga cara melanjutkan dari KV yang dipotong:
    A  potong saja                       (kode sekarang)
    B  potong + re-rotasi ke posisi kontigu   (perbaikan yang diusulkan)
    C  konteks segar berisi k token yang sama (rujukan "tanpa cacat posisi")
Bila pembukuan posisi tidak penting, A ≈ B. Kalau A menyimpang dan B mendekat
ke C, galatnya nyata dan perbaikannya bekerja.

    python lab/kv_truncate_probe.py --model Qwen/Qwen3-8B
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

TEXT = (
    "In quantitative finance, an alpha factor assigns a score to every stock on "
    "every trading day, and those scores are compared across stocks to rank them. "
    "The score is computed from daily price and volume data: open, high, low, "
    "close, volume, and return. A good factor is one whose ranking today "
    "correlates with the cross-sectional return tomorrow. "
) * 40


@torch.no_grad()
def next_dist(model, tok, ids, past=None, device="cuda"):
    past_len = 0 if past is None else past.get_seq_length()
    mask = torch.ones((1, past_len + ids.shape[-1]), dtype=torch.long, device=device)
    out = model(input_ids=ids, attention_mask=mask, past_key_values=past,
                use_cache=True, return_dict=True)
    return F.softmax(out.logits[0, -1].float(), dim=-1)


def kl(p, q):
    return float((p * (p.clamp_min(1e-12) / q.clamp_min(1e-12)).log()).sum())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--keep", type=int, default=512, help="k token yang disimpan")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
    from llm._shared import kv_truncate, _rerotate_keys_contiguous

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16).to("cuda").eval()

    ids = tok(TEXT, return_tensors="pt").input_ids.to("cuda")
    N = ids.shape[-1]
    k = args.keep
    d = N - k
    probe = tok("\nThe factor that ranks best is", return_tensors="pt").input_ids.to("cuda")
    print(f"[trunc] konteks {N} token → simpan {k} (buang d={d})")

    def full_cache():
        out = model(input_ids=ids, use_cache=True, return_dict=True)
        return out.past_key_values

    # A — potong apa adanya (perilaku kode sekarang)
    pa = kv_truncate(full_cache(), k)
    a = next_dist(model, tok, probe, pa)

    # B — potong + re-rotasi key ke posisi kontigu [0..k-1]
    pb = kv_truncate(full_cache(), k)
    orig = torch.arange(d, N, device="cuda").unsqueeze(0)   # posisi asli token yg disimpan
    pb = _rerotate_keys_contiguous(pb, orig, model)
    b = next_dist(model, tok, probe, pb)

    # B8 — jalur PRODUKSI setelah perbaikan: kv_truncate(model=...) harus
    # melakukan sendiri apa yang di lengan B dikerjakan manual. Kalau KL(B8||B)
    # tidak ~0, perbaikan itu tidak benar-benar terpasang di jalur yang dipakai
    # LocalLLMBackend.run — dan menyalakan anggaran KV (B9) akan berbahaya.
    pb8 = kv_truncate(full_cache(), k, model=model)
    b8 = next_dist(model, tok, probe, pb8)

    # C — rujukan: konteks segar berisi k token yang SAMA
    pc = model(input_ids=ids[:, -k:], use_cache=True, return_dict=True).past_key_values
    c = next_dist(model, tok, probe, pc)

    res = {
        "model": args.model, "n_context": int(N), "keep": k, "dropped": int(d),
        "KL(A||C)_potong_saja": round(kl(a, c), 4),
        "KL(B||C)_potong_plus_rerotasi": round(kl(b, c), 4),
        "KL(B8||C)_kv_truncate_dgn_model": round(kl(b8, c), 4),
        "KL(B8||B)_harus_nol": round(kl(b8, b), 6),
        "KL(A||B)": round(kl(a, b), 4),
        "top1_A": tok.decode([int(a.argmax())]),
        "top1_B": tok.decode([int(b.argmax())]),
        "top1_C": tok.decode([int(c.argmax())]),
        "top1_agree_A_C": int(a.argmax()) == int(c.argmax()),
        "top1_agree_B_C": int(b.argmax()) == int(c.argmax()),
        "H_A": round(float(-(a * a.clamp_min(1e-12).log()).sum()), 3),
        "H_B": round(float(-(b * b.clamp_min(1e-12).log()).sum()), 3),
        "H_C": round(float(-(c * c.clamp_min(1e-12).log()).sum()), 3),
    }
    print(json.dumps(res, indent=2, ensure_ascii=False))
    (OUT / f"kv_truncate_probe_{args.model.split('/')[-1]}.json").write_text(
        json.dumps(res, indent=2, ensure_ascii=False))

    print("\nBACAAN:")
    if res["KL(B8||B)_harus_nol"] < 1e-4:
        print("  kv_truncate(model=...) IDENTIK dengan re-rotasi manual → B8 terpasang "
              "di jalur produksi.")
    else:
        print(f"  PERINGATAN: KL(B8||B)={res['KL(B8||B)_harus_nol']} ≠ 0 — perbaikan B8 "
              f"TIDAK aktif di jalur produksi; jangan nyalakan anggaran KV (B9).")
    if res["KL(A||C)_potong_saja"] > 5 * max(res["KL(B||C)_potong_plus_rerotasi"], 1e-6):
        print("  kv_truncate apa adanya MENYIMPANG jauh dari rujukan; re-rotasi "
              "memulihkannya → pembukuan posisi RoPE memang salah.")
    else:
        print("  selisihnya kecil pada konfigurasi ini — laporkan angkanya apa adanya.")


if __name__ == "__main__":
    main()
