"""prod/transfer.py — kebijakan transfer KV antar-agent (NO-CROP).

Lihat DESIGN.md §2-3. Inti perbaikan ada di backend (client.py: crop kondisional +
_close_open_turn; agent.py: AgentSpec.keep_answer_in_kv). Modul ini hanya membungkus
operasi KV level-orkestrasi: salin untuk chain, gabung untuk crossover.

Reuse: latent_mas.kv_ops.{kv_deepcopy, kv_concat}.
"""
from __future__ import annotations

from typing import Any, List, Optional, Sequence


def chain_kv(parent_kv: Optional[Any]) -> Optional[Any]:
    """Salinan independen KV parent untuk di-chain ke SATU agent.

    Wajib clone: DynamicCache dimutasi in-place oleh agent berikut; tanpa clone,
    cabang lain yang memakai parent KV yang sama akan terkontaminasi.
    """
    from latent_mas.kv_ops import kv_deepcopy
    return kv_deepcopy(parent_kv)


def concat_kv(parent_kvs: Sequence[Optional[Any]]) -> Optional[Any]:
    """Gabung >=2 KV parent untuk crossover (LatentMAS Eq.4).

    Tiap parent di-deepcopy dulu lalu di-concat layer-wise sepanjang sekuens.
    Urutan: parent paling 'ingin didengar' di posisi akhir (paling dekat generate).
    """
    from latent_mas.kv_ops import kv_concat, kv_deepcopy
    srcs = [kv_deepcopy(kv) for kv in parent_kvs if kv is not None]
    return kv_concat(srcs) if srcs else None


def transfer_kv(parent_kvs: List[Optional[Any]]) -> Optional[Any]:
    """Pilih chain (1 parent) vs concat (>=2 parent) vs none (0 parent)."""
    kvs = [kv for kv in parent_kvs if kv is not None]
    if not kvs:
        return None
    return chain_kv(kvs[0]) if len(kvs) == 1 else concat_kv(kvs)
