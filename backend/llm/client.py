"""
new_client.py
=============
Gabungan konsep dari:
  - models.py  : KV-cache chaining, latent reasoning, realignment matrix
  - client.py  : ConvManager, SQliteLazyCache, ChatSession, APIBackend
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ─────────────────────────────────────────────────────────────────────────────
# Shared imports from _shared.py (single source of truth)
# ─────────────────────────────────────────────────────────────────────────────

from llm._shared import (
    KVCache,
    OutputMode,
    LatentRealigner,
    _past_length,
    _ensure_pad_token,
    _kv_to_cpu,
    _kv_to_device,
    kv_truncate,
    kv_knn_filter,
    kv_size_bytes,
    robust_json_parse,
    md5_hash,
)
# =============================================================================
# BAGIAN 2 -- CONV MANAGER (format tensor, bukan JSON)
# =============================================================================

@dataclass
class ConvRecord:
    """
    Satu record percakapan yang disimpan ke disk.
    Menyimpan tensor mentah agar bisa di-decode ulang untuk debugging.

    Fields:
        conv_id       : ID unik percakapan
        step          : urutan langkah dalam pipeline (propose=0, construct=1, dst)
        role          : nama agen
        input_ids     : token ID prompt yang dikirim ke model  [1, seq_len]
        output_ids    : token ID yang dihasilkan model         [1, gen_len]
        hidden_last   : last hidden state setelah forward pass [1, hidden_dim]
                        (opsional, hanya jika output_hidden_states=True)
        latent_vecs   : semua latent vectors selama latent steps [steps, hidden_dim]
                        (opsional, hanya jika ada latent pass)
        metadata      : dict bebas untuk info tambahan
    """
    conv_id     : str
    step        : int
    role        : str
    input_ids   : torch.Tensor
    output_ids  : Optional[torch.Tensor] = None
    hidden_last : Optional[torch.Tensor] = None
    latent_vecs : Optional[torch.Tensor] = None
    metadata    : Dict[str, Any] = field(default_factory=dict)


class TensorConvManager:
    """
    Menyimpan riwayat percakapan dalam format tensor (.pt) ke direktori.

    Struktur direktori:
        conv_dir/
          {conv_id}/
            step_{n:03d}_{role}.pt      <- ConvRecord tersimpan sebagai dict tensor
            index.json                  <- metadata ringan (bisa dibaca tanpa load tensor)
    """

    def __init__(self, conv_dir: str = "./debug/conv_logs") -> None:
        self.conv_dir = Path(conv_dir)
        self.conv_dir.mkdir(parents=True, exist_ok=True)

    def _get_conv_path(self, conv_id: str) -> Path:
        p = self.conv_dir / conv_id
        p.mkdir(parents=True, exist_ok=True)
        return p

    def save(self, record: ConvRecord) -> Path:
        """Simpan satu ConvRecord ke disk. Return path file."""
        conv_path = self._get_conv_path(record.conv_id)
        filename  = f"step_{record.step:03d}_{record.role}.pt"
        filepath  = conv_path / filename

        payload = {
            "conv_id"    : record.conv_id,
            "step"       : record.step,
            "role"       : record.role,
            "input_ids"  : record.input_ids.cpu() if record.input_ids is not None else None,
            "output_ids" : record.output_ids.cpu() if record.output_ids is not None else None,
            "hidden_last": record.hidden_last.cpu() if record.hidden_last is not None else None,
            "latent_vecs": record.latent_vecs.cpu() if record.latent_vecs is not None else None,
            "metadata"   : record.metadata,
        }
        torch.save(payload, filepath)

        # Update index ringan (JSON)
        index_path = conv_path / "index.json"
        index = self._load_index(index_path)
        index.append({
            "step"       : record.step,
            "role"       : record.role,
            "file"       : filename,
            "has_text"   : record.output_ids is not None,
            "has_hidden" : record.hidden_last is not None,
            "has_latent" : record.latent_vecs is not None,
            "metadata"   : record.metadata,
        })
        with open(index_path, "w") as f:
            json.dump(index, f, indent=2)

        return filepath

    def load(self, conv_id: str, step: int, role: str) -> Optional[Dict[str, Any]]:
        """Load ConvRecord dari disk. Return dict atau None."""
        filepath = self._get_conv_path(conv_id) / f"step_{step:03d}_{role}.pt"
        if not filepath.exists():
            return None
        return torch.load(filepath, map_location="cpu")

    def list_steps(self, conv_id: str) -> List[Dict[str, Any]]:
        """Daftar langkah yang sudah disimpan untuk conv_id ini."""
        index_path = self._get_conv_path(conv_id) / "index.json"
        return self._load_index(index_path)

    @staticmethod
    def _load_index(index_path: Path) -> List[Dict[str, Any]]:
        if index_path.exists():
            with open(index_path) as f:
                return json.load(f)
        return []

    def decode_step(self, conv_id: str, step: int, role: str,
                    tokenizer: AutoTokenizer) -> Dict[str, str]:
        """
        Helper debug: decode tensor ke teks yang bisa dibaca manusia.

        Return dict berisi:
            input_text   : teks prompt
            output_text  : teks output (jika ada)
            hidden_norm  : norm dari last hidden state
            latent_norms : list norm tiap latent step
        """
        record = self.load(conv_id, step, role)
        if record is None:
            return {"error": f"Tidak ditemukan: {conv_id}/step_{step}_{role}"}

        result: Dict[str, str] = {}

        if record["input_ids"] is not None:
            result["input_text"] = tokenizer.decode(
                record["input_ids"][0], skip_special_tokens=True
            )
        if record["output_ids"] is not None:
            result["output_text"] = tokenizer.decode(
                record["output_ids"][0], skip_special_tokens=True
            )
        if record["hidden_last"] is not None:
            norm = record["hidden_last"].float().norm().item()
            result["hidden_norm"] = f"{norm:.4f}"
        if record["latent_vecs"] is not None:
            norms = record["latent_vecs"].float().norm(dim=-1).tolist()
            result["latent_norms"] = str([f"{n:.4f}" for n in norms])

        return result


# =============================================================================
# BAGIAN 3 -- KV-CACHE STORAGE (file .pt + SQLite index)
# =============================================================================

class KVCacheStore:
    """
    Penyimpanan KV-cache ke disk dalam dua tier:

    Tier FULL  (.pt di full/):
        Seluruh KV-cache disimpan.
        Dipakai untuk resume evolution loop atau debugging mendalam.

    Tier SELECTIVE (.pt di selective/):
        Subset token tertentu dari KV-cache.
        Saat ini: N token terakhir (token akhir paling informatif karena
        attention sudah mengakumulasi seluruh konteks sebelumnya).
        Placeholder untuk strategi KNN di masa depan.

    SQLite hanya menyimpan metadata (path, seq_len, ukuran, timestamp).
    Pencarian dan listing cepat tanpa harus load file .pt.

    Struktur direktori:
        kv_store/
          full/
            {conv_id}_{step:03d}.pt
          selective/
            {conv_id}_{step:03d}_sel.pt
          kv_index.db
    """

    def __init__(self, store_dir: str = "./debug/kv_store") -> None:
        self.store_dir = Path(store_dir)
        self.full_dir  = self.store_dir / "full"
        self.sel_dir   = self.store_dir / "selective"
        self.full_dir.mkdir(parents=True, exist_ok=True)
        self.sel_dir.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        self.conn = sqlite3.connect(
            str(self.store_dir / "kv_index.db"), timeout=20
        )
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS kv_index (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                conv_id    TEXT    NOT NULL,
                step       INTEGER NOT NULL,
                tier       TEXT    NOT NULL,
                filepath   TEXT    NOT NULL,
                seq_len    INTEGER,
                n_layers   INTEGER,
                hidden_dim INTEGER,
                size_bytes INTEGER,
                created_at REAL,
                metadata   TEXT
            )
        """)
        self.conn.commit()

    def _index_add(self, conv_id: str, step: int, tier: str,
                   filepath: Path, kv: KVCache, metadata: Dict) -> None:
        seq_len    = _past_length(kv)
        n_layers   = len(kv)
        hidden_dim = kv[0][0].shape[-1] if kv else 0
        size_bytes = filepath.stat().st_size if filepath.exists() else 0
        self.conn.execute(
            """INSERT INTO kv_index
               (conv_id,step,tier,filepath,seq_len,n_layers,
                hidden_dim,size_bytes,created_at,metadata)
               VALUES (?,?,?,?,?,?,?,?,?,?)""",
            (conv_id, step, tier, str(filepath), seq_len, n_layers,
             hidden_dim, size_bytes, time.time(), json.dumps(metadata))
        )
        self.conn.commit()

    def lookup(self, conv_id: str, step: int,
               tier: str = "full") -> Optional[Path]:
        """Cari path file KV-cache. Return None jika tidak ada."""
        cur = self.conn.execute(
            "SELECT filepath FROM kv_index "
            "WHERE conv_id=? AND step=? AND tier=? "
            "ORDER BY id DESC LIMIT 1",
            (conv_id, step, tier)
        )
        row = cur.fetchone()
        if row:
            p = Path(row[0])
            return p if p.exists() else None
        return None

    def list_entries(self, conv_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """List semua entry (opsional filter by conv_id)."""
        if conv_id:
            cur = self.conn.execute(
                "SELECT conv_id,step,tier,seq_len,size_bytes,created_at "
                "FROM kv_index WHERE conv_id=? ORDER BY step",
                (conv_id,)
            )
        else:
            cur = self.conn.execute(
                "SELECT conv_id,step,tier,seq_len,size_bytes,created_at "
                "FROM kv_index ORDER BY created_at DESC LIMIT 100"
            )
        cols = ["conv_id","step","tier","seq_len","size_bytes","created_at"]
        return [dict(zip(cols, row)) for row in cur.fetchall()]

    def save_full(self, conv_id: str, step: int, kv: KVCache,
                  metadata: Optional[Dict] = None) -> Path:
        """Simpan full KV-cache (dipindah ke CPU sebelum simpan)."""
        metadata = metadata or {}
        filepath = self.full_dir / f"{conv_id}_{step:03d}.pt"
        torch.save(_kv_to_cpu(kv), filepath)
        self._index_add(conv_id, step, "full", filepath, kv, metadata)
        return filepath

    def save_selective(self, conv_id: str, step: int, kv: KVCache,
                       n_tokens: int = 64,
                       metadata: Optional[Dict] = None) -> Path:
        """
        Simpan N token terakhir dari KV-cache.

        KV-cache shape per layer: [batch, n_heads, seq_len, head_dim]
        Slice pada dim=-2 (seq_len): kv[..., -n_tokens:, :]

        Kenapa token terakhir?
          Dalam transformer causal, token posisi akhir dalam KV-cache
          sudah melalui attention dengan semua token sebelumnya.
          Konteks terkini paling relevan untuk prediksi berikutnya
          tersimpan di sini.
        """
        metadata = metadata or {}
        seq_len  = _past_length(kv)
        keep     = min(n_tokens, seq_len)

        selective: KVCache = tuple(
            tuple(t[..., -keep:, :].cpu() for t in layer)
            for layer in kv
        )

        filepath = self.sel_dir / f"{conv_id}_{step:03d}_sel.pt"
        torch.save(selective, filepath)
        self._index_add(conv_id, step, "selective", filepath, selective,
                        {**metadata, "n_tokens": keep, "original_seq_len": seq_len})
        return filepath

    def load(self, conv_id: str, step: int, tier: str = "full",
             device: Optional[torch.device] = None) -> Optional[KVCache]:
        """Load KV-cache. Opsional pindah ke device setelah load."""
        filepath = self.lookup(conv_id, step, tier)
        if filepath is None:
            return None
        kv = torch.load(filepath, map_location="cpu", weights_only=True)
        if device is not None:
            kv = _kv_to_device(kv, device)
        return kv

    def delete(self, conv_id: str, step: int, tier: str = "full") -> bool:
        filepath = self.lookup(conv_id, step, tier)
        if filepath and filepath.exists():
            filepath.unlink()
        self.conn.execute(
            "DELETE FROM kv_index WHERE conv_id=? AND step=? AND tier=?",
            (conv_id, step, tier)
        )
        self.conn.commit()
        return filepath is not None

    def total_size_mb(self) -> float:
        cur = self.conn.execute("SELECT SUM(size_bytes) FROM kv_index")
        total = cur.fetchone()[0] or 0
        return total / (1024 ** 2)

    # ── KV-cache pruning ──────────────────────────────────────────────────

    def prune_by_age(self, max_age_seconds: float) -> int:
        """
        Delete KV-cache entries older than max_age_seconds.
        Returns number of entries deleted.
        """
        cutoff = time.time() - max_age_seconds
        cur = self.conn.execute(
            "SELECT id, filepath FROM kv_index WHERE created_at < ?",
            (cutoff,)
        )
        rows = cur.fetchall()
        for row_id, fpath in rows:
            p = Path(fpath)
            if p.exists():
                p.unlink()
        self.conn.execute(
            "DELETE FROM kv_index WHERE created_at < ?", (cutoff,)
        )
        self.conn.commit()
        return len(rows)

    def prune_by_size(self, max_total_mb: float) -> int:
        """
        Evict oldest entries until total disk usage is under max_total_mb.
        Returns number of entries deleted.
        """
        deleted = 0
        while self.total_size_mb() > max_total_mb:
            cur = self.conn.execute(
                "SELECT id, filepath FROM kv_index "
                "ORDER BY created_at ASC LIMIT 1"
            )
            row = cur.fetchone()
            if row is None:
                break
            row_id, fpath = row
            p = Path(fpath)
            if p.exists():
                p.unlink()
            self.conn.execute("DELETE FROM kv_index WHERE id=?", (row_id,))
            self.conn.commit()
            deleted += 1
        return deleted

    def prune_by_count(self, max_entries: int) -> int:
        """
        Keep only the most recent max_entries entries (per tier).
        Deletes oldest entries first. Returns number of entries deleted.
        """
        deleted = 0
        for tier in ("full", "selective"):
            cur = self.conn.execute(
                "SELECT COUNT(*) FROM kv_index WHERE tier=?", (tier,)
            )
            count = cur.fetchone()[0]
            excess = count - max_entries
            if excess <= 0:
                continue
            cur = self.conn.execute(
                "SELECT id, filepath FROM kv_index "
                "WHERE tier=? ORDER BY created_at ASC LIMIT ?",
                (tier, excess)
            )
            rows = cur.fetchall()
            for row_id, fpath in rows:
                p = Path(fpath)
                if p.exists():
                    p.unlink()
                self.conn.execute("DELETE FROM kv_index WHERE id=?", (row_id,))
            self.conn.commit()
            deleted += len(rows)
        return deleted

    def prune_keep_conv_ids(self, keep_conv_ids: List[str]) -> int:
        """
        Delete all entries EXCEPT those matching keep_conv_ids.
        Useful after evolution round: keep only best trajectories' KV-caches.
        Returns number of entries deleted.
        """
        if not keep_conv_ids:
            return 0
        placeholders = ",".join("?" for _ in keep_conv_ids)
        cur = self.conn.execute(
            f"SELECT id, filepath FROM kv_index "
            f"WHERE conv_id NOT IN ({placeholders})",
            keep_conv_ids,
        )
        rows = cur.fetchall()
        for row_id, fpath in rows:
            p = Path(fpath)
            if p.exists():
                p.unlink()
        self.conn.execute(
            f"DELETE FROM kv_index WHERE conv_id NOT IN ({placeholders})",
            keep_conv_ids,
        )
        self.conn.commit()
        return len(rows)

    def auto_prune(
        self,
        max_total_mb: float = 2048.0,
        max_age_seconds: Optional[float] = None,
        max_entries_per_tier: Optional[int] = None,
    ) -> Dict[str, int]:
        """
        Run all applicable pruning strategies. Called automatically after save
        if auto_prune is enabled on the store.

        Returns dict with counts of deleted entries per strategy.
        """
        result: Dict[str, int] = {}
        if max_age_seconds is not None:
            result["age"] = self.prune_by_age(max_age_seconds)
        if max_entries_per_tier is not None:
            result["count"] = self.prune_by_count(max_entries_per_tier)
        # Size-based pruning always runs as final guard
        result["size"] = self.prune_by_size(max_total_mb)
        return result


# =============================================================================
# BAGIAN 4 -- REALIGNMENT MATRIX
# =============================================================================
# LatentRealigner is now imported from llm._shared (single source of truth).
# See llm/_shared.py for the full implementation.

# =============================================================================
# BAGIAN 5 -- RESULT DATACLASS (output fleksibel)
# =============================================================================

@dataclass
class LLMResult:
    """
    Output fleksibel dari LocalLLMBackend.

    Tiga mode:
        "kv_only"     : kv_cache ada, text=None
                        Untuk agen yang hanya membangun konteks (propose, construct).
        "text_only"   : text ada, kv_cache=None
                        Untuk agen yang butuh teks (judger, feedback).
        "kv_and_text" : keduanya ada
                        Untuk agen yang butuh output teks DAN meneruskan KV.
    """
    text        : Optional[str]           = None
    kv_cache    : Optional[KVCache]       = None
    input_ids   : Optional[torch.Tensor]  = None   # [1, seq_len]
    output_ids  : Optional[torch.Tensor]  = None   # [1, gen_len]
    hidden_last : Optional[torch.Tensor]  = None   # [1, d]
    latent_vecs : Optional[torch.Tensor]  = None   # [steps, d]
    mode        : OutputMode              = "text_only"

    @property
    def has_text(self) -> bool:
        return self.text is not None

    @property
    def has_kv(self) -> bool:
        return self.kv_cache is not None


# =============================================================================
# BAGIAN 6 -- CORE ENGINE
# =============================================================================

_MODEL_CACHE: Dict[Tuple[str, str], Tuple[Any, Any]] = {}
_MODEL_CACHE_LOCK = threading.Lock()


def _load_or_get_cached_model(
    model_name: str,
    device: torch.device,
) -> Tuple[Any, Any]:
    """Return shared (model, tokenizer) for (model_name, device).

    Why: Multiple LocalLLMBackend instances on the same GPU previously
    each reloaded ~8 GB of Qwen3-4B weights, causing OOM. This cache
    keeps a single copy per (model, device) and hands it to every
    _CoreEngine that asks for the same pair.
    """
    cache_key = (model_name, str(device))
    with _MODEL_CACHE_LOCK:
        cached = _MODEL_CACHE.get(cache_key)
        if cached is not None:
            print(f"[CoreEngine] Reusing cached {model_name} on {device}")
            return cached

        print(f"[CoreEngine] Loading {model_name} on {device} ...")
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        _ensure_pad_token(tokenizer)

        with torch.no_grad():
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                dtype=torch.bfloat16 if torch.cuda.is_available()
                             else torch.float32,
            )

        if len(tokenizer) != model.get_input_embeddings().weight.shape[0]:
            model.resize_token_embeddings(len(tokenizer))

        model.to(device).eval()
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = True

        _MODEL_CACHE[cache_key] = (model, tokenizer)
        return model, tokenizer


class _CoreEngine:
    """
    Engine internal: model HuggingFace + tokenizer + realigner.
    Tidak dipakai langsung dari luar -- diakses lewat LocalLLMBackend.

    Khusus Qwen3-4B:
      - Mode thinking: Qwen3 punya chain-of-thought bawaan (<think>...</think>).
        Untuk pipeline QuantaAlpha yang parse JSON output, thinking harus dimatikan
        dengan menambahkan '/no_think' di system prompt (instruksi resmi Qwen3).
      - Chat template Qwen3 sudah proper -- tidak perlu fallback manual.

    Model+tokenizer di-share via _MODEL_CACHE: instance kedua dengan
    (model_name, device) yang sama tidak reload bobot dari disk/GPU.
    """

    QWEN3_NOTHINK_SUFFIX = "/no_think"

    def __init__(
        self,
        model_name     : str,
        device         : torch.device,
        latent_steps   : int   = 0,
        use_realign    : bool  = False,
        enable_thinking: bool  = False,
        knn_enabled    : bool  = False,
        knn_percentage : float = 0.8,
        knn_min_keep   : int   = 5,
        knn_strategy   : str   = "top",
    ) -> None:
        self.model_name      = model_name
        self.device          = device
        self.latent_steps    = latent_steps
        self.enable_thinking = enable_thinking

        # KNN-based KV-cache filtering (diterapkan pada past_kv dari step sebelumnya)
        self.knn_enabled    = knn_enabled
        self.knn_percentage = knn_percentage
        self.knn_min_keep   = knn_min_keep
        self.knn_strategy   = knn_strategy

        self.model, self.tokenizer = _load_or_get_cached_model(model_name, device)

        self.realigner: Optional[LatentRealigner] = None
        if latent_steps > 0:
            self.realigner = LatentRealigner(self.model, device,
                                             use_realign=use_realign)
            print(f"[CoreEngine] Realigner built (use_realign={use_realign})")

        print(f"[CoreEngine] Ready. latent_steps={latent_steps}")

    # ── Chat formatting ────────────────────────────────────────────────────

    def format_messages(
        self,
        messages: List[Dict[str, str]],
        add_generation_prompt: bool = True,
    ) -> str:
        """
        Render messages ke prompt string.

        Untuk Qwen3 dengan thinking=False:
          '/no_think' ditambahkan ke konten system message.
          Ini adalah cara resmi mematikan thinking di Qwen3.

        Args:
            add_generation_prompt: Jika True, tambahkan prefix assistant
                (e.g. ``<|im_start|>assistant\\n``) di akhir prompt.
                Set False saat latent_pass di kv_and_text mode agar
                prefix hanya ditambahkan saat generate_from_kv().
        """
        if not self.enable_thinking:
            msgs = []
            for m in messages:
                if m["role"] == "system":
                    msgs.append({
                        "role": "system",
                        "content": m["content"].rstrip() + "\n" + self.QWEN3_NOTHINK_SUFFIX
                    })
                else:
                    msgs.append(m)
        else:
            msgs = messages

        if getattr(self.tokenizer, "chat_template", None):
            return self.tokenizer.apply_chat_template(
                msgs, tokenize=False,
                add_generation_prompt=add_generation_prompt,
            )

        # Fallback (tidak diharapkan untuk Qwen3)
        parts = []
        for m in msgs:
            parts.append(f"<|{m['role']}|>\n{m.get('content','')}\n<|/{m['role']}|>")
        if add_generation_prompt:
            parts.append("<|assistant|>")
        return "\n".join(parts)

    def tokenize(self, prompt: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (input_ids [1, L], attention_mask [1, L])."""
        enc = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        return enc["input_ids"].to(self.device), enc["attention_mask"].to(self.device)

    @staticmethod
    def _extend_mask(mask: torch.Tensor, past_len: int) -> torch.Tensor:
        """Perluas attention mask untuk token yang sudah di-cache."""
        if past_len == 0:
            return mask
        past_ones = torch.ones(
            (mask.shape[0], past_len), dtype=mask.dtype, device=mask.device
        )
        return torch.cat([past_ones, mask], dim=-1)

    # ── Forward pass ────────────────────────────────────────────────────────

    @torch.no_grad()
    def _forward(
        self,
        input_ids  : torch.Tensor,
        attn_mask  : torch.Tensor,
        past_kv    : Optional[KVCache],
        need_hidden: bool = True,
    ) -> Tuple[KVCache, Optional[torch.Tensor]]:
        """Satu forward pass. Return (past_kv_baru, last_hidden atau None).

        past_kv diasumsikan sudah DynamicCache (dinormalisasi di LocalLLMBackend.run
        entry). Engine internal bekerja penuh dalam format DynamicCache; konversi
        ke tuple hanya dilakukan sekali di run() exit.
        """
        out = self.model(
            input_ids=input_ids,
            attention_mask=attn_mask,
            past_key_values=past_kv,
            use_cache=True,
            output_hidden_states=need_hidden,
            return_dict=True,
        )
        last_hidden = None
        if need_hidden:
            # [B, seq, d] -> ambil posisi terakhir -> [B, d]
            last_hidden = out.hidden_states[-1][:, -1, :]
        return out.past_key_values, last_hidden

    # ── Latent pass ──────────────────────────────────────────────────────────

    @torch.no_grad()
    def latent_pass(
        self,
        messages    : List[Dict[str, str]],
        past_kv     : Optional[KVCache] = None,
        record_vecs : bool = True,
        latent_steps: Optional[int] = None,
        add_generation_prompt: bool = True,
    ) -> Tuple[KVCache, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Forward pass + N latent steps. Tidak menghasilkan teks.

        Setiap latent step:
            1. ambil last_hidden [B, d]
            2. realign: latent_vec = apply_realignment(last_hidden)
            3. buat virtual token: embed = latent_vec.unsqueeze(1)  [B, 1, d]
            4. forward pass dengan inputs_embeds=embed (bukan input_ids)
            5. dapat last_hidden baru

        Virtual token ini tidak punya representasi teks.
        Model "berpikir diam" -- memperbarui KV-cache tanpa menulis token.

        Args:
            add_generation_prompt: Teruskan ke format_messages().
                Untuk kv_and_text mode, set False agar assistant prefix
                ditambahkan hanya saat generate_from_kv().

        Returns:
            (updated_kv, last_hidden [B, d], latent_vecs [steps, d] atau None)
        """
        prompt = self.format_messages(messages, add_generation_prompt=add_generation_prompt)
        ids, mask = self.tokenize(prompt)

        # ── KNN filter past_kv dari step sebelumnya ─────────────
        # Hitung cosine similarity antara input embeddings prompt saat ini
        # dengan key vectors di middle layer KV-cache, lalu pertahankan
        # hanya token yang paling relevan. Ini mengurangi noise dari
        # step sebelumnya dan fokuskan konteks latent.
        if self.knn_enabled and past_kv is not None and _past_length(past_kv) > 0:
            query_embeds = self.model.get_input_embeddings()(ids)
            query_hidden = query_embeds.mean(dim=1)  # [B, hidden_dim]
            past_kv = kv_knn_filter(
                past_kv, query_hidden,
                percentage=self.knn_percentage,
                min_keep=self.knn_min_keep,
                strategy=self.knn_strategy,
            )

        past_len = _past_length(past_kv)
        ext_mask = self._extend_mask(mask, past_len)

        # Forward pass pertama: encode seluruh prompt
        past, last_hidden = self._forward(ids, ext_mask, past_kv, need_hidden=True)

        _n_steps = latent_steps if latent_steps is not None else self.latent_steps
        if _n_steps == 0 or self.realigner is None:
            return past, last_hidden, None

        vecs: List[torch.Tensor] = []

        for _ in range(_n_steps):
            latent_vec   = self.realigner.apply(last_hidden, self.model)  # [B, d]
            latent_embed = latent_vec.unsqueeze(1)             # [B, 1, d]

            if record_vecs:
                vecs.append(latent_vec.detach().cpu())

            past_len    = _past_length(past)
            latent_mask = torch.ones(
                (latent_embed.shape[0], past_len + 1),
                dtype=torch.long, device=self.device,
            )

            out = self.model(
                inputs_embeds=latent_embed,
                attention_mask=latent_mask,
                past_key_values=past,
                use_cache=True,
                output_hidden_states=True,
                return_dict=True,
            )
            past = out.past_key_values
            last_hidden = out.hidden_states[-1][:, -1, :]

        latent_tensor = None
        if record_vecs and vecs:
            # [steps, d], ambil batch index 0
            latent_tensor = torch.stack([v[0] for v in vecs], dim=0)

        return past, last_hidden, latent_tensor

    # ── Generate teks ────────────────────────────────────────────────────────

    @torch.no_grad()
    def generate_text(
        self,
        messages      : List[Dict[str, str]],
        past_kv       : Optional[KVCache] = None,
        max_new_tokens : int   = 2048,
        temperature    : float = 0.6,
        top_p          : float = 0.95,
        return_kv      : bool  = False,
        prefix_allowed_tokens_fn: Optional[Any] = None,
    ) -> Tuple[str, torch.Tensor, torch.Tensor, Optional[KVCache]]:
        """
        Generate teks. Opsional kembalikan KV-cache sesudah generate.

        Args:
            prefix_allowed_tokens_fn: Callable (batch_id, input_ids) -> list[int]
                yang membatasi token boleh-keluar di setiap step dekoder.
                Diteruskan ke model.generate() untuk guided decoding
                (contoh: enforce JSON schema via lm-format-enforcer).

        Returns:
            (text, input_ids [1,L], output_ids [1,G], kv atau None)
        """
        prompt = self.format_messages(messages)
        ids, mask = self.tokenize(prompt)
        prompt_len = int(mask.sum())

        # ── KNN filter past_kv (sama seperti di latent_pass) ────
        if self.knn_enabled and past_kv is not None and _past_length(past_kv) > 0:
            query_embeds = self.model.get_input_embeddings()(ids)
            query_hidden = query_embeds.mean(dim=1)
            past_kv = kv_knn_filter(
                past_kv, query_hidden,
                percentage=self.knn_percentage,
                min_keep=self.knn_min_keep,
                strategy=self.knn_strategy,
            )

        past_len      = _past_length(past_kv)
        ext_mask      = self._extend_mask(mask, past_len)
        cache_position = None
        if past_len > 0:
            cache_position = torch.arange(
                past_len, past_len + ids.shape[-1],
                dtype=torch.long, device=self.device,
            )

        out = self.model.generate(
            input_ids=ids,
            attention_mask=ext_mask,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=self.tokenizer.pad_token_id,
            return_dict_in_generate=True,
            output_scores=False,
            past_key_values=past_kv,
            cache_position=cache_position,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
        )

        generated_ids = out.sequences[0, prompt_len:]
        text = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

        if not self.enable_thinking:
            text = self._strip_thinking(text)

        kv_out = out.past_key_values if return_kv else None
        return text, ids, generated_ids.unsqueeze(0), kv_out

    # ── Generation from existing KV (kv_and_text mode) ──────────────────

    def _get_generation_prefix_ids(
        self, messages: List[Dict[str, str]],
    ) -> torch.Tensor:
        """
        Ekstrak token IDs untuk assistant generation prefix.

        Cara kerja:
            1. Tokenize prompt DENGAN add_generation_prompt=True
            2. Tokenize prompt TANPA add_generation_prompt
            3. Selisih = token prefix assistant (misal ``<|im_start|>assistant\\n``)

        Menggunakan tokenisasi full-text untuk menjaga BPE merge di batas.
        """
        prompt_with = self.format_messages(messages, add_generation_prompt=True)
        prompt_without = self.format_messages(messages, add_generation_prompt=False)

        ids_with = self.tokenizer(
            prompt_with, add_special_tokens=False
        )["input_ids"]
        ids_without = self.tokenizer(
            prompt_without, add_special_tokens=False
        )["input_ids"]

        prefix_ids = ids_with[len(ids_without):]
        if not prefix_ids:
            # Fallback: tokenize newline sebagai trigger minimal
            prefix_ids = self.tokenizer(
                "\n", add_special_tokens=False
            )["input_ids"][-1:]

        return torch.tensor([prefix_ids], dtype=torch.long, device=self.device)

    @torch.no_grad()
    def generate_from_kv(
        self,
        past_kv        : KVCache,
        messages       : List[Dict[str, str]],
        max_new_tokens : int   = 2048,
        temperature    : float = 0.6,
        top_p          : float = 0.95,
        return_kv      : bool  = True,
        prefix_allowed_tokens_fn: Optional[Any] = None,
    ) -> Tuple[str, torch.Tensor, torch.Tensor, Optional[KVCache]]:
        """
        Generate teks dari KV-cache yang sudah ada TANPA re-encode pesan.

        Dipakai setelah latent_pass() di kv_and_text mode.

        Flow:
            latent_pass(messages, add_generation_prompt=False)
                → KV berisi [prompt tanpa assistant prefix + latent steps]
            generate_from_kv(kv, messages)
                → Kirim HANYA assistant prefix tokens sebagai input_ids
                → Model generate response dari konteks latent

        Ini menghindari double-encoding: prompt hanya diproses sekali
        (di latent_pass), dan generate hanya menambahkan trigger minimal.

        Returns:
            (text, prefix_ids [1, P], output_ids [1, G], kv atau None)
        """
        prefix_ids = self._get_generation_prefix_ids(messages)
        prefix_len = prefix_ids.shape[-1]

        past_len = _past_length(past_kv)
        mask = torch.ones(
            (1, past_len + prefix_len), dtype=torch.long, device=self.device,
        )
        cache_position = torch.arange(
            past_len, past_len + prefix_len,
            dtype=torch.long, device=self.device,
        )

        out = self.model.generate(
            input_ids=prefix_ids,
            attention_mask=mask,
            past_key_values=past_kv,
            cache_position=cache_position,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=self.tokenizer.pad_token_id,
            return_dict_in_generate=True,
            output_scores=False,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
        )

        # out.sequences = [prefix_ids + generated_ids]
        generated_ids = out.sequences[0, prefix_len:]
        text = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

        if not self.enable_thinking:
            text = self._strip_thinking(text)

        kv_out = out.past_key_values if return_kv else None
        return text, prefix_ids, generated_ids.unsqueeze(0), kv_out

    @staticmethod
    def _strip_thinking(text: str) -> str:
        """Hapus blok <think>...</think> dari output Qwen3."""
        return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


# =============================================================================
# BAGIAN 7 -- LocalLLMBackend (API publik utama)
# =============================================================================

class LocalLLMBackend:
    """
    Backend LLM lokal sebagai pengganti APIBackend di 

    Kompatibel dengan APIBackend (client.py):
        build_messages()
        build_messages_and_create_chat_completion()
        build_chat_session()

    Tambahan untuk latent reasoning:
        run()                 <- titik masuk tunggal, mode fleksibel
        run_latent_pass()     <- shortcut kv_only
        run_generate()        <- shortcut text_only atau kv_and_text
        load_kv()             <- resume dari KV-cache tersimpan
        debug_step()          <- decode tensor log untuk debugging

    Args:
        model_name      : nama model HuggingFace
        device          : "cuda", "cuda:0", "cpu"
        latent_steps    : jumlah latent step per agen (0 = nonaktif)
        use_realign     : aktifkan proyeksi realignment
        enable_thinking : aktifkan thinking Qwen3 (default False = hemat token)
        log_tensors     : simpan tensor ke TensorConvManager
        store_kv        : simpan KV-cache ke disk
        conv_dir        : direktori log tensor
        kv_dir          : direktori KV-cache
        max_new_tokens  : batas token output
        temperature     : sampling temperature
        top_p           : nucleus sampling
    """

    DEFAULT_SYSTEM_PROMPT = "You are a helpful quantitative finance AI assistant."

    def __init__(
        self,
        model_name     : str   = "Qwen/Qwen3-14B",
        device         : str   = "cuda",
        latent_steps   : int   = 0,
        use_realign    : bool  = False,
        enable_thinking: bool  = False,
        log_tensors    : bool  = True,
        store_kv       : bool  = False,
        conv_dir       : str   = "./debug/conv_logs",
        kv_dir         : str   = "./debug/kv_store",
        output_log_dir : str   = "./debug/llm_outputs",
        max_new_tokens : int   = 2048,
        temperature    : float = 0.6,
        top_p          : float = 0.95,
        kv_prune_max_mb        : float = 2048.0,
        kv_prune_max_age_secs  : Optional[float] = None,
        kv_prune_max_entries   : Optional[int]    = None,
        kv_max_seq_len         : Optional[int]    = None,
        # KNN-based KV-cache filtering
        knn_enabled    : bool  = False,
        knn_percentage : float = 0.8,
        knn_min_keep   : int   = 5,
        knn_strategy   : str   = "top",
    ) -> None:

        self.max_new_tokens = max_new_tokens
        self.temperature    = temperature
        self.top_p          = top_p
        self.log_tensors    = log_tensors
        self.store_kv       = store_kv

        # KV-cache pruning config
        self._kv_prune_max_mb       = kv_prune_max_mb
        self._kv_prune_max_age_secs = kv_prune_max_age_secs
        self._kv_prune_max_entries  = kv_prune_max_entries
        self._kv_max_seq_len        = kv_max_seq_len

        _device = torch.device(
            device if (torch.cuda.is_available() or device == "cpu") else "cpu"
        )

        self._lock     = threading.Lock()

        self._engine   = _CoreEngine(
            model_name=model_name, device=_device,
            latent_steps=latent_steps, use_realign=use_realign,
            enable_thinking=enable_thinking,
            knn_enabled=knn_enabled, knn_percentage=knn_percentage,
            knn_min_keep=knn_min_keep, knn_strategy=knn_strategy,
        )
        self._conv_mgr = TensorConvManager(conv_dir) if log_tensors else None
        self._kv_store = KVCacheStore(kv_dir)        if store_kv    else None

        # ── Per-call LLM output log (append-mode JSONL) ──────────────────
        # Ditulis SEGERA setelah text di-generate (bukan di akhir loop),
        # sehingga crash tengah jalan pun tetap meninggalkan jejak output
        # LLM yang sudah tereksekusi. Berguna untuk debugging proposal/
        # coder yang gagal validasi berulang.
        try:
            _log_dir = Path(output_log_dir)
            _log_dir.mkdir(parents=True, exist_ok=True)
            _date_stamp = time.strftime("%Y%m%d_%H%M%S")
            self._output_log_path: Optional[Path] = (
                _log_dir / f"llm_outputs_{_date_stamp}.jsonl"
            )
        except Exception:
            self._output_log_path = None

    # ── Kompatibilitas APIBackend ──────────────────────────────────────────

    def build_messages(
        self,
        user_prompt    : str,
        system_prompt  : Optional[str]      = None,
        former_messages: Optional[List[Dict]] = None,
    ) -> List[Dict[str, str]]:
        """Susun messages [system, history, user]. Identik dengan APIBackend."""
        msgs = [{"role": "system",
                 "content": system_prompt or self.DEFAULT_SYSTEM_PROMPT}]
        if former_messages:
            msgs.extend(former_messages)
        msgs.append({"role": "user", "content": user_prompt})
        return msgs

    def build_messages_and_create_chat_completion(
        self,
        user_prompt    : str,
        system_prompt  : Optional[str]      = None,
        former_messages: Optional[List[Dict]] = None,
        *,
        json_mode      : bool = False,
        past_key_values: Optional[KVCache]  = None,
        max_new_tokens : Optional[int]      = None,
        temperature    : Optional[float]    = None,
        top_p          : Optional[float]    = None,
        conv_id        : Optional[str]      = None,
        step           : int = 0,
        role           : str = "assistant",
        mode           : Optional[OutputMode] = None,
        latent_steps   : Optional[int] = None,
        json_schema    : Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Drop-in replacement untuk APIBackend.build_messages_and_create_chat_completion.

        Tambahan dibanding APIBackend:
          past_key_values : KV-cache dari agen sebelumnya
          conv_id/step/role : untuk logging TensorConvManager
          mode            : output mode override.  Default: "kv_and_text"
                            jika latent_steps > 0 atau past_key_values ada,
                            otherwise "text_only" (backward compatible).
        """
        if mode is None:
            # Auto-select: use kv_and_text when latent steps are configured
            # or external KV-cache is provided, otherwise text_only
            if self._engine.latent_steps > 0 or past_key_values is not None:
                mode = "kv_and_text"
            else:
                mode = "text_only"

        messages = self.build_messages(user_prompt, system_prompt, former_messages)
        result   = self.run(
            messages=messages, mode=mode,
            past_key_values=past_key_values,
            max_new_tokens=max_new_tokens, temperature=temperature, top_p=top_p,
            json_mode=json_mode, conv_id=conv_id, step=step, role=role,
            latent_steps=latent_steps, json_schema=json_schema,
        )
        return result.text or ""

    def build_messages_and_run(
        self,
        user_prompt    : str,
        system_prompt  : Optional[str]      = None,
        former_messages: Optional[List[Dict]] = None,
        *,
        json_mode      : bool = False,
        past_key_values: Optional[KVCache]  = None,
        max_new_tokens : Optional[int]      = None,
        temperature    : Optional[float]    = None,
        top_p          : Optional[float]    = None,
        conv_id        : Optional[str]      = None,
        step           : int = 0,
        role           : str = "assistant",
        mode           : Optional[OutputMode] = None,
        latent_steps   : Optional[int] = None,
        json_schema    : Optional[Dict[str, Any]] = None,
    ) -> LLMResult:
        """
        Sama seperti build_messages_and_create_chat_completion, tapi return
        LLMResult lengkap (termasuk kv_cache, hidden_last, latent_vecs).

        Dipakai oleh Latent pipeline classes (LatentHypothesisGen, dsb.)
        yang butuh akses ke KV-cache output untuk di-chain ke step berikutnya.
        """
        if mode is None:
            if self._engine.latent_steps > 0 or past_key_values is not None:
                mode = "kv_and_text"
            else:
                mode = "text_only"

        messages = self.build_messages(user_prompt, system_prompt, former_messages)
        return self.run(
            messages=messages, mode=mode,
            past_key_values=past_key_values,
            max_new_tokens=max_new_tokens, temperature=temperature, top_p=top_p,
            json_mode=json_mode, conv_id=conv_id, step=step, role=role,
            latent_steps=latent_steps, json_schema=json_schema,
        )

    # ── Titik masuk fleksibel ──────────────────────────────────────────────

    def run(
        self,
        messages           : List[Dict[str, str]],
        mode               : OutputMode = "text_only",
        past_key_values    : Optional[KVCache]  = None,
        max_new_tokens     : Optional[int]      = None,
        temperature        : Optional[float]    = None,
        top_p              : Optional[float]    = None,
        json_mode          : bool = False,
        record_latent_vecs : bool = True,
        conv_id            : Optional[str] = None,
        step               : int  = 0,
        role               : str  = "agent",
        kv_n_selective     : int  = 64,
        latent_steps       : Optional[int] = None,
        json_schema        : Optional[Dict[str, Any]] = None,
    ) -> LLMResult:
        """
        Titik masuk tunggal untuk semua mode.

        mode="kv_only":
            Jalankan latent pass, kembalikan KV-cache.
            Gunakan untuk: propose, construct — agen yang hanya membangun konteks.
            text=None, kv_cache ada.

        mode="text_only":
            Generate teks, tidak kembalikan KV-cache.
            Gunakan untuk: judger, feedback — agen yang butuh output string.
            text ada, kv_cache=None.

        mode="kv_and_text":
            Latent pass dulu, lalu generate teks dari KV yang dibangun.
            Gunakan untuk: agen yang butuh KEDUANYA.
            text ada, kv_cache ada.

        Args:
            messages          : list message
            mode              : output mode
            past_key_values   : KV-cache dari agen sebelumnya
            json_mode         : auto-extract + fix JSON dari output
            record_latent_vecs: catat latent vectors ke log
            conv_id           : ID percakapan (auto-generate jika None)
            step              : nomor langkah pipeline
            role              : nama agen
            kv_n_selective    : jumlah token untuk selective cache
            latent_steps      : override jumlah latent steps (None = pakai default engine)
        """
        _conv_id = conv_id or str(uuid.uuid4())[:8]
        _max_tok = max_new_tokens or self.max_new_tokens
        _temp    = temperature    or self.temperature
        _top_p   = top_p          or self.top_p

        # ── Build guided-decoding prefix_fn (opsional) ──────────────────────
        # Jika `json_schema` di-supply, bangun prefix_allowed_tokens_fn dari
        # lm-format-enforcer — di setiap step dekoder, hanya token yang
        # melanjutkan parse JSON valid terhadap schema yang boleh keluar.
        # Diterapkan hanya pada mode yang generate text (bukan kv_only).
        _prefix_fn = None
        if json_schema is not None and mode != "kv_only":
            from llm.guided_decoding import build_guided_json_prefix_fn
            _prefix_fn = build_guided_json_prefix_fn(
                self._engine.tokenizer, json_schema,
            )
            print(
                f"[GuidedDecoding] role={role}, mode={mode}: "
                f"enforcing JSON schema via prefix_allowed_tokens_fn"
            )

        # Pipeline monitor (safe — no-op if unavailable)
        try:
            from debug import get_monitor as _get_mon
            _mon = _get_mon(auto_create=False)
        except ImportError:
            _mon = None

        _run_t0 = time.time()
        if _mon:
            _input_tok_count = 0
            try:
                prompt_text = self._engine.format_messages(messages)
                _ids_tmp, _ = self._engine.tokenize(prompt_text)
                _input_tok_count = _ids_tmp.shape[-1] if _ids_tmp is not None else 0
            except Exception:
                pass
            _mon.track_llm_call_start(
                caller=role, mode=mode,
                input_tokens=_input_tok_count,
                temperature=_temp or 0.0,
                latent_steps=latent_steps or self._engine.latent_steps,
                has_past_kv=past_key_values is not None,
            )

        with self._lock:
            result = LLMResult(mode=mode)

            # ── Normalize past_kv ke DynamicCache (boundary masuk) ────
            # Engine internal bekerja penuh dalam DynamicCache; konversi
            # dilakukan SEKALI di sini (bukan di setiap method _CoreEngine)
            # untuk menghindari alokasi wrapper berulang & fragmentasi.
            if past_key_values is not None and isinstance(past_key_values, tuple):
                from transformers import DynamicCache
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)

            # ── KNN filter logging ────────────────────────────────────
            if (self._engine.knn_enabled
                    and past_key_values is not None
                    and _past_length(past_key_values) > 0):
                _pre_len = _past_length(past_key_values)
                _post_len = max(
                    int(_pre_len * self._engine.knn_percentage),
                    self._engine.knn_min_keep,
                )
                _post_len = min(_post_len, _pre_len)
                print(
                    f"[KNN] role={role}, mode={mode}, "
                    f"past_kv: {_pre_len} → ~{_post_len} tokens "
                    f"({self._engine.knn_percentage:.0%}, "
                    f"strategy={self._engine.knn_strategy})"
                )

            # ── kv_only ────────────────────────────────────────────────
            if mode == "kv_only":
                kv, last_hidden, latent_vecs = self._engine.latent_pass(
                    messages, past_key_values, record_vecs=record_latent_vecs,
                    latent_steps=latent_steps,
                )
                result.kv_cache    = kv
                result.hidden_last = last_hidden
                result.latent_vecs = latent_vecs

                prompt = self._engine.format_messages(messages)
                ids, _ = self._engine.tokenize(prompt)
                result.input_ids = ids

            # ── text_only ──────────────────────────────────────────────
            elif mode == "text_only":
                text, ids, out_ids, _ = self._engine.generate_text(
                    messages, past_key_values,
                    max_new_tokens=_max_tok, temperature=_temp, top_p=_top_p,
                    return_kv=False,
                    prefix_allowed_tokens_fn=_prefix_fn,
                )
                if json_mode:
                    text = self._fix_json(text)
                result.text       = text
                result.input_ids  = ids
                result.output_ids = out_ids

            # ── kv_and_text ────────────────────────────────────────────
            elif mode == "kv_and_text":
                # Step 1: Latent pass TANPA assistant prefix.
                #   KV berisi: [prompt tokens + latent virtual tokens]
                #   Model "berpikir diam" di ruang latent sebelum menjawab.
                kv, last_hidden, latent_vecs = self._engine.latent_pass(
                    messages, past_key_values, record_vecs=record_latent_vecs,
                    latent_steps=latent_steps,
                    add_generation_prompt=False,
                )

                # Step 2: Generate teks dari KV TANPA re-encode pesan.
                #   Hanya kirim assistant prefix tokens (e.g. <|im_start|>assistant\n)
                #   sebagai trigger. Menghindari double-encoding prompt.
                #
                # Catatan KV-chain (ANTI-PATTERN CONTAMINATION):
                #   HuggingFace DynamicCache dimutasi in-place oleh
                #   model.generate — `kv` hasil latent_pass akan ter-append
                #   assistant prefix + answer tokens setelah generate selesai.
                #   Kalau seluruh KV ini di-chain ke step berikutnya, answer
                #   tokens step sekarang menjadi konteks-terdekat bagi step
                #   berikutnya → pattern-match ke schema jawaban sebelumnya
                #   (misal propose flat JSON mem-bias construct yang
                #   seharusnya nested). Maka kita CROP kembali ke panjang
                #   pre-generation sebelum di-store — konsisten dengan
                #   filosofi Latent-MAS: yang di-pass antar agent adalah
                #   latent reasoning (virtual tokens), bukan discrete
                #   answer tokens.
                _latent_kv_len = _past_length(kv)
                text, prefix_ids, out_ids, _ = self._engine.generate_from_kv(
                    past_kv=kv, messages=messages,
                    max_new_tokens=_max_tok, temperature=_temp, top_p=_top_p,
                    return_kv=False,
                    prefix_allowed_tokens_fn=_prefix_fn,
                )
                try:
                    kv.crop(_latent_kv_len)
                except AttributeError:
                    pass
                if json_mode:
                    text = self._fix_json(text)

                # input_ids untuk logging: tokenize full prompt (murah, no forward pass)
                full_prompt = self._engine.format_messages(messages)
                full_ids, _ = self._engine.tokenize(full_prompt)

                result.text        = text
                result.kv_cache    = kv
                result.input_ids   = full_ids
                result.output_ids  = out_ids
                result.hidden_last = last_hidden
                result.latent_vecs = latent_vecs

            # ── Simpan output LLM ke disk SEGERA (flush per-call) ─────
            # Tulis append-mode sebelum proses lain — kalau pipeline crash
            # di tensor logging / KV store, snapshot text tetap persisten.
            if result.text is not None and mode != "kv_only":
                self._save_output_snapshot(
                    conv_id=_conv_id, step=step, role=role, mode=mode,
                    messages=messages, text=result.text,
                    temperature=_temp, has_past_kv=past_key_values is not None,
                )

            # ── Logging tensor ─────────────────────────────────────────
            if self._conv_mgr is not None:
                self._conv_mgr.save(ConvRecord(
                    conv_id     = _conv_id,
                    step        = step,
                    role        = role,
                    input_ids   = result.input_ids,
                    output_ids  = result.output_ids,
                    hidden_last = result.hidden_last,
                    latent_vecs = result.latent_vecs,
                    metadata    = {
                        "mode" : mode,
                        "model": self._engine.model_name,
                        "ts"   : time.time(),
                    },
                ))

            # ── Konversi DynamicCache → tuple (boundary keluar) ────────
            # Downstream code (loop.py, evolution, trajectory) pakai format
            # tuple legacy. Konversi dilakukan SEKALI di sini, bukan di tiap
            # method _CoreEngine. `to_legacy_cache()` cuma restructure
            # reference list — tidak menyalin tensor.
            if result.kv_cache is not None and hasattr(result.kv_cache, 'to_legacy_cache'):
                result.kv_cache = result.kv_cache.to_legacy_cache()

            # ── In-memory KV-cache truncation ─────────────────────────
            if result.kv_cache is not None and self._kv_max_seq_len is not None:
                result.kv_cache = kv_truncate(result.kv_cache, self._kv_max_seq_len)

            # ── Simpan KV-cache ke disk ────────────────────────────────
            if self._kv_store is not None and result.kv_cache is not None:
                self._kv_store.save_full(
                    _conv_id, step, result.kv_cache, metadata={"role": role, "mode": mode}
                )
                self._kv_store.save_selective(
                    _conv_id, step, result.kv_cache,
                    n_tokens=kv_n_selective, metadata={"role": role, "mode": mode}
                )
                # Auto-prune disk storage
                self._kv_store.auto_prune(
                    max_total_mb=self._kv_prune_max_mb,
                    max_age_seconds=self._kv_prune_max_age_secs,
                    max_entries_per_tier=self._kv_prune_max_entries,
                )

        # ── Pipeline monitor: record LLM call end + quality ──────────
        if _mon:
            try:
                _run_dur = time.time() - _run_t0
                _out_tok = result.output_ids.shape[-1] if result.output_ids is not None else 0
                _tok_sec = _out_tok / _run_dur if _run_dur > 0 else 0
                _mon.track_llm_call_end(
                    caller=role, duration_s=_run_dur,
                    output_tokens=_out_tok, tokens_per_sec=_tok_sec,
                    total_tokens=(_input_tok_count + _out_tok) if '_input_tok_count' in dir() else _out_tok,
                    mode=mode,
                )
                if result.text:
                    _out_ids_list = result.output_ids[0].tolist() if result.output_ids is not None else None
                    _mon.analyze_llm_output(result.text, caller=role, token_ids=_out_ids_list)
                if result.hidden_last is not None:
                    _mon.check_tensor(result.hidden_last, name=f"{role}_hidden_last")
                if result.latent_vecs is not None:
                    _mon.check_tensor(result.latent_vecs, name=f"{role}_latent_vecs")
            except Exception:
                pass

        return result

    # ── Shortcut methods ─────────────��────────────────────────────────────

    def run_latent_pass(
        self,
        messages       : List[Dict[str, str]],
        past_key_values: Optional[KVCache] = None,
        **kwargs
    ) -> LLMResult:
        """Shortcut mode='kv_only'. Untuk agen yang hanya membangun konteks."""
        return self.run(messages, mode="kv_only",
                        past_key_values=past_key_values, **kwargs)

    def run_generate(
        self,
        messages       : List[Dict[str, str]],
        past_key_values: Optional[KVCache] = None,
        return_kv      : bool = False,
        **kwargs
    ) -> LLMResult:
        """
        Shortcut untuk generate teks.
        return_kv=False -> text_only
        return_kv=True  -> kv_and_text
        """
        mode = "kv_and_text" if return_kv else "text_only"
        return self.run(messages, mode=mode,
                        past_key_values=past_key_values, **kwargs)

    # ── Session (kompatibel dengan ChatSession di client.py) ──────────────

    def build_chat_session(
        self,
        conversation_id      : Optional[str] = None,
        session_system_prompt: Optional[str] = None,
    ) -> "LocalChatSession":
        """Buat session multi-turn. Compatible dengan APIBackend.build_chat_session."""
        return LocalChatSession(
            backend=self,
            conversation_id=conversation_id,
            system_prompt=session_system_prompt,
        )

    # ── Utilitas ──────────────────────────────────────────────────────────

    def load_kv(self, conv_id: str, step: int,
                tier: str = "full") -> Optional[KVCache]:
        """Load KV-cache dari disk untuk resume evolution."""
        if self._kv_store is None:
            raise RuntimeError("store_kv=False, KVCacheStore tidak aktif.")
        return self._kv_store.load(conv_id, step, tier,
                                   device=self._engine.device)

    def debug_step(self, conv_id: str, step: int, role: str) -> Dict[str, str]:
        """Decode tensor log satu langkah untuk debugging."""
        if self._conv_mgr is None:
            return {"error": "log_tensors=False, TensorConvManager tidak aktif."}
        return self._conv_mgr.decode_step(
            conv_id, step, role, self._engine.tokenizer
        )

    def kv_store_info(self, conv_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """List metadata KV-cache yang tersimpan."""
        if self._kv_store is None:
            return []
        return self._kv_store.list_entries(conv_id)

    def kv_total_size_mb(self) -> float:
        if self._kv_store is None:
            return 0.0
        return self._kv_store.total_size_mb()

    def get_info(self) -> Dict[str, Any]:
        """Info singkat untuk monitoring."""
        info: Dict[str, Any] = {
            "model"        : self._engine.model_name,
            "device"       : str(self._engine.device),
            "latent_steps" : self._engine.latent_steps,
            "use_realign"  : (self._engine.realigner is not None and
                              self._engine.realigner.use_realign),
            "thinking"     : self._engine.enable_thinking,
            "log_tensors"  : self._conv_mgr is not None,
            "store_kv"     : self._kv_store is not None,
        }
        if self._kv_store:
            info["kv_store_mb"] = round(self.kv_total_size_mb(), 2)
        return info

    # ── Kompatibilitas APIBackend: token counting & embedding ──────────

    def build_messages_and_calculate_token(
        self,
        user_prompt  : str,
        system_prompt: Optional[str] = None,
    ) -> int:
        """Hitung jumlah token. Compatible dengan APIBackend."""
        msgs = self.build_messages(user_prompt, system_prompt)
        prompt = self._engine.format_messages(msgs)
        ids, _ = self._engine.tokenize(prompt)
        return ids.shape[-1]

    @torch.no_grad()
    def create_embedding(
        self,
        input_content: Union[str, List[str]],
    ) -> List[List[float]]:
        """
        Buat embedding menggunakan model lokal (mean-pool input embeddings).
        Compatible dengan APIBackend.create_embedding.
        """
        if isinstance(input_content, str):
            input_content = [input_content]
        embeddings = []
        embed_layer = self._engine.model.get_input_embeddings()
        for text in input_content:
            encoded = self._engine.tokenizer(
                text, return_tensors="pt", truncation=True, max_length=512,
            )
            ids = encoded["input_ids"].to(self._engine.device)
            mask = encoded["attention_mask"].to(self._engine.device)
            with self._lock:
                emb = embed_layer(ids)  # [1, seq, dim]
            # mean pool over non-padding tokens
            mask_expanded = mask.unsqueeze(-1).float()
            pooled = (emb * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1e-6)
            embeddings.append(pooled.squeeze(0).float().cpu().tolist())
        return embeddings

    def _save_output_snapshot(
        self,
        conv_id     : str,
        step        : int,
        role        : str,
        mode        : str,
        messages    : List[Dict[str, str]],
        text        : str,
        temperature : float,
        has_past_kv : bool,
    ) -> None:
        """Append satu baris JSONL berisi output LLM ke file log.

        Ditulis dengan flush eksplisit supaya isi segera ada di disk
        (bukan ditahan di buffer) — kalau pipeline crash tengah iterasi,
        riwayat output yang sudah tereksekusi tetap bisa di-review.

        Tidak pernah raise: exception di-swallow agar logging tidak
        mematikan pipeline utama.
        """
        if self._output_log_path is None:
            return
        try:
            # Ambil user/system prompt dari messages list. Tidak truncate —
            # prompt bisa panjang, tapi JSONL satu baris per record aman.
            sys_prompt = ""
            usr_prompt = ""
            for m in messages:
                if m.get("role") == "system" and not sys_prompt:
                    sys_prompt = m.get("content", "") or ""
                elif m.get("role") == "user":
                    usr_prompt = m.get("content", "") or ""
            record = {
                "ts"          : time.strftime("%Y-%m-%d %H:%M:%S"),
                "conv_id"     : conv_id,
                "step"        : step,
                "role"        : role,
                "mode"        : mode,
                "temperature" : temperature,
                "has_past_kv" : has_past_kv,
                "text_len"    : len(text),
                "text"        : text,
                "system_prompt": sys_prompt,
                "user_prompt" : usr_prompt,
            }
            with open(self._output_log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                f.flush()
                os.fsync(f.fileno())  # paksa write ke disk, tahan crash
        except Exception:
            # Logging TIDAK boleh mematikan pipeline
            pass

    @staticmethod
    def _fix_json(text: str) -> str:
        """Ekstrak + perbaiki JSON dari output LLM.

        Mencari pasangan {…} dari kanan ke kiri supaya JSON yang muncul
        di akhir response (setelah penjelasan panjang) tetap bisa diekstrak.
        """
        t = text.strip()
        fence = re.search(r"```json\s*(.*?)```", t, re.DOTALL | re.IGNORECASE)
        if fence:
            t = fence.group(1).strip()

        def _try_latex_fix(s: str) -> Optional[str]:
            fixed = s
            for cmd in ["text","frac","left","right","times","cdot",
                        "sqrt","sum","prod","int","alpha","beta","gamma","delta"]:
                fixed = re.sub(rf"(?<!\\)\\({cmd})", r"\\\\\1", fixed)
            fixed = re.sub(r"(?<!\\)\\([_\{}\[\]])", r"\\\\\1", fixed)
            try:
                json.loads(fixed)
                return fixed
            except json.JSONDecodeError:
                return None

        # Scan dari kanan ke kiri: coba setiap pasangan } … { yang valid
        end = len(t)
        while end > 0:
            e = t.rfind("}", 0, end)
            if e == -1:
                break
            s = t.rfind("{", 0, e + 1)
            if s == -1:
                break
            candidate = t[s:e + 1]
            try:
                json.loads(candidate)
                return candidate
            except json.JSONDecodeError:
                fixed = _try_latex_fix(candidate)
                if fixed is not None:
                    return fixed
            end = e  # geser pointer sebelum } ini, coba pasangan berikutnya

        return text


# =============================================================================
# BAGIAN 8 -- LOCAL CHAT SESSION
# =============================================================================

class LocalChatSession:
    """
    Multi-turn session. History di memory.
    Compatible dengan ChatSession di client.py.
    """

    def __init__(
        self,
        backend        : LocalLLMBackend,
        conversation_id: Optional[str] = None,
        system_prompt  : Optional[str] = None,
    ) -> None:
        self.backend         = backend
        self.conversation_id = conversation_id or str(uuid.uuid4())
        self.system_prompt   = system_prompt or LocalLLMBackend.DEFAULT_SYSTEM_PROMPT
        self._history: List[Dict[str, str]] = []

    def build_chat_completion(self, user_prompt: str, **kwargs) -> str:
        resp = self.backend.build_messages_and_create_chat_completion(
            user_prompt=user_prompt,
            system_prompt=self.system_prompt,
            former_messages=self._history,
            **kwargs,
        )
        self._history.append({"role": "user",      "content": user_prompt})
        self._history.append({"role": "assistant", "content": resp})
        return resp

    def get_conversation_id(self) -> str:
        return self.conversation_id

    def clear_history(self) -> None:
        self._history = []


# =============================================================================
# BAGIAN 9 -- FACTORY
# =============================================================================

def get_local_backend(
    model_name  : str  = "Qwen/Qwen3-14B",
    latent_steps: int  = 0,
    use_realign : bool = False,
    device      : str  = "cuda",
    **kwargs,
) -> LocalLLMBackend:
    """
    Factory untuk LocalLLMBackend.

    Contoh:
        # Mode standar (drop-in APIBackend, tanpa latent)
        backend = get_local_backend()

        # Mode latent 5 steps dengan realignment
        backend = get_local_backend(latent_steps=5, use_realign=True)

        # Mode latent + simpan KV ke disk (untuk evolution loop)
        backend = get_local_backend(latent_steps=5, store_kv=True, kv_dir="./kv")
    """
    return LocalLLMBackend(
        model_name=model_name, device=device,
        latent_steps=latent_steps, use_realign=use_realign,
        **kwargs,
    )


# =============================================================================
# BAGIAN 10 -- EMBEDDING UTILITIES (kompatibilitas APIBackend)
# =============================================================================

# Singleton backend untuk embedding operations (lazy-init)
_embedding_backend: Optional[LocalLLMBackend] = None


def _get_embedding_backend() -> LocalLLMBackend:
    """Dapatkan/buat singleton backend untuk embedding utilities."""
    global _embedding_backend
    if _embedding_backend is None:
        _embedding_backend = LocalLLMBackend(latent_steps=0)
    return _embedding_backend


def calculate_embedding_distance_between_str_list(
    source_list: List[str],
    target_list: List[str],
) -> List[List[float]]:
    """
    Hitung cosine similarity antara dua list string.
    Compatible dengan fungsi lama di client.py.

    Returns:
        Matrix [len(source), len(target)] berisi cosine similarity scores.
    """
    backend = _get_embedding_backend()
    src_emb = backend.create_embedding(source_list)
    tgt_emb = backend.create_embedding(target_list)

    # Cosine similarity
    result: List[List[float]] = []
    for s in src_emb:
        row: List[float] = []
        s_norm = sum(x * x for x in s) ** 0.5
        for t in tgt_emb:
            t_norm = sum(x * x for x in t) ** 0.5
            if s_norm == 0 or t_norm == 0:
                row.append(0.0)
            else:
                dot = sum(a * b for a, b in zip(s, t))
                row.append(dot / (s_norm * t_norm))
        result.append(row)
    return result
