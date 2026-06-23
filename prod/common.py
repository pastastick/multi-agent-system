"""
common.py — lazy-init singleton backend untuk prod/run.py.

Modul ini ada di root proyek supaya bisa di-import sebagai `common` ketika
menjalankan `python -m prod.run` dari direktori root.  Fungsi-fungsi yang sama
juga ada di try/common.py (khusus lingkungan pengujian), tetapi paket `try`
tidak bisa di-import secara normal karena `try` adalah keyword Python.

Konfigurasi model dibaca dari env var agar mudah di-override tanpa mengedit kode:
  PROD_MODEL  — nama model HuggingFace (default: Qwen/Qwen3-4B)
  PROD_DEVICE — device torch (default: cuda)
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Pastikan direktori backend/ ada di sys.path agar sub-paket backend bisa
# diimpor baik sebagai `backend.llm.client` maupun `llm.client`.
_BACKEND_DIR = Path(__file__).resolve().parent / "backend"
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

_MODEL_NAME: str = os.environ.get("PROD_MODEL", "Qwen/Qwen3-4B")
_DEVICE: str = os.environ.get("PROD_DEVICE", "cuda")
_MAX_NEW_TOKENS: int = int(os.environ.get("PROD_MAX_NEW_TOKENS", "10000"))
_TEMPERATURE: float = float(os.environ.get("PROD_TEMPERATURE", "0.7"))
_TOP_P: float = float(os.environ.get("PROD_TOP_P", "0.95"))

_BACKEND_INSTANCE = None
_LATENT_BACKEND_INSTANCE = None


def get_backend():
    """Lazy-init LocalLLMBackend (text-only, latent_steps=0).

    Load model hanya sekali per proses — model Qwen3-4B/14B makan ~8-15 GB VRAM.
    """
    global _BACKEND_INSTANCE
    if _BACKEND_INSTANCE is None:
        from backend.llm.client import LocalLLMBackend
        print(f"[common] Loading LocalLLMBackend model={_MODEL_NAME} device={_DEVICE} ...")
        _BACKEND_INSTANCE = LocalLLMBackend(
            model_name=_MODEL_NAME,
            device=_DEVICE,
            max_new_tokens=_MAX_NEW_TOKENS,
            temperature=_TEMPERATURE,
            top_p=_TOP_P,
            log_tensors=False,
            store_kv=False,
        )
        print("[common] Backend ready.")
    return _BACKEND_INSTANCE


def get_latent_backend(latent_steps_init: int = 10):
    """Lazy-init LocalLLMBackend dengan latent support aktif (use_realign=True).

    Model weights di-share via _MODEL_CACHE dengan get_backend(); overhead
    hanya pada build LatentRealigner (~few seconds, VRAM negligible).
    latent_steps_init menentukan default engine — bisa di-override per-call.
    """
    global _LATENT_BACKEND_INSTANCE
    if _LATENT_BACKEND_INSTANCE is None:
        from backend.llm.client import LocalLLMBackend
        print(f"[common] Loading LATENT LocalLLMBackend (use_realign=True) ...")
        _LATENT_BACKEND_INSTANCE = LocalLLMBackend(
            model_name=_MODEL_NAME,
            device=_DEVICE,
            max_new_tokens=_MAX_NEW_TOKENS,
            temperature=_TEMPERATURE,
            top_p=_TOP_P,
            latent_steps=latent_steps_init,
            use_realign=True,
            log_tensors=False,
            store_kv=False,
        )
        print("[common] Latent backend ready.")
    return _LATENT_BACKEND_INSTANCE
