#!/usr/bin/env bash
# /workspace/runpod_env.sh — sumber file ini di awal setiap session
# Di-copy ke /workspace/ oleh setup_runpod.sh

# uv & cargo binary location (uv installer default ke ~/.local/bin → ephemeral)
export XDG_DATA_HOME=/workspace/.local/share
export XDG_CONFIG_HOME=/workspace/.config
export XDG_CACHE_HOME=/workspace/.cache
export PATH=/workspace/.local/bin:$PATH

# uv cache & virtualenv
export UV_CACHE_DIR=/workspace/.cache/uv
export UV_PYTHON_INSTALL_DIR=/workspace/.local/share/uv/python
export UV_TOOL_DIR=/workspace/.local/share/uv/tools

# pip cache (untuk fallback jika tidak pakai uv)
export PIP_CACHE_DIR=/workspace/.cache/pip

# HuggingFace model & dataset cache (default ~/.cache/huggingface → ephemeral)
export HF_HOME=/workspace/.cache/huggingface
export HUGGINGFACE_HUB_CACHE=/workspace/.cache/huggingface/hub
export TRANSFORMERS_CACHE=/workspace/.cache/huggingface/hub

# HuggingFace token — untuk download model/dataset private (Qwen3, qlib_csi300)
export HF_TOKEN="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"

# Torch hub & inductor cache
export TORCH_HOME=/workspace/.cache/torch
export TORCHINDUCTOR_CACHE_DIR=/workspace/.cache/torchinductor

# Izinkan transformers download model dari HF saat run pertama.
# Default kode adalah local_files_only=True (offline) — set 0 agar model
# Qwen3 ter-download otomatis jika belum ada di cache HF.
export HF_LOCAL_ONLY=0

# Project-specific
export PYTHONPATH=/workspace/project/multi-agent-system/backend
