"""
QuantaAlpha pipeline settings.

Defines class-path configuration for pipeline components.
Components are loaded dynamically via string class paths for flexibility.

Latent pipeline (Latent-MAS adaptation):
    Pada Latent-MAS original, agent non-judger berkomunikasi via KV-cache
    ONLY (pure latent), dan hanya judger yang generate text.

    Di QuantaAlpha, semua step menggunakan mode ``kv_and_text`` karena
    pipeline downstream butuh text output:
      - propose  → hypothesis text (untuk trace + logging)
      - construct → formula expression (HARUS text, di-parse oleh QlibFactorParser)
      - feedback  → evaluation text (untuk trace history)

    KV-cache adalah channel komunikasi latent TAMBAHAN di samping text.
    Model "berpikir diam" via virtual tokens (latent steps) sebelum
    menghasilkan output text, sehingga konteks lebih kaya meskipun
    token output lebih sedikit.

    Latent realignment (``use_realign=True``) memproyeksikan hidden state
    sebelum inject sebagai virtual token — membantu model "fokus" pada
    representasi yang relevan saat latent reasoning.

    Env var prefix: ``QLIB_FACTOR_`` (contoh: ``QLIB_FACTOR_LATENT_ENABLED=true``)
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from core.conf import ExtendedBaseSettings, ExtendedSettingsConfigDict

# Absolute path ke backend/ sehingga output_log_dir tidak bergantung CWD.
# settings.py ada di backend/pipeline/ → parent.parent = backend/
_BACKEND_DIR = Path(__file__).resolve().parent.parent
_DEFAULT_OUTPUT_LOG_DIR = str(_BACKEND_DIR / "debug" / "llm_outputs")


# =============================================================================
# Base setting classes
# =============================================================================

class BasePropSetting(ExtendedBaseSettings):
    """Common base for RD Loop configuration."""

    scen: str = ""                  # class path untuk scenario(deskripsi prompt)
    knowledge_base: str = ""
    knowledge_base_path: str = ""
    hypothesis_gen: str = ""        # Generate hipotesis trading via LLM
    hypothesis2experiment: str = ""     # Convert hipotesis → factor expression via LLM + validation loop
    coder: str = ""                 # Parse expression → kode Python, lalu evolve jika gagal
    runner: str = ""                # Jalankan kode, hasilkan DataFrame factor value
    summarizer: str = ""            # Buat feedback dari hasil backtest
    evolving_n: int = 10


class BaseFacSetting(ExtendedBaseSettings):
    """Common base for Alpha Agent Loop configuration."""

    scen: str = ""
    knowledge_base: str = ""
    knowledge_base_path: str = ""
    hypothesis_gen: str = ""
    construction: str = ""
    calculation: str = ""
    coder: str = ""
    runner: str = ""
    summarizer: str = ""
    evolving_n: int = 10


#* Factor mining settings (main experiment)
class AlphaAgentFactorBasePropSetting(BasePropSetting):
    """
    Main experiment: LLM-driven factor mining.

    Saklar pipeline: `latent_enabled`
      - False : jalur QuantaAlpha lama (build_messages_and_create_chat_completion,
                tanpa KV chaining). Tidak dipakai untuk studi banding medium.
      - True  : jalur FrontEndPipeline (latent_mas). Medium komunikasi antar-agen
                ditentukan `comm_mode` (variabel eksperimen utama KV-vs-TEXT).

    `comm_mode` (hanya berlaku saat latent_enabled=True) — tiga mode terkontrol
    yang memakai PROMPT & AGEN yang SAMA, beda hanya medium komunikasinya:
      1. "text"        : semua agen generate teks; handoff antar-agen via TEKS
                         (tanpa KV). = baseline TEKS terkontrol.
      2. "kv_and_text" : semua agen generate teks, TETAPI handoff via KV-cache.
      3. "kv" (default): hanya construct & feedback yang generate teks; proposal/
                         design/guidance laten murni (kv_only). Handoff via KV-cache.
    """
    model_config = ExtendedSettingsConfigDict(env_prefix="QLIB_FACTOR_", protected_namespaces=())

    # ── Component class paths ────────────────────────────────────────────

    # TODO(naming): 'coder'/'runner' = istilah rdagent lama. Di jalur latent:
    #   coder=QlibFactorParser (parse ekspresi→kode), runner=QlibFactorRunner
    #   (eksekusi + RankIC). Pertimbangkan rename agar selaras arsitektur latent_mas.
    scen: str = "factors.experiment.QlibAlphaAgentScenario"
    hypothesis_gen: str = "factors.proposal.AlphaAgentHypothesisGen"
    hypothesis2experiment: str = "factors.proposal.AlphaAgentHypothesis2FactorExpression"
    coder: str = "factors.qlib_coder.QlibFactorParser"
    runner: str = "factors.runner.QlibFactorRunner"
    summarizer: str = "factors.feedback.AlphaAgentQlibFactorHypothesisExperiment2Feedback"
    evolving_n: int = 5

    # ── Latent pipeline config ───────────────────────────────────────────
    # Master switch: aktifkan KV-cache chaining + latent reasoning.
    # Saat True, factor_mining.py akan auto-create LocalLLMBackend
    # dan pipeline menggunakan LatentHypothesisGen/Experiment/Feedback.
    latent_enabled: bool = True

    # Medium komunikasi antar-agen front-end (hanya saat latent_enabled=True).
    # Salah satu: "text" | "kv_and_text" | "kv". Lihat docstring kelas.
    # Diteruskan ke FrontEndPipeline(comm_mode=...) di loop.py.
    comm_mode: str = "kv"

    # Model HuggingFace untuk LocalLLMBackend.
    # Di-load sekali, di-share ke semua step dalam satu loop.
    latent_model_name: str = "Qwen/Qwen3-8B"
    latent_device: str = "cuda"

    # ── Latent reasoning steps ───────────────────────────────────────────
    # Berapa kali model melakukan "thinking diam" (inject virtual token
    # ke KV-cache tanpa generate text) per LLM call.
    # Lebih tinggi = reasoning lebih dalam, tapi lebih lambat.
    # Referensi: LatentMASMethod.latent_steps di core/latent/latent_method.py
    # B1 (dari G2, Qwen3-8B): pada ls=60 mode `kv` menghasilkan 0 ekspresi dalam
    # 6 run, sementara ls=5/10 menghasilkan 6/6 — tanpa penurunan |IC|. Nilai 60
    # yang lama adalah penyebab langsung kolaps produksi, bukan setelan "reasoning
    # lebih dalam". Lihat lab/HASIL_GPU.md §2.
    latent_steps: int = 10                      # default global untuk _CoreEngine

    # Per-step override (None = pakai latent_steps global).
    # Dinaikkan 10 → 40: dengan reasoning-via-latent sebagai jalur utama (output
    # kv_only TIDAK di-decode), kedalaman "thinking diam" jadi knob utama kualitas
    # niat yang masuk ke KV. Tradeoff: lebih lambat per call.
    latent_steps_propose: Optional[int] = None
    latent_steps_construct: Optional[int] = None
    latent_steps_coder: Optional[int] = None
    latent_steps_feedback: Optional[int] = None

    # ── Mode langkah laten (B2, dari G3) ─────────────────────────────────
    # "raw" = perilaku lama (realign(h) dinormalkan): vektornya di LUAR manifold
    # embedding dan rollout-nya deterministik → entropi jalur laten nol.
    # "gumbel" memakai kombinasi konveks embedding NYATA + noise Gumbel, jadi
    # vektor laten selalu in-distribution dan T menjadi knob entropi.
    # Terukur G3 (Qwen3-8B, ls=10, 18 run): lolos gate 54% → 91%, faktor hidup
    # 13 → 27, klaster sinyal 6 → 9. PERINGATAN: kekuatan sinyal TIDAK membaik
    # (t = −0,27) — ini perbaikan KEANDALAN PRODUKSI, bukan perbaikan mutu.
    latent_step_mode: str = "gumbel"
    latent_step_temp: float = 0.7

    # ── Latent realignment ───────────────────────────────────────────────
    # Proyeksi hidden state sebelum inject sebagai virtual token.
    # LatentRealigner di _CoreEngine: identity (False) atau learned (True).
    # True membantu model "fokus" latent reasoning pada domain quantitative.
    use_realign: bool = True

    # Qwen3 chain-of-thought dalam <think>...</think> tags.
    # False = hemat token, cocok untuk pipeline yang parse JSON output.
    # True  = model menulis reasoning eksplisit sebelum jawaban.
    enable_thinking: bool = False

    # ── KV-cache management ──────────────────────────────────────────────
    # Max KV tokens yang di-carry antar iterasi loop.
    # Setelah feedback, KV di-truncate ke jumlah ini sebelum
    # dikirim ke propose step di iterasi berikutnya.
    # Terlalu kecil: hilang konteks.  Terlalu besar: lambat + OOM.
    kv_max_tokens: int = 20480

    # Simpan KV-cache ke disk (untuk resume/debugging).
    # Pakai KVCacheStore di llm/client.py.
    store_kv: bool = False

    # ── KNN-based KV-cache filtering ────────────────────────────────────
    # Diadaptasi dari LatentMAS paper (core/latent/latent_mas_knn.py).
    #
    # Alih-alih memotong N token terakhir secara buta (kv_truncate),
    # KNN filter menghitung cosine similarity antara input prompt saat
    # ini dan key vectors di middle layer KV-cache, lalu mempertahankan
    # hanya token yang paling relevan.
    #
    # Komplementer dengan kv_max_tokens:
    #   1. kv_truncate(kv_max_tokens): hard cap ukuran (cegah OOM)
    #   2. kv_knn_filter: seleksi kualitas (pertahankan yang relevan)
    #
    # Diterapkan transparan di _CoreEngine sebelum setiap forward pass
    # yang menerima past_kv dari step sebelumnya.
    knn_enabled: bool = True
    knn_percentage: float = 0.8       # fraksi token yang dipertahankan (0.0-1.0)
    knn_min_keep: int = 5             # minimum token terbaru selalu dipertahankan
    knn_strategy: str = "top"         # "top" (paling mirip), "bottom", "random"

    # ── Generation parameters ────────────────────────────────────────────
    max_new_tokens: int = 40960
    temperature: float = 0.8
    top_p: float = 0.95

    # Per-step temperature override.
    # Construct (formula/expression): temperature rendah → output lebih presisi.
    # Ini PENTING karena output construct di-parse oleh QlibFactorParser
    # menjadi kode Python — output yang terlalu "kreatif" akan gagal parse.
    # Coder (expression fix): temperature rendah seperti construct, karena
    # output coder juga berupa ekspresi yang harus bisa di-parse.
    # Propose: bisa lebih tinggi untuk eksplorasi hipotesis yang beragam.
    temperature_propose: Optional[float] = None     # None = pakai temperature global
    temperature_construct: float = 0.7              # rendah: formula presisi

    # TODO(naming): samakan pola nama per-step (propose/construct/coder/feedback).
    temperature_coder: float = 0.4                  # rendah: expression fix presisi
    temperature_feedback: Optional[float] = None    # None = pakai temperature global

    # Debug: simpan tensor (input_ids, output_ids, hidden_last) ke disk.
    # DIPAKAI: bila True, LocalLLMBackend mengaktifkan TensorConvManager (dump
    # tensor per-call ke folder run). Lihat client.py:1182 & factor_mining.py:87.
    log_tensors: bool = True

    # Debug: simpan output text LLM per-call ke JSONL.
    # Ditulis SEGERA setelah generation (flush + fsync) — sehingga
    # crash tengah iterasi pun tetap meninggalkan jejak lengkap.
    output_log_dir: str = _DEFAULT_OUTPUT_LOG_DIR

    # ── Factory methods ──────────────────────────────────────────────────
    # TODO(naming): label `step` pakai {propose, construct, coder, feedback}.
    def get_latent_steps_for(self, step: str) -> Optional[int]:
        """Return per-step latent_steps override, atau None jika pakai default engine."""
        mapping = {
            "propose": self.latent_steps_propose,
            "construct": self.latent_steps_construct,
            "coder": self.latent_steps_coder,
            "feedback": self.latent_steps_feedback,
        }
        return mapping.get(step)

    # TODO(naming): label `step` pakai {propose, construct, coder, feedback}.
    def get_temperature_for(self, step: str) -> Optional[float]:
        """Return per-step temperature override, atau None jika pakai default backend."""
        mapping = {
            "propose": self.temperature_propose,
            "construct": self.temperature_construct,
            "coder": self.temperature_coder,
            "feedback": self.temperature_feedback,
        }
        return mapping.get(step)

    def create_llm_backend(self):
        """
        Factory: buat LocalLLMBackend dari settings ini.

        Dipanggil oleh factor_mining.py saat latent_enabled=True
        dan belum ada llm_backend yang di-pass manual.

        Model di-load sekali di sini, lalu di-share ke semua
        step dalam pipeline (propose, construct, feedback).
        Ini WAJIB karena KV-cache hanya kompatibel dengan model
        yang sama — KV dari model A tidak bisa dipakai oleh model B.
        """
        from llm.client import LocalLLMBackend
        # OUTPUT TERPADU: anchor SEMUA sink debug backend (llm_outputs, conv_logs,
        # kv_store) di bawah folder run yang sama (parent dari output_log_dir =
        # backend/runs/<id>/). Jadi tak ada lagi ./debug/ liar relatif-cwd.
        _run_dir = Path(self.output_log_dir).parent
        return LocalLLMBackend(
            model_name=self.latent_model_name,
            device=self.latent_device,
            latent_steps=self.latent_steps,
            use_realign=self.use_realign,
            enable_thinking=self.enable_thinking,
            log_tensors=self.log_tensors,
            store_kv=self.store_kv,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            knn_enabled=self.knn_enabled,
            knn_percentage=self.knn_percentage,
            knn_min_keep=self.knn_min_keep,
            knn_strategy=self.knn_strategy,
            latent_step_mode=self.latent_step_mode,
            latent_step_temp=self.latent_step_temp,
            # B9: `kv_max_tokens` selama ini kode mati — nilainya tak pernah
            # sampai ke backend, dan `knn_enabled` otomatis mati saat
            # latent_steps>0, sehingga TIDAK ADA kendali ukuran KV sama sekali
            # (HASIL_GPU §8.1). Prasyaratnya (B8, pembukuan RoPE di kv_truncate)
            # sudah dipasang di llm/_shared.py.
            kv_max_seq_len=self.kv_max_tokens,
            output_log_dir=self.output_log_dir,
            conv_dir=str(_run_dir / "conv_logs"),
            kv_dir=str(_run_dir / "kv_store"),
        )


class FactorBasePropSetting(BasePropSetting):
    """Basic factor experiment (traditional RD Loop mode)."""
    model_config = ExtendedSettingsConfigDict(env_prefix="QLIB_FACTOR_", protected_namespaces=())

    scen: str = "factors.experiment.QlibFactorScenario"
    hypothesis_gen: str = "factors.proposal.QlibFactorHypothesisGen"
    hypothesis2experiment: str = "factors.proposal.QlibFactorHypothesis2Experiment"
    coder: str = "factors.qlib_coder.QlibFactorCoSTEER"
    runner: str = "factors.runner.QlibFactorRunner"
    summarizer: str = "factors.feedback.QlibFactorHypothesisExperiment2Feedback"
    evolving_n: int = 10


class FactorBackTestBasePropSetting(BasePropSetting):
    """Factor backtest mode."""
    model_config = ExtendedSettingsConfigDict(env_prefix="QLIB_FACTOR_", protected_namespaces=())

    scen: str = "factors.experiment.QlibAlphaAgentScenario"
    hypothesis_gen: str = "factors.proposal.EmptyHypothesisGen"
    hypothesis2experiment: str = "factors.proposal.BacktestHypothesis2FactorExpression"
    coder: str = "factors.qlib_coder.QlibFactorCoder"
    runner: str = "factors.runner.QlibFactorRunner"
    summarizer: str = "factors.feedback.QlibFactorHypothesisExperiment2Feedback"
    evolving_n: int = 1


class FactorFromReportPropSetting(FactorBasePropSetting):
    """Factor extraction from research reports."""
    scen: str = "factors.experiment.QlibFactorFromReportScenario"
    report_result_json_file_path: str = "git_ignore_folder/report_list.json"
    max_factors_per_exp: int = 1000
    is_report_limit_enabled: bool = False


#* Model experiment settings (contrib, optional)
class ModelBasePropSetting(BasePropSetting):
    """Model experiment (extended feature)."""
    model_config = ExtendedSettingsConfigDict(env_prefix="QLIB_MODEL_", protected_namespaces=())

    scen: str = "contrib.model.experiment.QlibModelScenario"
    hypothesis_gen: str = "contrib.model.proposal.QlibModelHypothesisGen"
    hypothesis2experiment: str = "contrib.model.proposal.QlibModelHypothesis2Experiment"
    coder: str = "contrib.model.qlib_coder.QlibModelCoSTEER"
    runner: str = "contrib.model.runner.QlibModelRunner"
    summarizer: str = "factors.feedback.QlibModelHypothesisExperiment2Feedback"
    evolving_n: int = 10


# =============================================================================
# Singleton instances (global)
# =============================================================================

ALPHA_AGENT_FACTOR_PROP_SETTING = AlphaAgentFactorBasePropSetting()
FACTOR_PROP_SETTING = FactorBasePropSetting()
FACTOR_BACK_TEST_PROP_SETTING = FactorBackTestBasePropSetting()
FACTOR_FROM_REPORT_PROP_SETTING = FactorFromReportPropSetting()
MODEL_PROP_SETTING = ModelBasePropSetting()
