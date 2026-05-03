"""
QuantaAlpha factor experiment module: Scenario and Experiment classes.
Uses project QlibFBWorkspace (no ProcessInf / pandas 1.5.x issues).

Latent pipeline integration:
    QlibAlphaAgentScenario menyediakan tiga level deskripsi skenario:

    1. get_scenario_all_desc(filtered_tag=...)
       Full-text deskripsi dengan proper filtering per-tag.
       rdagent's version mengabaikan filtered_tag — kita override agar
       setiap step hanya dapat konteks yang relevan.

    2. get_compact_desc(step)
       Deskripsi ringkas per-step untuk latent mode.
       Prompt lebih pendek → lebih banyak ruang KV-cache untuk
       latent reasoning (virtual tokens).

    3. Properties (background, interface, dll.)
       Akses langsung ke section individual — self-contained,
       tidak bergantung pada rdagent property implementation.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Optional

from rdagent.scenarios.qlib.experiment.factor_experiment import (
    QlibFactorScenario,
    FactorExperiment,
    FactorTask,
    FactorFBWorkspace,
)
from rdagent.utils.agent.tpl import T

from factors.workspace import QlibFBWorkspace
from core.experiment import Task
from rdagent.scenarios.qlib.experiment.factor_experiment import (
    QlibFactorExperiment as _OrigQlibFactorExperiment,
)

# Override dari rdagent, satu-satunya bedanya: mengganti experiment_workspace dengan QlibFBWorkspace lokal (bukan Docker-based).
# Ini memastikan template konfigurasi yang dipakai sesuai project
class QlibFactorExperiment(_OrigQlibFactorExperiment):
    """Override rdagent QlibFactorExperiment with project QlibFBWorkspace (correct config template)."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        import rdagent.scenarios.qlib.experiment.factor_experiment as _fe_mod

        rdagent_template_path = Path(_fe_mod.__file__).parent / "factor_template"
        self.experiment_workspace = QlibFBWorkspace(
            template_folder_path=rdagent_template_path
        )


class QlibAlphaAgentScenario(QlibFactorScenario):
    """Scenario untuk AlphaAgent factor mining pipeline.

    Menyediakan deskripsi skenario untuk prompt LLM dengan dua mode:

    Text-only mode (default):
        get_scenario_all_desc() — full atau filtered deskripsi.
        Dipakai oleh proposal.py, feedback.py sebagai bagian prompt.
        Teks ini di-tokenize dan di-encode ke KV-cache oleh LLM backend
        saat prompt diproses melalui build_messages_and_run().

    Latent mode (latent_enabled=True):
        get_compact_desc(step) — deskripsi ringkas per pipeline step.
        Prompt lebih pendek menyisakan lebih banyak kapasitas KV-cache
        untuk latent reasoning (virtual tokens via _CoreEngine.latent_pass).
        Setiap step hanya mendapat konteks yang relevan untuk tugasnya:
          - propose: background + strategy (untuk ideasi hipotesis)
          - construct: interface + output_format (untuk syntax formula)
          - feedback: background + experiment_setting (untuk evaluasi)

    Kedua mode tetap menghasilkan TEKS — encoding ke KV terjadi di
    LLM backend level, bukan di Scenario. Scenario hanya mengontrol
    KONTEN teks yang masuk ke prompt.
    """

    def get_runtime_environment(self) -> str:
        """Override rdagent's get_runtime_environment() yang butuh conda.
        Kita pakai venv, bukan conda — jadi generate deskripsi environment sendiri.
        Output ini hanya teks informatif yang masuk ke background prompt LLM.
        """
        import sys
        import subprocess
        try:
            pkgs = subprocess.check_output(
                [sys.executable, "-m", "pip", "list", "--format=freeze"],
                text=True, timeout=10
            )
        except Exception:
            pkgs = "unavailable"
        return f"Python {sys.version}\nExecutable: {sys.executable}\nInstalled packages:\n{pkgs}"

    def __init__(self, use_local: bool = True, *args, **kwargs):
        from rdagent.core.scenario import Scenario
        from factors.qlib_utils import get_data_folder_intro as local_get_data_folder_intro

        Scenario.__init__(self)
        tpl_prefix = "scenarios.qlib.experiment.prompts"

        self._background = deepcopy(
            T(f"{tpl_prefix}:qlib_factor_background").r(
                runtime_environment=self.get_runtime_environment(),
            )
        )
        # Versi background tanpa runtime_environment, untuk propose step.
        # Propose hanya butuh domain knowledge (apa itu faktor, market context),
        # bukan Python environment info yang relevan hanya untuk coder step.
        self._domain_background = deepcopy(
            T(f"{tpl_prefix}:qlib_factor_background").r(
                runtime_environment=None,
            )
        )
        self._source_data = deepcopy(local_get_data_folder_intro(use_local=use_local))
        self._output_format = deepcopy(T(f"{tpl_prefix}:qlib_factor_output_format").r())
        self._interface = deepcopy(T(f"{tpl_prefix}:qlib_factor_interface").r())
        self._strategy = deepcopy(T(f"{tpl_prefix}:qlib_factor_strategy").r())
        self._simulator = deepcopy(T(f"{tpl_prefix}:qlib_factor_simulator").r())
        self._rich_style_description = deepcopy(T(f"{tpl_prefix}:qlib_factor_rich_style_description").r())
        self._experiment_setting = deepcopy(T(f"{tpl_prefix}:qlib_factor_experiment_setting").r())

    # ── Properties (self-contained, tidak bergantung pada rdagent) ───

    @property
    def background(self) -> str:
        return self._background

    def get_source_data_desc(self, task: Optional[Task] = None) -> str:
        return self._source_data

    @property
    def output_format(self) -> str:
        return self._output_format

    @property
    def interface(self) -> str:
        return self._interface

    @property
    def strategy(self) -> str:
        """Trading strategy context — dipakai di filtered_tag='hypothesis_and_experiment'."""
        return self._strategy

    @property
    def simulator(self) -> str:
        return self._simulator

    @property
    def rich_style_description(self) -> str:
        return self._rich_style_description

    @property
    def experiment_setting(self) -> str:
        return self._experiment_setting

    # ── Full scenario descriptions ──────────────────────────────────

    def get_scenario_all_desc(
        self,
        task: Optional[Task] = None,
        filtered_tag: Optional[str] = None,
        simple_background: Optional[bool] = None,
    ) -> str:
        """Assemble scenario description dengan proper filtering.

        Berbeda dari rdagent's QlibFactorScenario.get_scenario_all_desc()
        yang MENGABAIKAN filtered_tag, implementasi ini benar-benar
        mem-filter section berdasarkan tag.

        Ini penting untuk latent pipeline karena:
          - Prompt lebih pendek → lebih banyak kapasitas KV-cache
          - Konteks lebih fokus → latent reasoning lebih akurat
          - Setiap step hanya mendapat informasi yang relevan

        Args:
            task: Optional task context untuk source_data filtering.
            filtered_tag: Filter preset:
                None — full description (semua section).
                "hypothesis_and_experiment" — background + strategy +
                    experiment_setting (konteks untuk hipotesis).
                    Dipakai oleh propose step (AlphaAgentHypothesisGen.gen).
                "feature" — background + source_data + interface +
                    output_format (konteks untuk kode/formula).
                    Dipakai oleh coder (evolving_strategy, eva_utils).
            simple_background: Jika True, hanya return background.
        """
        if simple_background:
            return f"Background of the scenario:\n{self.background}"

        if filtered_tag == "hypothesis_and_experiment":
            # Propose step: hanya butuh domain/market context untuk ideasi hipotesis.
            # - Pakai _domain_background (tanpa runtime_environment/pip list).
            # - Hapus strategy (Python coding guide — hanya relevan untuk coder).
            # - Hapus experiment_setting (detail eksekusi — tidak dibutuhkan di ideasi).
            parts = [f"Background of the scenario:\n{self._domain_background}"]
            return "\n\n".join(parts)

        if filtered_tag == "feature":
            # Coder/feature step: butuh data + interface + output format.
            # Konteks teknis untuk code generation.
            return (
                f"Background of the scenario:\n{self.background}\n\n"
                f"The source data you can use:\n{self.get_source_data_desc(task)}\n\n"
                f"The interface you should follow to write the runnable code:\n{self.interface}\n\n"
                f"The output of your code should be in the format:\n{self.output_format}"
            )

        # Default: full description — backward-compatible format.
        # Menyertakan strategy dan experiment_setting yang sebelumnya
        # TIDAK ada di rdagent's version.
        parts = [
            f"Background of the scenario:\n{self.background}",
            f"The source data you can use:\n{self.get_source_data_desc(task)}",
            f"The interface you should follow to write the runnable code:\n{self.interface}",
            f"The output of your code should be in the format:\n{self.output_format}",
            f"The simulator user can use to test your factor:\n{self.simulator}",
        ]
        if self._strategy:
            parts.append(f"Strategy context:\n{self.strategy}")
        if self._experiment_setting:
            parts.append(f"Experiment setting:\n{self.experiment_setting}")
        return "\n\n".join(parts)

    # ── Compact descriptions untuk latent mode ──────────────────────

    def get_compact_desc(self, step: str) -> str:
        """Return deskripsi skenario ringkas per pipeline step.

        Digunakan saat latent_enabled=True. Prompt lebih pendek
        menyisakan lebih banyak kapasitas KV-cache (kv_max_tokens)
        untuk latent reasoning via virtual tokens.

        Setiap step hanya mendapat konteks yang relevan:
          - propose: APA yang di-explore (market + strategy)
          - construct: BAGAIMANA menulis formula (interface + format)
          - feedback: BAGAIMANA mengevaluasi (background + setting)

        Section markers (<scenario_*>) membantu attention mechanism
        model membedakan boundaries antar bagian skenario selama
        latent processing, sehingga representasi latent lebih terstruktur.

        Args:
            step: Nama pipeline step — "propose", "construct", atau "feedback".

        Returns:
            Deskripsi ringkas dengan section markers.
        """
        if step == "propose":
            # Propose: ideasi hipotesis — butuh pemahaman market + strategi.
            # Propose hanya butuh domain context (apa itu faktor, market background).
            # - Runtime environment (pip list) hanya relevan untuk coder → pakai _domain_background.
            # - qlib_factor_strategy (Python coding guide) hanya relevan untuk coder → dihapus.
            sections = [
                f"<scenario_background>\n{self._domain_background}\n</scenario_background>"
            ]
            return "\n".join(sections)

        if step == "construct":
            # Construct hanya menulis formula JSON — bukan kode Python.
            # - Background domain sudah di KV dari propose.
            # - Variabel data, function lib, dan output format JSON sudah
            #   di user_prompt hypothesis2experiment + _COMPACT_OUTPUT_FORMAT.
            # - interface (Python code structure) dan output_format (pandas DataFrame)
            #   adalah untuk coder, bukan untuk construct — jangan kirim ke sini.
            return "<scenario>You are generating quantitative factor expressions as JSON. Background and allowed operations are already in context.</scenario>"

        if step == "feedback":
            # Feedback: evaluasi hasil backtest — butuh konteks + setting.
            sections = [
                f"<scenario_background>\n{self.background}\n</scenario_background>"
            ]
            if self._experiment_setting:
                sections.append(
                    f"<scenario_experiment>\n{self.experiment_setting}\n</scenario_experiment>"
                )
            return "\n".join(sections)

        # Fallback: full compact (semua section dengan markers).
        return self.get_scenario_all_desc(simple_background=True)
