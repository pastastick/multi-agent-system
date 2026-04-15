from typing import Any, Optional

from coder.costeer import CoSTEER
from coder.costeer.evaluators import CoSTEERMultiEvaluator
from factors.coder.config import FACTOR_COSTEER_SETTINGS
from factors.coder.evaluators import FactorEvaluatorForCoder
from factors.coder.evolving_strategy import (
    FactorMultiProcessEvolvingStrategy, FactorParsingStrategy, FactorRunningStrategy
)
from core.scenario import Scenario

"""
yang membedakan 3 class ini hanya strategi implementasi(proses bagaimana code dihasilkan)
"""


class FactorCoSTEER(CoSTEER):
    def __init__(
        self,
        scen: Scenario,
        *args,
        **kwargs,
    ) -> None:
        setting = FACTOR_COSTEER_SETTINGS       # load konfiguration

        #* Evaluator: jalankan kode, cek output, AST regularization, LLM feedback
        eva = CoSTEERMultiEvaluator(FactorEvaluatorForCoder(scen=scen), scen=scen)

        #* Strategy: LLM generate kode DARI NOL, bisa evolve (retry dengan feedback)
        es = FactorMultiProcessEvolvingStrategy(scen=scen, settings=FACTOR_COSTEER_SETTINGS)

        #* Serahkan semua ke CoSTEER (rdagent) untuk loop evolve
        # CoSTEER parent menjalankan loop: implement -> evaluate -> learn -> repeat
        super().__init__(*args, settings=setting, eva=eva, es=es, evolving_version=2, scen=scen, **kwargs)



class FactorParser(CoSTEER):
    """
    Main coder untuk AlphaAgent pipeline.

    Run pertama: render template dari ekspresi (tanpa LLM).
    Jika gagal: panggil LLM untuk perbaiki ekspresi.

    Saat llm_backend tersedia (latent pipeline aktif):
      - Strategy dan evaluator menggunakan shared llm_backend
      - KV-cache dari construct step di-chain ke LLM calls
      - Evolve loop berjalan sequential (n=1) karena GPU tensor
        tidak bisa cross process boundaries
      - KV akumulasi dalam evolve loop, exposed via last_kv
    """
    def __init__(
        self,
        scen: Scenario,
        *args,
        llm_backend: Optional[Any] = None,
        latent_steps: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs,
    ) -> None:
        setting = FACTOR_COSTEER_SETTINGS

        # Thread llm_backend ke evaluator agar LLM calls di
        # FactorCodeEvaluator dan FactorFinalDecisionEvaluator
        # menggunakan shared backend + bisa terima KV-cache.
        eva = CoSTEERMultiEvaluator(
            FactorEvaluatorForCoder(scen=scen, llm_backend=llm_backend),
            scen=scen,
            llm_backend=llm_backend,
        )

        #* run pertama dengan TEMPLATE(tanpa LLM); kalau gagal baru panggil LLM untuk perbaiki espression
        es = FactorParsingStrategy(
            scen=scen, settings=FACTOR_COSTEER_SETTINGS,
            llm_backend=llm_backend,
            latent_steps=latent_steps,
            temperature=temperature,
        )

        super().__init__(
            *args, settings=setting, eva=eva, es=es,
            evolving_version=2, scen=scen,
            llm_backend=llm_backend, **kwargs,
        )


class FactorCoder(CoSTEER):
    def __init__(
        self,
        scen: Scenario,
        *args,
        **kwargs,
    ) -> None:
        setting = FACTOR_COSTEER_SETTINGS
        eva = CoSTEERMultiEvaluator(FactorEvaluatorForCoder(scen=scen), scen=scen)

        #* langsung render template, TIDAK pernah panggil LLM
        es = FactorRunningStrategy(scen=scen, settings=FACTOR_COSTEER_SETTINGS)

        super().__init__(*args, settings=setting, eva=eva, es=es, evolving_version=2, scen=scen, **kwargs)
