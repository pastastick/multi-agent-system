import pickle
from pathlib import Path
from typing import Any, Optional

from coder.costeer.config import CoSTEERSettings
from coder.costeer.evolvable_subjects import EvolvingItem
from coder.costeer.evolving_agent import FilterFailedRAGEvoAgent
from coder.costeer.knowledge_management import (
    CoSTEERKnowledgeBaseV1,
    CoSTEERKnowledgeBaseV2,
    CoSTEERRAGStrategyV1,
    CoSTEERRAGStrategyV2,
)
from core.developer import Developer
from core.evaluation import Evaluator
from core.evolving_agent import EvolvingStrategy
from core.experiment import Experiment
from log import logger


class CoSTEER(Developer[Experiment]):
    def __init__(
        self,
        settings: CoSTEERSettings,
        eva: Evaluator,
        es: EvolvingStrategy,
        evolving_version: int,
        *args,
        with_knowledge: bool = True,
        with_feedback: bool = True,
        knowledge_self_gen: bool = True,
        filter_final_evo: bool = True,
        llm_backend: Optional[Any] = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.max_loop = settings.max_loop
        self.knowledge_base_path = (
            Path(settings.knowledge_base_path) if settings.knowledge_base_path is not None else None
        )
        self.new_knowledge_base_path = (
            Path(settings.new_knowledge_base_path) if settings.new_knowledge_base_path is not None else None
        )

        self.with_knowledge = with_knowledge
        self.with_feedback = with_feedback
        self.knowledge_self_gen = knowledge_self_gen
        self.filter_final_evo = filter_final_evo
        self.evolving_strategy = es
        self.evaluator = eva
        self.evolving_version = evolving_version

        # ── Latent pipeline (KV-cache) ──────────────────────────────
        # Shared llm_backend instance for KV-cache compatibility.
        # When set, evolving_strategy and evaluator use this backend
        # instead of creating LocalLLMBackend() on the fly.
        self.llm_backend = llm_backend
        # KV-cache output from the last develop() call.
        # Exposed so pipeline steps downstream can chain from it.
        self._last_kv: Optional[Any] = None

        # init knowledge base
        self.knowledge_base = self.load_or_init_knowledge_base(
            former_knowledge_base_path=self.knowledge_base_path,
            component_init_list=[],
        )
        # init rag method
        self.rag = (
            CoSTEERRAGStrategyV2(self.knowledge_base, settings=settings)
            if self.evolving_version == 2
            else CoSTEERRAGStrategyV1(self.knowledge_base, settings=settings)
        )

    def load_or_init_knowledge_base(self, former_knowledge_base_path: Path = None, component_init_list: list = []):
        if former_knowledge_base_path is not None and former_knowledge_base_path.exists():
            knowledge_base = pickle.load(open(former_knowledge_base_path, "rb"))
            if self.evolving_version == 1 and not isinstance(knowledge_base, CoSTEERKnowledgeBaseV1):
                raise ValueError("The former knowledge base is not compatible with the current version")
            elif self.evolving_version == 2 and not isinstance(
                knowledge_base,
                CoSTEERKnowledgeBaseV2,
            ):
                raise ValueError("The former knowledge base is not compatible with the current version")
        else:
            knowledge_base = (
                CoSTEERKnowledgeBaseV2(
                    init_component_list=component_init_list,
                )
                if self.evolving_version == 2
                else CoSTEERKnowledgeBaseV1()
            )
        return knowledge_base

    @property
    def last_kv(self) -> Optional[Any]:
        """KV-cache output dari develop() terakhir.

        Digunakan oleh pipeline step downstream (feedback) untuk
        menerima konteks latent dari coder evolve loop.
        """
        return self._last_kv

    def develop(self, exp: Experiment, past_kv: Optional[Any] = None) -> Experiment:

        # init intermediate items
        experiment = EvolvingItem.from_experiment(exp)

        # ── Latent pipeline: pass KV-cache ke evolving strategy ──
        # Jika llm_backend dan past_kv tersedia, strategy akan
        # menggunakan latent path (build_messages_and_run) di LLM calls.
        if self.llm_backend is not None and hasattr(self.evolving_strategy, 'set_llm_backend'):
            self.evolving_strategy.set_llm_backend(self.llm_backend)
        if past_kv is not None and hasattr(self.evolving_strategy, 'set_past_kv'):
            self.evolving_strategy.set_past_kv(past_kv)

        self.evolve_agent = FilterFailedRAGEvoAgent(
            max_loop=self.max_loop,
            evolving_strategy=self.evolving_strategy,
            rag=self.rag,
            with_knowledge=self.with_knowledge,
            with_feedback=self.with_feedback,
            knowledge_self_gen=self.knowledge_self_gen,
        )

        experiment = self.evolve_agent.multistep_evolve(
            experiment,
            self.evaluator,
            filter_final_evo=self.filter_final_evo,
        )

        # ── Latent pipeline: capture output KV dari strategy ─────
        # Setelah semua evolve iterations selesai, ambil KV terakhir.
        # KV ini merepresentasikan akumulasi konteks dari semua
        # LLM calls dalam coder evolve loop.
        if hasattr(self.evolving_strategy, 'last_kv'):
            self._last_kv = self.evolving_strategy.last_kv

        # save new knowledge base
        if self.new_knowledge_base_path is not None:
            pickle.dump(self.knowledge_base, open(self.new_knowledge_base_path, "wb"))
            logger.info(f"New knowledge base saved to {self.new_knowledge_base_path}")
        exp.sub_workspace_list = experiment.sub_workspace_list
        return exp
