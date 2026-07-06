"""
Evolution controller for managing the original→mutation→crossover cycle.

The controller orchestrates the evolutionary process:
1. Original round: Initial exploration in each direction
2. Mutation round: Orthogonal exploration from each original trajectory
3. Crossover round: Combine trajectories across directions
4. Repeat mutation→crossover cycle
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional
import threading

from log import logger
from llm.client import LocalLLMBackend
from .trajectory import StrategyTrajectory, TrajectoryPool, RoundPhase
from .mutation import MutationOperator
from .crossover import CrossoverOperator
from .negative_memory import NegativeMemory

# [terjawab — skripsi Bab 4 §Evolusi dan Aliran KV + §Seleksi Trajectory]: konfigurasi
#   & metrik evolusi (ambang sukses, seleksi induk s(τ), ganti-SOTA) terurai di Bab 4.
@dataclass
class EvolutionConfig:
    """Configuration for evolution process."""
    # Number of planning directions (parallel original rounds)
    num_directions: int = 2

    # Steps per loop (5 for: propose/construct/calculate/backtest/feedback)
    steps_per_loop: int = 5

    # Maximum total rounds (original + mutation + crossover rounds)
    max_rounds: int = 10

    # Enable/disable mutation phase; when false, skip mutation rounds entirely
    mutation_enabled: bool = True

    # Enable/disable crossover phase; when false, skip crossover rounds entirely
    crossover_enabled: bool = True

    # Crossover parameters
    crossover_size: int = 2  # Number of parents per crossover
    crossover_n: int = 3     # Number of crossover combinations per round

    # Whether to prefer diverse crossover combinations
    prefer_diverse_crossover: bool = True

    # Parent selection for crossover: best | random | weighted | weighted_inverse | top_percent_plus_random
    parent_selection_strategy: str = "best"

    # Top percent threshold when parent_selection_strategy = "top_percent_plus_random"
    top_percent_threshold: float = 0.3

    # Enable parallel execution within each round
    parallel_enabled: bool = False

    # Path to save trajectory pool
    pool_save_path: Optional[str] = None

    # Path to evolution prompts
    mutation_prompt_path: Optional[str] = None
    crossover_prompt_path: Optional[str] = None

    # Start with empty trajectory pool (ignore existing data)
    fresh_start: bool = True

    # HYBRID part-2: penalti diversitas family-operator pada SKOR SELEKSI.
    # effective_metric = primary_metric − diversity_lambda · family_penalty.
    # 0.0 = OFF (tanpa perubahan perilaku). ~0.01-0.03 mendemosikan faktor yang
    # family operatornya redundan terhadap populasi → seleksi parent/best lebih
    # beragam, tanpa hard-reject. Skala: metric (RankIC) ~0.01-0.05, penalty ∈[0,1].
    diversity_lambda: float = 0.0

    # Gate is_successful() — ambang model-free per-factor (default LONGGAR = >0).
    # success_ic_threshold:   FactorIC_mean harus > nilai ini (kekuatan sinyal).
    # success_icir_threshold: FactorICIR_mean harus > nilai ini (stabilitas sinyal).
    # Default 0.0 (IC>0 AND ICIR>0) untuk bootstrap pool tanpa kelaparan; perketat
    # via experiment.yaml setelah data GPU nyata (mis. 0.01 / 0.3 ala QuantaAlpha).
    success_ic_threshold: float = 0.0
    success_icir_threshold: float = 0.0

    # Correlation gate: faktor baru di-drop jika |Pearson corr| terhadap faktor lain
    # dalam ronde yang sama (within-round) ATAU terhadap faktor di persistent store
    # (cross-round) melebihi threshold ini. 0.7 default; 0.0 = nonaktif.
    # Ekspos via env FACTOR_CoSTEER_CORR_GATE_THRESHOLD atau experiment.yaml.
    corr_gate_threshold: float = 0.7

    # Negative-hypothesis memory: rekam MEKANISME yang gagal lintas-generasi
    # (evaluasi menyeluruh — hypothesis + ekspresi + metrik IC/ICIR) lalu suntik
    # daftar "AVOID" ke prompt proposal agar tak mengusulkan ulang mekanisme gagal.
    # Berbeda dari diversity_lambda (altitude OPERATOR) — ini altitude MEKANISME.
    # negative_memory_max_items: jumlah maksimum entri yang dirender ke hint.
    negative_memory_enabled: bool = True
    negative_memory_max_items: int = 8


class EvolutionController:
    """
    Controls the evolutionary exploration process.

    The evolution cycle:
    1. Original rounds: Run initial exploration for each planning direction
    2. Mutation rounds: Generate orthogonal strategies from each original
    3. Crossover rounds: Combine top trajectories across all directions
    4. Repeat: mutation → crossover → mutation → crossover → ...

    The controller:
    - Tracks all trajectories in a pool
    - Determines which phase/round to run next
    - Generates strategy guidance for each round
    - Manages parent selection for mutation/crossover

    After crossover, the number of parallel branches changes:
    - Initial: num_directions branches
    - After first crossover: crossover_n branches (and so on)
    """

    def __init__(
        self,
        config: EvolutionConfig,
        llm_backend: Optional[LocalLLMBackend] = None,
    ):
        """
        Initialize evolution controller.

        Args:
            config: Evolution configuration
            llm_backend: Shared LocalLLMBackend instance. Jika disediakan,
                        mutation/crossover operators menggunakan latent path
                        (KV-cache dari parent trajectory sebagai past_key_values).
        """
        self.config = config
        self._llm_backend = llm_backend
        # Mode laten = backend di-share (loop memakai sinyal yang sama: llm_backend
        # ada → factor_propose lewat run_evolution judger-only yang membaca TEKS
        # parent dan MENGABAIKAN strategy_suffix/parent_kv dari task). Maka operator
        # teks-suffix LAMA (mutation_op/crossover_op.generate_*) — masing-masing 1
        # generasi LLM kv_and_text — tak perlu dipanggil karena hasilnya dibuang.
        # Gate-nya terpusat di _mutation_suffix/_crossover_suffix. select_crossover_pairs
        # (pure-Python, tanpa LLM) TETAP dipakai untuk memilih pasangan parent.
        self.latent_mode = llm_backend is not None

        # Initialize trajectory pool with fresh_start option
        pool_path = Path(config.pool_save_path) if config.pool_save_path else None
        self.pool = TrajectoryPool(save_path=pool_path, fresh_start=config.fresh_start)

        # Negative-hypothesis memory — mekanisme gagal lintas-generasi → hint AVOID
        # untuk proposal. Persist di samping pool (negative_memory.json).
        neg_path = (pool_path.parent / "negative_memory.json") if pool_path else None
        self.negative_memory = NegativeMemory(
            save_path=neg_path,
            max_items=getattr(config, "negative_memory_max_items", 8),
            enabled=getattr(config, "negative_memory_enabled", True),
            fresh_start=config.fresh_start,
        )

        # Initialize operators dengan shared llm_backend untuk latent path.
        # Operators akan gunakan parent.kv_cache sebagai past_key_values
        # saat generate mutation/crossover guidance.
        mutation_path = Path(config.mutation_prompt_path) if config.mutation_prompt_path else None
        crossover_path = Path(config.crossover_prompt_path) if config.crossover_prompt_path else None
        self.mutation_op = MutationOperator(prompt_path=mutation_path, llm_backend=llm_backend)
        self.crossover_op = CrossoverOperator(prompt_path=crossover_path, llm_backend=llm_backend)

        # State tracking
        self._current_round = 0
        self._current_phase = RoundPhase.ORIGINAL #* mulai dari original
        self._directions_completed = set()  # Track which directions completed original
        self._crossover_groups: list[list[StrategyTrajectory]] = []  # Current crossover groups
        self._crossover_idx = 0  # Which crossover group is next

        # Track active branch count (changes after crossover)
        self._active_branch_count = config.num_directions
        # Track trajectories to mutate in current mutation round
        self._mutation_targets: list[StrategyTrajectory] = [] #* target untuk dimutasi
        self._mutation_idx = 0  # Current index in mutation targets

    def set_llm_backend(self, backend: Optional[LocalLLMBackend]) -> None:
        """Propagate llm_backend ke mutation/crossover operators."""
        self._llm_backend = backend
        self.latent_mode = backend is not None  # jaga sinkron dgn __init__
        self.mutation_op.set_llm_backend(backend)
        self.crossover_op.set_llm_backend(backend)
    # DIPANGGIL: factor_mining.py:752 (`controller.get_current_state()`) untuk
    # logging progres ronde/fase di tiap iterasi evolution loop.
    def get_current_state(self) -> dict[str, Any]:
        """Get current evolution state."""
        return {
            "round": self._current_round,
            "phase": self._current_phase.value,
            "directions_completed": list(self._directions_completed),
            "active_branch_count": self._active_branch_count,
            "mutation_targets_remaining": len(self._mutation_targets) - self._mutation_idx if self._mutation_targets else 0,
            "crossover_groups_remaining": len(self._crossover_groups) - self._crossover_idx,
            "pool_stats": self.pool.get_statistics(),
        }
    # KOREKSI: negative memory SUDAH persisten + aktif. Alurnya: trajectory gagal →
    # negative_memory.record() (controller:859) → _negative_hint() → render_hint →
    # disuntik ke task['negative_hint'] di get_next_task() → diteruskan ke
    # FrontEndPipeline.run(negative_hint=...) → masuk prompt proposal ronde berikut.
    # Persistensi: negative_memory.json (NegativeMemory._save). Jadi BUKAN sekadar
    # disimpan; ia mengubah prompt proposal. Yang terbuka tinggal efektivitas empiris.
    def _negative_hint(self) -> str:
        """Hint AVOID dari negative memory untuk disuntik ke task → proposal."""
        try:
            return self.negative_memory.render_hint()
        except Exception:  # noqa: BLE001 — hint opsional, jangan ganggu loop
            return ""

    def get_next_task(self) -> Optional[dict[str, Any]]: #* Sequential mode
        """Wrapper: dispatch task berikutnya lalu suntik negative_hint (memori
        kegagalan mekanisme lintas-generasi) ke task sebelum dikembalikan."""
        task = self._dispatch_next_task()
        # Guard overshoot max_rounds: metode transisi-fase (_get_mutation_task /
        # _get_crossover_task) menaikkan _current_round LALU langsung mengembalikan
        # task ronde baru dalam SATU call — melewati guard `>= max_rounds` yang hanya
        # dicek di AWAL _dispatch_next_task (baru berlaku di call berikutnya). Tanpa
        # cek ini, max_rounds=3 bocor menjalankan 1 task ronde-3 sebelum berhenti.
        # round_idx task = _current_round saat dibuat → chokepoint tunggal yang andal.
        if task is not None and task.get("round_idx", 0) >= self.config.max_rounds:
            logger.info(
                f"Evolution complete: round {task['round_idx']} reached max rounds "
                f"({self.config.max_rounds})"
            )
            return None
        if task is not None:
            task.setdefault("negative_hint", self._negative_hint())
        return task

    def _dispatch_next_task(self) -> Optional[dict[str, Any]]:
        """
        Determine the next task to run.

        Returns:
            Dictionary describing the next task:
            - "phase": RoundPhase (original/mutation/crossover)
            - "direction_id": Which direction (for original/mutation)
            - "parent_trajectories": Parent trajectories (for mutation/crossover)
            - "strategy_suffix": Prompt suffix for hypothesis generator
            - "round_idx": Current round index

            Returns None if evolution is complete.
        """
        # Check if we've reached max rounds
        if self._current_round >= self.config.max_rounds:
            logger.info(f"Evolution complete: reached max rounds ({self.config.max_rounds})")
            return None #* sudah max -> selesai

        # Phase: ORIGINAL
        if self._current_phase == RoundPhase.ORIGINAL:
            return self._get_original_task() #* cari direction yang belum selesai di original

        # Phase: MUTATION
        elif self._current_phase == RoundPhase.MUTATION:
            return self._get_mutation_task() #* ambil mutation target berikutnya

        # Phase: CROSSOVER
        elif self._current_phase == RoundPhase.CROSSOVER:
            return self._get_crossover_task() #* ambil crossover group berikutnya

        return None

    #* Parallel execution methods
    def get_all_tasks_for_current_phase(self) -> list[dict[str, Any]]:
        """
        Get all remaining tasks for the current phase.

        This is used for parallel execution - returns all tasks that can
        be executed in parallel within the current round/phase.

        Returns:
            List of task dictionaries, or empty list if phase is complete
        """
        # Check if we've reached max rounds
        if self._current_round >= self.config.max_rounds:
            return []

        tasks = []

        # Phase: ORIGINAL - collect all remaining original tasks
        if self._current_phase == RoundPhase.ORIGINAL:
            for d in range(self.config.num_directions):
                if d not in self._directions_completed:
                    tasks.append({
                        "phase": RoundPhase.ORIGINAL,
                        "direction_id": d,
                        "parent_trajectories": [],
                        "strategy_suffix": "",
                        "round_idx": self._current_round,
                    })

            # If no tasks, transition phase for next call
            if not tasks:
                self._current_round += 1
                # Transition based on enabled phases
                if self.config.mutation_enabled:
                    self._current_phase = RoundPhase.MUTATION
                elif self.config.crossover_enabled:
                    self._prepare_crossover_groups()
                    self._current_phase = RoundPhase.CROSSOVER
                else:
                    return []  # No evolution, just original
                return self.get_all_tasks_for_current_phase()

        # Phase: MUTATION - collect all remaining mutation tasks
        elif self._current_phase == RoundPhase.MUTATION:
            # Skip if mutation is disabled
            if not self.config.mutation_enabled:
                if self.config.crossover_enabled:
                    self._prepare_crossover_groups()
                    self._current_phase = RoundPhase.CROSSOVER
                    self._current_round += 1
                    return self.get_all_tasks_for_current_phase()
                return []

            # Prepare mutation targets if needed
            if not self._mutation_targets:
                self._prepare_mutation_targets()

            for idx, parent in enumerate(self._mutation_targets):
                if idx < self._mutation_idx:
                    continue  # Skip already processed

                # Check if this mutation already exists
                existing = [t for t in self.pool.get_all()
                           if t.round_idx == self._current_round
                           and t.phase == RoundPhase.MUTATION
                           and parent.trajectory_id in t.parent_ids]
                if existing:
                    continue

                suffix = self._mutation_suffix(parent)
                # Jangan pass mutation/feedback KV ke propose — akan prime model ke
                # feedback-format. factor_mining fallback ke _planning_kv (netral).
                tasks.append({
                    "phase": RoundPhase.MUTATION,
                    "direction_id": idx,
                    "parent_trajectories": [parent],
                    "strategy_suffix": suffix,
                    "round_idx": self._current_round,
                    "parent_kv": None,
                })

            # If no tasks, transition phase for next call
            if not tasks:
                self._mutation_targets = []
                self._mutation_idx = 0
                self._current_round += 1

                if self.config.crossover_enabled:
                    self._prepare_crossover_groups()
                    self._current_phase = RoundPhase.CROSSOVER
                else:
                    # Stay in mutation mode
                    self._current_phase = RoundPhase.MUTATION
                return self.get_all_tasks_for_current_phase()

        # Phase: CROSSOVER - collect all remaining crossover tasks
        elif self._current_phase == RoundPhase.CROSSOVER:
            # Skip if crossover is disabled
            if not self.config.crossover_enabled:
                if self.config.mutation_enabled:
                    self._current_phase = RoundPhase.MUTATION
                    self._current_round += 1
                    return self.get_all_tasks_for_current_phase()
                return []

            for idx in range(self._crossover_idx, len(self._crossover_groups)):
                parents = self._crossover_groups[idx]
                suffix = self._crossover_suffix(parents)
                # parent_kv = None: sama dengan mutation, parent feedback KV
                # mem-prime propose ke format feedback. Crossover guidance sudah
                # lengkap di text suffix.
                tasks.append({
                    "phase": RoundPhase.CROSSOVER,
                    "direction_id": idx,
                    "parent_trajectories": parents,
                    "strategy_suffix": suffix,
                    "round_idx": self._current_round,
                    "parent_kv": None,
                })

            # If no tasks, transition phase for next call
            if not tasks:
                self._current_round += 1

                if self.config.mutation_enabled:
                    self._current_phase = RoundPhase.MUTATION
                else:
                    # Stay in crossover mode, prepare new groups
                    self._prepare_crossover_groups()
                    self._current_phase = RoundPhase.CROSSOVER
                return self.get_all_tasks_for_current_phase()

        # Suntik negative_hint (memori kegagalan mekanisme lintas-generasi) ke
        # tiap task sebelum dieksekusi paralel. setdefault: aman bila rekursif.
        nh = self._negative_hint()
        for t in tasks:
            t.setdefault("negative_hint", nh)
        return tasks

    #* Parallel execution helper methods
    def advance_phase_after_parallel_completion(self, completed_tasks: list[dict[str, Any]]):
        """
        Update controller state after parallel tasks complete.

        Called after all parallel tasks in a phase complete to
        advance the controller to the next phase.

        Args:
            completed_tasks: List of completed task dictionaries
        """
        if not completed_tasks:
            return

        phase = completed_tasks[0]["phase"]

        if phase == RoundPhase.ORIGINAL:
            # Mark all directions as completed
            for task in completed_tasks:
                self._directions_completed.add(task["direction_id"])

            # Transition based on enabled phases
            if len(self._directions_completed) >= self.config.num_directions:
                self._current_round += 1
                if self.config.mutation_enabled:
                    self._current_phase = RoundPhase.MUTATION
                    logger.info(f"All original rounds complete, transitioning to mutation (round {self._current_round})")
                elif self.config.crossover_enabled:
                    self._prepare_crossover_groups()
                    self._current_phase = RoundPhase.CROSSOVER
                    logger.info(f"All original rounds complete, transitioning to crossover (round {self._current_round})")
                else:
                    logger.info("Neither mutation nor crossover enabled, evolution complete")

        elif phase == RoundPhase.MUTATION:
            # Update mutation index to skip completed
            self._mutation_idx = len(self._mutation_targets)
            self._mutation_targets = []
            self._mutation_idx = 0
            self._current_round += 1

            # Transition based on enabled phases
            if self.config.crossover_enabled:
                self._prepare_crossover_groups()
                self._current_phase = RoundPhase.CROSSOVER
                logger.info(f"All mutation rounds complete, transitioning to crossover (round {self._current_round})")
            else:
                # Stay in mutation mode
                self._current_phase = RoundPhase.MUTATION
                logger.info(f"All mutation rounds complete, continuing with mutation (round {self._current_round})")

        elif phase == RoundPhase.CROSSOVER:
            # Update crossover index
            self._crossover_idx = len(self._crossover_groups)
            self._current_round += 1

            # Transition based on enabled phases
            if self.config.mutation_enabled:
                self._current_phase = RoundPhase.MUTATION
                logger.info(f"All crossover rounds complete, transitioning to mutation (round {self._current_round})")
            else:
                # Stay in crossover mode, prepare new groups
                self._prepare_crossover_groups()
                self._current_phase = RoundPhase.CROSSOVER
                logger.info(f"All crossover rounds complete, continuing with crossover (round {self._current_round})")

    # ── Gate operator teks-suffix LAMA (hemat GPU di mode laten) ──────────────
    # !!! apakah karena perbedaan prompt?
    def _mutation_suffix(self, parent: StrategyTrajectory) -> str:
        """strategy_suffix mutation via operator LAMA (1 generasi LLM kv_and_text).
        Di mode laten DILEWATI: loop memakai run_evolution(kind='mutation') yang
        membaca teks parent langsung & mengabaikan suffix → panggilannya mubazir."""
        if self.latent_mode:
            return ""
        return self.mutation_op.generate_mutation_prompt_suffix(parent)

    def _crossover_suffix(self, parents: list) -> str:
        """strategy_suffix crossover via operator LAMA (1 generasi LLM kv_and_text).
        Di mode laten DILEWATI (run_evolution(kind='crossover') baca teks parent)."""
        if self.latent_mode:
            return ""
        return self.crossover_op.generate_crossover_prompt_suffix(parents)

    def _get_original_task(self) -> Optional[dict[str, Any]]:
        """Get next original round task."""
        # Find a direction that hasn't completed original
        for d in range(self.config.num_directions):
            if d not in self._directions_completed:
                return {
                    "phase": RoundPhase.ORIGINAL,
                    "direction_id": d,
                    "parent_trajectories": [], # Original has no parents
                    "strategy_suffix": "",  # No guidance for original
                    "round_idx": self._current_round,
                }

        # All directions completed original, transition to next phase
        self._current_round += 1
        return self._transition_to_next_phase_after_original()

    def _transition_to_next_phase_after_original(self) -> Optional[dict[str, Any]]:
        """
        Determine and transition to the next phase after original round completes.

        Returns the first task of the next phase, or None if evolution is complete.
        """
        #* Case 1: Both mutation and crossover enabled - follow standard flow
        # original -> mutation dulu
        if self.config.mutation_enabled and self.config.crossover_enabled:
            self._current_phase = RoundPhase.MUTATION
            logger.info(f"All original rounds complete, transitioning to mutation (round {self._current_round})")
            return self._get_mutation_task()

        #* Case 2: Only mutation enabled - go to mutation
        # original -> mutation -> mutation -> ...
        elif self.config.mutation_enabled:
            self._current_phase = RoundPhase.MUTATION
            logger.info(f"All original rounds complete, transitioning to mutation (round {self._current_round})")
            return self._get_mutation_task()

        #* Case 3: Only crossover enabled - go to crossover
        # original -> crossover -> crossover -> ...
        elif self.config.crossover_enabled:
            self._prepare_crossover_groups()
            self._current_phase = RoundPhase.CROSSOVER
            logger.info(f"All original rounds complete, transitioning to crossover (round {self._current_round})")
            return self._get_crossover_task()

        # Case 4: Neither enabled - evolution is complete after original
        else:
            logger.info("Neither mutation nor crossover enabled, evolution complete after original")
            return None

    def _get_mutation_task(self) -> Optional[dict[str, Any]]:
        """Get next mutation round task."""
        # If mutation is disabled, skip to crossover or stay in mutation loop
        if not self.config.mutation_enabled:
            if self.config.crossover_enabled:
                #* siapkan crossover groups untuk round pertama(kasus mutation disable)
                self._prepare_crossover_groups()
                self._current_phase = RoundPhase.CROSSOVER
                self._current_round += 1
                return self._get_crossover_task()
            return None

        # If mutation targets not prepared, prepare them
        if not self._mutation_targets:
            self._prepare_mutation_targets() #* siapkan list trajectory untuk dimutasi di round ini

        # Process next mutation target
        while self._mutation_idx < len(self._mutation_targets):
            parent = self._mutation_targets[self._mutation_idx]
            direction_id = self._mutation_idx  # Use index as new direction ID

            # Check if this mutation already exists (hindari duplikasi)
            existing = [t for t in self.pool.get_all()
                       if t.round_idx == self._current_round
                       and t.phase == RoundPhase.MUTATION
                       and parent.trajectory_id in t.parent_ids]
            if existing:
                self._mutation_idx += 1
                continue #* skip jika sudah ada

            # Generate mutation guidance
            suffix = self._mutation_suffix(parent)
            #* generate prompt suffix berisi info parent trajectory
            #* yang akan disisipkan ke prompt LLM supaya tahu harus "memutasi" apa

            # !!! perlu penyesuaian lagi karena beberapa komentar berdasarkan pada arsitektur lama dan sudah tidak valid.
            task = {
                "phase": RoundPhase.MUTATION,
                "direction_id": direction_id,
                "parent_trajectories": [parent],
                "strategy_suffix": suffix,      #* instruksi mutation untuk LLM
                "round_idx": self._current_round,
                "parent_kv": None,
            }

            self._mutation_idx += 1
            return task

        # All mutation tasks complete, transition to next phase
        self._mutation_targets = []  # Reset for next mutation round
        self._mutation_idx = 0
        self._current_round += 1

        # Determine next phase based on config
        if self.config.crossover_enabled:
            self._prepare_crossover_groups()
            self._current_phase = RoundPhase.CROSSOVER
            logger.info(f"All mutation rounds complete, transitioning to crossover (round {self._current_round})")
            return self._get_crossover_task()
        else:
            # Stay in mutation mode (mutation-only loop)
            logger.info(f"All mutation rounds complete, continuing with mutation (round {self._current_round})")
            return self._get_mutation_task()

    def _prepare_mutation_targets(self):
        """
        Prepare targets for current mutation round.

        For the first mutation round (after original), mutate each original trajectory.
        For subsequent mutation rounds (after crossover), mutate each crossover result.
        """
        self._mutation_targets = []
        self._mutation_idx = 0

        # Get the previous round's outputs
        prev_round = self._current_round - 1

        if prev_round < 0:
            # This shouldn't happen - mutation should come after original
            logger.warning("Mutation round before any original rounds")
            return

        # Get trajectories from the previous phase
        prev_phase_trajs = []

        # After original round (round 0), we mutate original trajectories
        if self._current_round == 1:
            prev_phase_trajs = self.pool.get_by_phase(RoundPhase.ORIGINAL)
            #* mutation pertama: mutasi dari trajectory original (round 0)
        else:
            # After crossover round, we mutate the crossover outputs
            # Find crossover trajectories from the most recent crossover round
            all_crossover = self.pool.get_by_phase(RoundPhase.CROSSOVER)
            # Get the most recent crossover round index
            if all_crossover:
                max_crossover_round = max(t.round_idx for t in all_crossover)
                prev_phase_trajs = [t for t in all_crossover if t.round_idx == max_crossover_round]

        if not prev_phase_trajs:
            # Fallback: get all trajectories from previous round
            prev_phase_trajs = [t for t in self.pool.get_all() if t.round_idx == prev_round]

        # Sort by direction_id for consistent ordering
        prev_phase_trajs.sort(key=lambda t: t.direction_id)
        self._mutation_targets = prev_phase_trajs

        # Update active branch count
        self._active_branch_count = len(self._mutation_targets)

        logger.info(f"Prepared {len(self._mutation_targets)} mutation targets for round {self._current_round}")

# !!! coba dicek lagi kodennya, karena takut ada beberapa yang terlewat akhibat perubahan arsitektur dan ada folder baru(./latent_mas)
    def _prepare_crossover_groups(self):
        """
        Prepare crossover groups for the next crossover round.

        Crossover candidates are selected from the two most recent rounds:
        - First crossover (after round 1): original (round 0) + mutation (round 1)
        - Subsequent crossovers: previous mutation + previous crossover

        This ensures that crossover combines the latest evolutionary results,
        not arbitrarily old trajectories.
        """
        # Find the two most recent rounds to use as crossover candidates
        candidates = self._get_crossover_candidates()
        #* ambil trajectory dari 2 round terbaru

        if len(candidates) < self.config.crossover_size:
            logger.warning(f"Not enough candidates for crossover: {len(candidates)} < {self.config.crossover_size}")
            self._crossover_groups = []
            self._crossover_idx = 0
            return #* tidak cukup kandidat untuk crossover, skip crossover round

        #* pilih pasangan crossover dari kandidat yang ada
        self._crossover_groups = self.crossover_op.select_crossover_pairs(
            candidates=candidates,
            crossover_size=self.config.crossover_size,
            crossover_n=self.config.crossover_n,
            prefer_diverse=self.config.prefer_diverse_crossover,
            selection_strategy=self.config.parent_selection_strategy,
            top_percent_threshold=self.config.top_percent_threshold,
            diversity_lambda=self.config.diversity_lambda,
        )

        self._crossover_idx = 0
        logger.info(f"Prepared {len(self._crossover_groups)} crossover groups from {len(candidates)} candidates")

    def _get_crossover_candidates(self) -> list[StrategyTrajectory]:
        """
        Get candidates for crossover from the two most recent relevant rounds.

        Logic depends on enabled phases:
        - Both enabled:
          - First crossover: original (round 0) + mutation (round 1)
          - Subsequent crossovers: latest mutation + latest crossover
        - Crossover-only (mutation disabled):
          - First crossover: original trajectories only
          - Subsequent crossovers: two most recent crossover rounds

        Returns:
            List of trajectories to use as crossover candidates
        """
        all_trajs = self.pool.get_all()
        if not all_trajs:
            return []

        # Get trajectories by phase
        original_trajs = self.pool.get_by_phase(RoundPhase.ORIGINAL)
        mutation_trajs = self.pool.get_by_phase(RoundPhase.MUTATION)
        crossover_trajs = self.pool.get_by_phase(RoundPhase.CROSSOVER)

        # Find the most recent mutation round
        latest_mutation_round = -1
        if mutation_trajs:
            latest_mutation_round = max(t.round_idx for t in mutation_trajs)

        # Find the most recent crossover round
        latest_crossover_round = -1
        if crossover_trajs:
            latest_crossover_round = max(t.round_idx for t in crossover_trajs)

        candidates = []

        # ========================================================
        # CROSSOVER-ONLY MODE (mutation disabled)
        # ========================================================
        if not self.config.mutation_enabled:
            # Case 1: First crossover - use original trajectories
            if latest_crossover_round < 0:
                candidates.extend(original_trajs)
                logger.info(f"First crossover (crossover-only mode): using {len(original_trajs)} original trajectories")

            # Case 2: Subsequent crossovers - use two most recent crossover rounds
            else:
                # Get unique crossover round indices, sorted descending
                crossover_rounds = sorted(set(t.round_idx for t in crossover_trajs), reverse=True)

                if len(crossover_rounds) >= 2:
                    # Use two most recent crossover rounds
                    round1, round2 = crossover_rounds[0], crossover_rounds[1]
                    trajs_round1 = [t for t in crossover_trajs if t.round_idx == round1]
                    trajs_round2 = [t for t in crossover_trajs if t.round_idx == round2]
                    candidates.extend(trajs_round1)
                    candidates.extend(trajs_round2)
                    logger.info(f"Crossover-only mode: using {len(trajs_round1)} from round {round1} + "
                               f"{len(trajs_round2)} from round {round2}")
                else:
                    # Only one crossover round exists, use that + original
                    latest_crossovers = [t for t in crossover_trajs if t.round_idx == latest_crossover_round]
                    candidates.extend(latest_crossovers)
                    candidates.extend(original_trajs)
                    logger.info(f"Crossover-only mode (fallback): using {len(latest_crossovers)} crossover + "
                               f"{len(original_trajs)} original")

            return candidates

        # ========================================================
        # STANDARD MODE (mutation enabled)
        # ========================================================
        # Case 1: First crossover (no previous crossover exists)
        # Use: original + latest mutation
        if latest_crossover_round < 0:
            candidates.extend(original_trajs)
            if latest_mutation_round >= 0:
                candidates.extend([t for t in mutation_trajs if t.round_idx == latest_mutation_round])
            logger.info(f"First crossover: using {len(original_trajs)} original + "
                       f"{len(candidates) - len(original_trajs)} mutation (round {latest_mutation_round})")

        # Case 2: Subsequent crossover
        # Use: latest mutation + latest crossover
        else:
            # Add latest mutation trajectories
            if latest_mutation_round >= 0:
                latest_mutations = [t for t in mutation_trajs if t.round_idx == latest_mutation_round]
                candidates.extend(latest_mutations)
                logger.info(f"Adding {len(latest_mutations)} mutation trajectories from round {latest_mutation_round}")

            # Add latest crossover trajectories
            latest_crossovers = [t for t in crossover_trajs if t.round_idx == latest_crossover_round]
            candidates.extend(latest_crossovers)
            logger.info(f"Adding {len(latest_crossovers)} crossover trajectories from round {latest_crossover_round}")

        return candidates
    # [terjawab — skripsi Bab 4 §Evolusi dan Aliran KV]: di jalur latent saat ini, evolusi
    #   mentransfer parent sebagai TEKS (run_evolution(parent_text)); agen guidance membangun
    #   KV sendiri dari teks itu. Jadi parent_kv pada task praktis vestigial untuk jalur
    #   latent — parent_kv=None TIDAK menghilangkan info parent (teks tetap dibaca guidance).
    def _get_crossover_task(self) -> Optional[dict[str, Any]]:
        """Get next crossover round task."""
        # If crossover is disabled, skip to mutation or stay in crossover loop
        if not self.config.crossover_enabled:
            if self.config.mutation_enabled:
                self._current_phase = RoundPhase.MUTATION
                self._current_round += 1
                return self._get_mutation_task()
            return None

        # Check if there are remaining crossover groups
        if self._crossover_idx >= len(self._crossover_groups):
            # All crossover tasks complete, transition to next phase
            self._current_round += 1

            if self.config.mutation_enabled:
                self._current_phase = RoundPhase.MUTATION
                logger.info(f"All crossover rounds complete, transitioning to mutation (round {self._current_round})")
                return self._get_mutation_task()
            else:
                # Stay in crossover mode (crossover-only loop)
                # Prepare next crossover groups from the two most recent rounds
                self._prepare_crossover_groups()
                logger.info(f"All crossover rounds complete, continuing with crossover (round {self._current_round})")
                return self._get_crossover_task()

        # Get next crossover group
        parents = self._crossover_groups[self._crossover_idx]

        # Crossover guidance (teks-suffix + seed KV) lewat operator LAMA HANYA untuk
        # mode standar. _crossover_suffix sudah melewati panggilan LLM di mode laten.
        suffix = self._crossover_suffix(parents)

        if self.latent_mode:
            # run_evolution membaca TEKS parent → loop mengabaikan parent_kv. Skip
            # total seleksi KV (tak ada last_kv karena generate tak dipanggil).
            seed_kv = None
        else:
            # Gunakan KV output crossover LLM (crossover_kv) sebagai seed propose,
            # bukan best_parent_kv (feedback_kv mem-prime format feedback). Fallback
            # ke best_parent_kv bila crossover LLM tak produce KV (collapse text_only).
            crossover_kv = self.crossover_op.last_kv
            best_parent_kv = None
            best_metric = -float("inf")
            for p in parents:
                if p.kv_cache is not None:
                    m = p.get_primary_metric() or 0.0
                    if m > best_metric:
                        best_metric = m
                        best_parent_kv = p.kv_cache
            seed_kv = crossover_kv if crossover_kv is not None else best_parent_kv

        task = {
            "phase": RoundPhase.CROSSOVER,
            "direction_id": self._crossover_idx,  # Use crossover index as direction
            "parent_trajectories": parents,
            "strategy_suffix": suffix,
            "round_idx": self._current_round,
            "parent_kv": seed_kv,
        }

        self._crossover_idx += 1
        return task

    def report_task_complete(
        self,
        task: dict[str, Any],
        trajectory: StrategyTrajectory
    ):
        """
        Report that a task has been completed.

        Args:
            task: The task that was completed
            trajectory: The resulting trajectory
        """
        # Add trajectory to pool (save ke JSON)
        self.pool.add(trajectory)
        # [terjawab — investigasi]: DIGUNAKAN (bukan sekadar disimpan). record() di bawah →
        #   render_hint → di-inject sebagai negative_hint ke prompt proposal ronde berikut.
        # Rekam ke negative memory bila trajectory GAGAL (evaluasi menyeluruh:
        # ekspresi + metrik IC/ICIR). Ambang sama dengan gate is_successful.
        thr_ic = getattr(self.config, "success_ic_threshold", 0.0)
        thr_icir = getattr(self.config, "success_icir_threshold", 0.0)
        reason = self.negative_memory.record(trajectory, thr_ic, thr_icir)
        if reason:
            logger.info(
                f"Negative memory: recorded failed mechanism "
                f"{trajectory.trajectory_id} [{reason}]"
            )

        # Update state based on phase
        phase = task["phase"]
        direction_id = task["direction_id"]

        if phase == RoundPhase.ORIGINAL:
            self._directions_completed.add(direction_id)
            logger.info(f"Original round complete for direction {direction_id}")

        elif phase == RoundPhase.MUTATION:
            logger.info(f"Mutation round complete for direction {direction_id}")

        elif phase == RoundPhase.CROSSOVER:
            logger.info(f"Crossover round complete (group {direction_id})")

    def create_trajectory_from_loop_result( #* convert hasil AlphaAgentLoop menjadi StrategyTrajectory untuk disimpan di pool
        self,
        task: dict[str, Any],
        hypothesis: Any,
        experiment: Any,
        feedback: Any,
        hypothesis_embedding: Optional[list[float]] = None,
        kv_cache: Optional[Any] = None,
    ) -> StrategyTrajectory:
        """
        Create a trajectory from loop execution results.

        Args:
            task: The task that was executed
            hypothesis: The hypothesis object
            experiment: The experiment object (with factors and results)
            feedback: The feedback object
            hypothesis_embedding: Hidden state embedding from propose step [d]
            kv_cache: Pipeline KV-cache from end of loop (in-memory, not serialized)

        Returns:
            A new StrategyTrajectory
        """
        phase = task["phase"]
        direction_id = task["direction_id"]
        round_idx = task["round_idx"]

        # Generate trajectory ID
        traj_id = StrategyTrajectory.generate_id(direction_id, round_idx, phase)

        # Extract hypothesis info
        hypothesis_text = str(hypothesis) if hypothesis else ""
        hypothesis_details = {}
        if hypothesis:
            #* ambil semua atribut hipotesis yang relevan
            for attr in ["hypothesis", "reason", "concise_reason", "concise_observation",
                        "concise_justification", "concise_knowledge"]:
                if hasattr(hypothesis, attr):
                    hypothesis_details[attr] = getattr(hypothesis, attr, "")

        # Extract factor info
        # explanation (intent) per ekspresi dari front-end construct → audit/repair.
        _expl_by_expr = {
            (f.get("expression") or "").replace(" ", ""): f.get("explanation", "")
            for f in (getattr(experiment, "front_factors", None) or [])
        }
        factors = []
        if experiment and hasattr(experiment, "sub_tasks"):
            #* ambil semua faktor: nama, rumus, deskripsi dan kode kalau ada
            for idx, task_obj in enumerate(experiment.sub_tasks):
                expr = getattr(task_obj, "factor_expression", "")
                factor_info = {
                    "name": getattr(task_obj, "factor_name", f"factor_{idx}"),
                    "expression": expr,
                    "description": getattr(task_obj, "factor_description", ""),
                    "explanation": _expl_by_expr.get((expr or "").replace(" ", ""), ""),
                }
                # Try to get code
                if (hasattr(experiment, "sub_workspace_list") and
                    idx < len(experiment.sub_workspace_list)):
                    ws = experiment.sub_workspace_list[idx]
                    if ws and hasattr(ws, "code_dict") and ws.code_dict:
                        factor_info["code"] = ws.code_dict.get("factor.py", "")
                factors.append(factor_info)

        # Extract backtest metrics
        backtest_metrics = {}
        backtest_result = getattr(experiment, "result", None) if experiment else None
        if backtest_result is not None:
            backtest_metrics = self._extract_metrics(backtest_result)

        # HYBRID: per-factor RankIC (reward `L` paper) dari runner.develop → agregat
        # ke metrics. FactorIC_mean dipakai get_primary_metric (seleksi evolution);
        # dict mentah disimpan di extra_info untuk inspeksi.
        factor_ic = getattr(experiment, "factor_ic", None) or {}
        ic_vals = [v for v in factor_ic.values() if isinstance(v, (int, float))]
        if ic_vals:
            backtest_metrics["FactorIC_mean"] = float(sum(ic_vals) / len(ic_vals))
            backtest_metrics["FactorIC_max"] = float(max(ic_vals))

        # ICIR = stabilitas IC (gate is_successful). Mean dari per-factor ICIR.
        factor_icir = getattr(experiment, "factor_icir", None) or {}
        icir_vals = [v for v in factor_icir.values() if isinstance(v, (int, float))]
        if icir_vals:
            backtest_metrics["FactorICIR_mean"] = float(sum(icir_vals) / len(icir_vals))

        # Extract feedback info
        feedback_text = str(feedback) if feedback else ""
        feedback_details = {}
        if feedback:
            for attr in ["observations", "hypothesis_evaluation", "new_hypothesis",
                        "reason", "decision"]:
                if hasattr(feedback, attr):
                    feedback_details[attr] = getattr(feedback, attr, "")

        # Get parent IDs
        parent_ids = [p.trajectory_id for p in task.get("parent_trajectories", [])]

        return StrategyTrajectory(
            trajectory_id=traj_id,
            direction_id=direction_id,
            round_idx=round_idx,
            phase=phase,
            hypothesis=hypothesis_text,
            hypothesis_details=hypothesis_details,
            factors=factors,
            backtest_result=backtest_result,
            backtest_metrics=backtest_metrics,
            feedback=feedback_text,
            feedback_details=feedback_details,
            parent_ids=parent_ids,
            hypothesis_embedding=hypothesis_embedding,
            kv_cache=kv_cache,
            extra_info={
                "factor_ic": factor_ic,
                "factor_icir": factor_icir,
                "correlation_dropped": getattr(experiment, "correlation_dropped", []),
                # AUDIT front-end (no-crop): keputusan gate per-ekspresi + repair.
                "gate_log": getattr(experiment, "front_gate_log", []),
                "repaired": getattr(experiment, "front_repaired", False),
                "repair_attempts": getattr(experiment, "front_repair_attempts", 0),
                "gate_error": getattr(experiment, "front_gate_error", ""),
            },
        )

    def _extract_metrics(self, result: Any) -> dict[str, Optional[float]]:
        """Extract metrics from backtest result."""
        import pandas as pd

        metrics = {
            "IC": None,
            "ICIR": None,
            "RankIC": None,
            "RankICIR": None,
            "annualized_return": None,
            "information_ratio": None,
            "max_drawdown": None
        }

        if result is None:
            return metrics

        try:
            index_mapping = {
                'IC': ['IC', 'ic'],
                'ICIR': ['ICIR', 'icir'],
                'RankIC': ['RankIC', 'Rank IC', 'rank_ic'],
                'RankICIR': ['RankICIR', 'Rank ICIR', 'rank_icir'],
                'annualized_return': [
                    '1day.excess_return_with_cost.annualized_return',
                    '1day.excess_return_without_cost.annualized_return',
                    'annualized_return',
                    'Annualized Return'
                ],
                'information_ratio': [
                    '1day.excess_return_with_cost.information_ratio',
                    '1day.excess_return_without_cost.information_ratio',
                    'information_ratio',
                    'Information Ratio'
                ],
                'max_drawdown': [
                    '1day.excess_return_with_cost.max_drawdown',
                    '1day.excess_return_without_cost.max_drawdown',
                    'max_drawdown',
                    'Max Drawdown'
                ],
            }

            if isinstance(result, pd.DataFrame):
                col = result.columns[0] if len(result.columns) > 0 else 0
                for target, names in index_mapping.items():
                    for name in names:
                        if name in result.index:
                            val = result.loc[name, col] if col in result.columns else result.loc[name]
                            if pd.notna(val):
                                metrics[target] = float(val)
                                break

            elif isinstance(result, pd.Series):
                for target, names in index_mapping.items():
                    for name in names:
                        if name in result.index:
                            val = result[name]
                            if pd.notna(val):
                                metrics[target] = float(val)
                                break
        except Exception as e:
            logger.warning(f"Failed to extract metrics: {e}")

        return metrics

    def is_complete(self) -> bool:
        """Check if evolution is complete."""
        return self._current_round >= self.config.max_rounds
    # [terjawab — skripsi Bab 4 §Seleksi Trajectory dan Pemilihan Induk]: filter sukses
    #   (FactorIC_mean & FactorICIR_mean > ambang) → skor s(τ)=m(τ)−λ·pen_family → urut
    #   menurun → ambil top-n. λ=0 default → murni m(τ).
    def get_best_trajectories(self, top_n: int = 5) -> list[StrategyTrajectory]:
        """Get the best performing trajectories.

        Seleksi pakai effective = primary_metric − λ·family_penalty (HYBRID part-2):
        faktor yang family operatornya redundan terhadap populasi didemosikan agar
        'best' lebih beragam. λ=0 (default) → murni primary_metric (perilaku lama)."""
        from latent_mas.operator_families import trajectory_families, diversity_penalized
        all_trajs = self.pool.get_all()

        # Filter to successful trajectories (gate IC+ICIR dari config)
        thr_ic = getattr(self.config, "success_ic_threshold", 0.0)
        thr_icir = getattr(self.config, "success_icir_threshold", 0.0)
        valid = [t for t in all_trajs if t.is_successful(thr_ic, thr_icir)]

        lam = getattr(self.config, "diversity_lambda", 0.0)
        scored = diversity_penalized(
            valid,
            lambda t: trajectory_families(t.factors),
            lambda t: t.get_primary_metric(),
            lam,
        )
        scored.sort(key=lambda x: x[1], reverse=True)
        return [t for t, _ in scored[:top_n]]

    def save_state(self, path: Path):
        """Save controller state to disk."""
        import json

        state = {
            "current_round": self._current_round,
            "current_phase": self._current_phase.value,
            "directions_completed": list(self._directions_completed),
            "crossover_idx": self._crossover_idx,
            "active_branch_count": self._active_branch_count,
            "mutation_idx": self._mutation_idx,
            "mutation_target_ids": [t.trajectory_id for t in self._mutation_targets],
            "config": {
                "num_directions": self.config.num_directions,
                "max_rounds": self.config.max_rounds,
                "mutation_enabled": self.config.mutation_enabled,
                "crossover_enabled": self.config.crossover_enabled,
                "crossover_size": self.config.crossover_size,
                "crossover_n": self.config.crossover_n,
            }
        }

        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)

        logger.info(f"Saved evolution state to {path}")

    def load_state(self, path: Path):
        """Load controller state from disk."""
        import json

        if not path.exists():
            logger.warning(f"State file not found: {path}")
            return

        with open(path, "r", encoding="utf-8") as f:
            state = json.load(f)

        self._current_round = state.get("current_round", 0)
        self._current_phase = RoundPhase(state.get("current_phase", "original"))
        self._directions_completed = set(state.get("directions_completed", []))
        self._crossover_idx = state.get("crossover_idx", 0)
        self._active_branch_count = state.get("active_branch_count", self.config.num_directions)
        self._mutation_idx = state.get("mutation_idx", 0)

        # Restore mutation targets from IDs
        mutation_target_ids = state.get("mutation_target_ids", [])
        self._mutation_targets = []
        for tid in mutation_target_ids:
            traj = self.pool.get(tid)
            if traj:
                self._mutation_targets.append(traj)

        # Re-prepare crossover groups if in crossover phase.
        # _prepare_crossover_groups() me-reset _crossover_idx ke 0 — simpan lalu
        # pulihkan cursor dari state agar group yang sudah selesai tidak diulang
        # saat resume di tengah fase crossover.
        if self._current_phase == RoundPhase.CROSSOVER:
            _saved_idx = self._crossover_idx
            self._prepare_crossover_groups()
            self._crossover_idx = min(_saved_idx, len(self._crossover_groups))

        logger.info(f"Loaded evolution state from {path}")

