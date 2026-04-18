"""
Factor workflow with session control and evolution support.

Supports three round phases:
- Original: Initial exploration in each direction
- Mutation: Orthogonal exploration from parent trajectories
- Crossover: Hybrid strategies from multiple parents

Supports parallel execution within each phase when enabled.
"""

from typing import Any, List, Optional
from pathlib import Path #* manipulasi path antar platform
import fire #* ubah fungsi jadi cli
import signal #* sinyal os
import sys #* sys.exit() untuk force timeout
import threading #*stop signal
from multiprocessing import Process, Queue #* running task di proses terpisah
from functools import wraps #* agar nama fungsi asli terjaga
import time
import ctypes
import os
import pickle
from pipeline.settings import ALPHA_AGENT_FACTOR_PROP_SETTING 
from pipeline.planning import generate_parallel_directions
from pipeline.planning import load_run_config
from pipeline.loop import AlphaAgentLoop

# External agents (optional — imported lazily to avoid hard dependency)
try:
    from eksternal.base import ExternalAgentBase, ExternalInsight
    from pipeline.insight import InsightOrchestrator, InsightResult, build_orchestrator
    _HAS_EXTERNAL = True
except ImportError:
    _HAS_EXTERNAL = False
    ExternalAgentBase = Any     # type: ignore[assignment,misc]
    ExternalInsight = Any       # type: ignore[assignment,misc]
    InsightOrchestrator = Any   # type: ignore[assignment,misc]
    InsightResult = Any         # type: ignore[assignment,misc]
from pipeline.evolution import (
    EvolutionController, 
    EvolutionConfig,
    StrategyTrajectory,
    RoundPhase,
)
from core.exception import FactorEmptyError
from log import logger
from log.time import measure_time
from llm.config import LLM_SETTINGS

# Pipeline monitor (safe import)
try:
    from debug import PipelineMonitor, set_monitor, get_monitor, reset_monitor
    _HAS_MONITOR = True
except ImportError:
    _HAS_MONITOR = False


# YAML key (under `latent:`) → ALPHA_AGENT_FACTOR_PROP_SETTING attribute.
# YAML takes precedence over QLIB_FACTOR_* env vars so experiments
# are reproducible from a single config file. Applied before
# create_llm_backend() so the backend picks up overridden values.
_LATENT_YAML_TO_SETTING = {
    "enabled":               "latent_enabled",
    "model_name":            "latent_model_name",
    "device":                "latent_device",
    "steps":                 "latent_steps",
    "steps_propose":         "latent_steps_propose",
    "steps_construct":       "latent_steps_construct",
    "steps_coder":           "latent_steps_coder",
    "steps_feedback":        "latent_steps_feedback",
    "use_realign":           "use_realign",
    "enable_thinking":       "enable_thinking",
    "kv_max_tokens":         "kv_max_tokens",
    "store_kv":              "store_kv",
    "knn_enabled":           "knn_enabled",
    "knn_percentage":        "knn_percentage",
    "knn_min_keep":          "knn_min_keep",
    "knn_strategy":          "knn_strategy",
    "max_new_tokens":        "max_new_tokens",
    "temperature":           "temperature",
    "top_p":                 "top_p",
    "temperature_propose":   "temperature_propose",
    "temperature_construct": "temperature_construct",
    "temperature_coder":     "temperature_coder",
    "temperature_feedback":  "temperature_feedback",
    "log_tensors":           "log_tensors",
}


def _apply_latent_overrides(run_cfg: dict, setting) -> None:
    """Override latent-pipeline fields in `setting` from run_cfg['latent'].

    Keys present in YAML win over env-var defaults. Keys absent from
    YAML leave the env-var / class default untouched. Unknown keys
    are ignored with a warning so typos are visible.
    """
    if not isinstance(run_cfg, dict):
        return
    latent_cfg = run_cfg.get("latent") or {}
    if not latent_cfg:
        return

    applied: list[str] = []
    for yaml_key, value in latent_cfg.items():
        setting_attr = _LATENT_YAML_TO_SETTING.get(yaml_key)
        if setting_attr is None:
            logger.warning(f"[Config] Unknown latent key in YAML: {yaml_key!r} (ignored)")
            continue
        setattr(setting, setting_attr, value)
        applied.append(f"{setting_attr}={value}")

    if applied:
        logger.info(f"[Config] Latent overrides from YAML → {', '.join(applied)}")


def force_timeout(): #* membatasi lamanya mining
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            seconds = LLM_SETTINGS.factor_mining_timeout
            def handle_timeout(signum, frame):
                logger.error(f"Process terminated: timeout exceeded ({seconds}s)")
                sys.exit(1)
                #* ^ kalau waktu habis -> log error lalu keluar paksa

            signal.signal(signal.SIGALRM, handle_timeout) #* kalau terima signal, panggil handle_timeout
            signal.alarm(seconds)

            try:
                result = func(*args, **kwargs)
            finally:
                signal.alarm(0) #* batalkan alarm jika fungsi selesai sebelum timeout
            return result
        return wrapper
    return decorator


def _run_branch( #* menjalankan satu loop tanpa evolusi, untuk setiap direction yang dihasilkan planning
    direction: str | None, #* arah eksplorasi, misal "momentun cross sectional"
    step_n: int,
    use_local: bool, #* True backtest local, False = Docker
    idx: int, #* nomer urut branch (untuk log)
    log_root: str,
    log_prefix: str, #* prefix nama folder
    quality_gate_cfg: dict = None,
    external_context: Optional[str] = None, #* konteks dari eksternal agent
    llm_backend: Optional[Any] = None,
    past_kv: Optional[Any] = None,
):
    if log_root:
        branch_name = f"{log_prefix}_{idx:02d}" #* nama folder, misal "branch_01"
        branch_log = Path(log_root) / branch_name
        branch_log.mkdir(parents=True, exist_ok=True) #* buat folder jika belum ada
        logger.set_trace_path(branch_log) #* set path log untuk branch ini
    
    model_loop = AlphaAgentLoop(
        ALPHA_AGENT_FACTOR_PROP_SETTING, #* setting untuk loop (misal model, environment, dll)
        potential_direction=direction,
        stop_event=None, #* tidak ada signal stop
        use_local=use_local,
        quality_gate_config=quality_gate_cfg or {},
        external_context=external_context,
        llm_backend=llm_backend,
        past_kv=past_kv,
    )
    model_loop.user_initial_direction = direction #* simpan direction asli user
    model_loop.run(step_n=step_n, stop_event=None) #* run loop untuk branch ini (misal 5 langkah); terjadi proses 'propose -> experiment(code) ->backtest -> feedback -> repeat
    logger.info(f"Branch {idx} done: direction='{direction}'")


def _run_evolution_task( #* jalankan satu task dalam evolution loop (Original/Mutation/Crossover)
    task: dict[str, Any], #* deskripsi task, misal {'phase': RoundPhase.MUTATION, 'direction_id': 0, 'round_idx': 1, 'parent_trajectories': [...]}
    directions: list[str],
    step_n: int, #* step per loop
    use_local: bool,
    user_direction: str | None, #* direction asli user
    log_root: str,
    stop_event: threading.Event | None,
    quality_gate_cfg: dict[str, Any] | None = None,
    external_context: Optional[str] = None,
    llm_backend: Optional[Any] = None,
    past_kv: Optional[Any] = None,
) -> dict[str, Any]:
    """
    Run a single evolution task (one small loop).

    Args:
        task: Evolution task descriptor
        directions: List of original directions
        step_n: Steps per round
        use_local: Use local backtest
        user_direction: User initial direction
        log_root: Log root directory
        stop_event: Stop event
        quality_gate_cfg: Quality gate config

    Returns:
        Dict containing trajectory data
    """
    phase = task["phase"] #* fase task, misal "ORIGINAL", "MUTATION", "CROSSOVER"
    direction_id = task["direction_id"]
    strategy_suffix = task.get("strategy_suffix", "")
    round_idx = task["round_idx"]
    parent_trajectories = task.get("parent_trajectories", [])
    
    # Resolve direction by phase
    if phase == RoundPhase.ORIGINAL: #* fase original : pakai direction dari list planning
        direction = directions[direction_id] if direction_id < len(directions) else None
    elif phase == RoundPhase.MUTATION: #* fase mutation : pakai direction dari list, tapi nanti di mutasi 
        direction = directions[direction_id] if direction_id < len(directions) else None
    else:  #* CROSSOVER : gabungan dari beberapa parent trajectory, tidak ada direction tunggal
        direction = None

    trajectory_id = StrategyTrajectory.generate_id(direction_id, round_idx, phase) #* buat id trajectory, dalam hash
    parent_ids = [p.trajectory_id for p in parent_trajectories] #* ekstrak id dari parent trajectory (kosong untuk original, 1 untuk mutation, banyak untuk crossover)

    if log_root: #* setup log
        branch_name = f"{phase.value}_{round_idx:02d}_{direction_id:02d}"
        branch_log = Path(log_root) / branch_name
        branch_log.mkdir(parents=True, exist_ok=True)
        logger.set_trace_path(branch_log) #* arahkan semua log ke folder ini

    logger.info(f"Starting evolution task: phase={phase.value}, round={round_idx}, direction={direction_id}")

    # Create and run loop
    model_loop = AlphaAgentLoop(
        ALPHA_AGENT_FACTOR_PROP_SETTING,
        potential_direction=direction,
        stop_event=stop_event,
        use_local=use_local,
        strategy_suffix=strategy_suffix,
        evolution_phase=phase.value,
        trajectory_id=trajectory_id,
        parent_trajectory_ids=parent_ids,
        direction_id=direction_id,
        round_idx=round_idx,
        quality_gate_config=quality_gate_cfg or {},
        external_context=external_context,
        llm_backend=llm_backend,
        past_kv=past_kv,
    )
    model_loop.user_initial_direction = user_direction #* simpan direction asli user sebelum di pecah jadi sub-direction
    
    # Run one small loop (5 steps)
    model_loop.run(step_n=step_n, stop_event=stop_event) #* running mining loop: propose → code → backtest → feedback × step_n kali

    traj_data = model_loop._get_trajectory_data() #* extrak dict hasil/ trajectory: hypothesis, experiment, feedback, hypothesis_embedding => untuk membuat StrategyTrajectory
    traj_data["task"] = task #* tambahkan info task
    
    return traj_data

def _parallel_task_worker( #* dijalankan di proses anak (child process) via multiprocessing.Process. Satu worker = satu evolution task.
    
    task: dict[str, Any],
    directions: list[str],
    step_n: int,
    use_local: bool,
    user_direction: str | None,
    log_root: str,
    result_queue: Queue, #* antrian untuk kirim hasil kembali ke parent proses
    task_idx: int,
    external_context: Optional[str] = None,
):
    """
    Worker for parallel evolution tasks. Runs one evolution task in a separate process and puts result in queue.
    Args: task, directions, step_n, use_local, user_direction, log_root, result_queue, task_idx.
    """
    try:
        from core.conf import RD_AGENT_SETTINGS
        RD_AGENT_SETTINGS.use_file_lock = False #* matikan file lock di child process karena file lock bisa menyebabkan deadlock saat multiprocessing
        RD_AGENT_SETTINGS.pickle_cache_folder_path_str = str( #* setiap child punya folder cache sendiri
            Path(log_root) / f"pickle_cache_{task_idx}"
        )

        traj_data = _run_evolution_task(
            task=task,
            directions=directions,
            step_n=step_n,
            use_local=use_local,
            user_direction=user_direction,
            log_root=log_root,
            stop_event=None, #* child proses tidak pakai stop event
            external_context=external_context,
        )
        result_queue.put({ #* result_queue adalah multiprocessing.Queue => cara child mengirim hasil kembali ke parent proses
            "success": True,
            "task_idx": task_idx,
            "task": task,
            "traj_data": traj_data, #* hasil mining
        })
    except Exception as e:
        import traceback
        result_queue.put({
            "success": False,
            "task_idx": task_idx,
            "task": task,
            "error": str(e),
            "traceback": traceback.format_exc(),
        }) #* kirim full traceback ke parent untuk debugging


def _serialize_task_for_parallel(task: dict[str, Any]) -> dict[str, Any]: 
    #* Menyiapkan task dict agar bisa dikirim ke child process. multiprocessing.Process perlu data yang bisa di-pickle (serialize).
    """Serialize task for use in child process (parent_trajectories are complex objects)."""
    serialized = task.copy()
    
    # RoundPhase -> string
    if "phase" in serialized and isinstance(serialized["phase"], RoundPhase):
        serialized["phase"] = serialized["phase"]
        #* RoundPhase(str, enum) bisa langsung di-pickle karena inherit str
    
    # Convert parent_trajectories to serializable info
    if "parent_trajectories" in serialized: 
        #* parent trajectory adalah objek kompleks yang tidak bisa di-pickle, jadi hanya kirim id-nya saja ke child process
        serialized["parent_trajectory_ids"] = [
            p.trajectory_id for p in serialized.get("parent_trajectories", [])
        ]
        # Child process does not need full trajectory objects; strategy_suffix has required info
        serialized["parent_trajectories"] = [] 
        #* child process tidak butuh full object, cukup info di strategy_suffix
    
    return serialized


def _run_tasks_parallel(
    tasks: list[dict[str, Any]],
    directions: list[str],
    step_n: int,
    use_local: bool,
    user_direction: str | None,
    log_root: str,
    external_context: Optional[str] = None,
) -> list[dict[str, Any]]:
    """
    Run multiple evolution tasks in parallel.
    Returns list of results, each with task and traj_data.
    """
    if not tasks:
        return []

    result_queue = Queue() #* antrian shared: semua child process akan mengirim hasil ke sini
    processes = []

    logger.info(f"Starting {len(tasks)} parallel evolution tasks")

    for idx, task in enumerate(tasks): #* spawn semua child process
        serialized_task = _serialize_task_for_parallel(task) 
        #* siapkan task yang sudah di serialize agar bisa dikirim ke child process

        p = Process(
            target=_parallel_task_worker, #* fungsi yang dijalankan di child process
            args=(
                serialized_task,
                directions,
                step_n,
                use_local,
                user_direction,
                log_root,
                result_queue,
                idx,
                external_context,
            ),
        )
        p.start()
        processes.append(p)
        logger.info(f"Started task {idx}: phase={task['phase'].value}, direction={task['direction_id']}")

    results = []
    for _ in range(len(tasks)):
        result = result_queue.get() #* blocking: tunggi hasil dari child
        if result["success"]:
            original_task = tasks[result["task_idx"]]
            result["task"] = original_task
            result["traj_data"]["task"] = original_task
            results.append(result)
            logger.info(f"Task {result['task_idx']} completed")
        else:
            logger.error(f"Task {result['task_idx']} failed: {result['error']}")
            logger.error(result.get('traceback', ''))

    for p in processes: #* tunggu semua proses selesai
        p.join()
        #*  blocking: tunggu sampai child process benar-benar terminated
        #*   ini dilakukan SETELAH semua result sudah diambil dari queue
        #*  join() memastikan tidak ada zombie process

    logger.info(f"Parallel tasks done: {len(results)}/{len(tasks)} succeeded")
    
    return results


def _build_strategy_feedback(
    best_trajs: list,
    insights: List["ExternalInsight"],
) -> dict[str, Any]:
    #! LIMITASI => heuristik: satu direction bisa cocok dengan beberapa topik
    #! LIMITASI => kalau direction tidak mengandung keyword,  topik tidak terepresentasi meski performanya bagus
    #! LIMITASI => tidak ada fallback jika semua direction tidak mengandung keyword 
    """
    Build feedback dict untuk ExternalAgentBase.update_strategy().

    Extracts:
    - top_directions     : direction strings dari best trajectories
    - topic_performance  : rough topic → metric mapping
    - successful_queries : queries yang digunakan oleh insights
    """
    top_directions = []
    for t in best_trajs:
        direction = getattr(t, "direction", None) or getattr(t, "hypothesis", None)
        #* ambil direction dari trajectory, kalau tidak ada coba ambil dari hypothesis 
        #*(karena direction bisa jadi None untuk crossover, tapi hypothesis biasanya tetap ada)
        
        if direction and str(direction) not in top_directions:
            top_directions.append(str(direction))

    # Infer topic performance from direction text (heuristic)
    _TOPIC_KEYWORDS = { #* dictionary list topik kata kunci
        "monetary_policy": ["rate", "fed", "ecb", "central bank", "monetary", "hike", "cut"],
        "inflation":       ["inflation", "cpi", "pce", "price", "deflation"],
        "gdp_growth":      ["gdp", "growth", "recession", "expansion"],
        "employment":      ["employment", "jobs", "labor", "payroll", "unemployment"],
        "geopolitical_risk": ["geopolitical", "war", "sanctions", "risk", "conflict"],
        "commodity_prices": ["oil", "gold", "commodity", "energy", "metal"],
        "currency_markets": ["dollar", "usd", "currency", "forex", "yen", "euro"],
        "credit_spreads":  ["credit", "spread", "bond", "yield", "treasury"],
    }
    topic_hits: dict[str, list[float]] = {}
    for t in best_trajs:
        metric = t.get_primary_metric() #* ambil metric utama
        if metric is None:
            continue
        direction_text = str(getattr(t, "direction", "") or "").lower()
        #* ambil teks direction
        for topic, keywords in _TOPIC_KEYWORDS.items():
            if any(kw in direction_text for kw in keywords):
                topic_hits.setdefault(topic, []).append(float(metric))
                #* simpan topik dan metricnya
                
    topic_performance = {
        topic: sum(vals) / len(vals)
        for topic, vals in topic_hits.items()
    } #* rata-rata metric per topik

    successful_queries = []
    for ins in insights:
        successful_queries.extend(ins.search_queries)

    return {
        "top_directions": top_directions,
        "topic_performance": topic_performance,
        "successful_queries": list(dict.fromkeys(successful_queries)),  # dedup
    }

def _run_external_agents(
    agents: List["ExternalAgentBase"],
    llm_backend: Optional[Any] = None,
) -> tuple[List["ExternalInsight"], str, Optional[Any]]:
    """
    Run all external agents and collect insights.

    Returns:
        (insights, combined_context_str, final_kv)
        - insights           → passed to generate_parallel_directions (kv_cache)
        - combined_context_str → passed to AlphaAgentLoop (factor_propose context)
        - final_kv           → KV-cache from last agent, passed to AlphaAgentLoop
    """
    if not agents:
        return [], "", None

    insights: List["ExternalInsight"] = []
    prev_kv = None

    for agent in agents:
        try:
            logger.info(f"[External] Running {getattr(agent, 'agent_name', type(agent).__name__)}")
            insight = agent.run(past_kv=prev_kv) #* jalankan agent (class ExternalAgentBase)
            insights.append(insight)
            #* Chain KV-cache: next agent gets this one's kv_cache as context
            if insight.has_kv: #* kalau agent ini menghasilkan KV-cache, simpan dan teruskan ke agent berikutnya
                prev_kv = insight.kv_cache
            logger.info(
                f"[External] Done: summary_len={len(insight.summary)}, "
                f"has_kv={insight.has_kv}, docs={insight.n_documents_collected}"
            )
        except Exception as e:
            logger.warning(f"[External] Agent {type(agent).__name__} failed: {e}")
            #* kalau agen gagal, log warning tapi LANJUTKAN ke agen berikutnya
            #* satu agen gagal tidak boleh menghentikan seluruh pipeline

    combined_context = "\n\n".join(ins.to_context_str() for ins in insights)
    #* gabung semua context jadi satu string
    
    return insights, combined_context, prev_kv
    #*   insights        → list ExternalInsight (dipakai di planning + feedback)
    #*   combined_context → string (dipakai di AlphaAgentLoop sebagai external_context)
    #*   prev_kv         → KV-cache terakhir (dipakai di planning + AlphaAgentLoop)


#* orkestra seluruh siklus evolusi
def run_evolution_loop(
    initial_direction: str | None,
    evolution_cfg: dict[str, Any], #* config evolution dari YAML
    exec_cfg: dict[str, Any], #* cfg = config
    planning_cfg: dict[str, Any],
    stop_event: threading.Event | None = None,
    quality_gate_cfg: dict[str, Any] | None = None,
    external_agents: Optional[List["ExternalAgentBase"]] = None,
    llm_backend: Optional[Any] = None,
    insight_orchestrator: Optional["InsightOrchestrator"] = None,
    insight_mode: str = "sequential",
):
    """
    Run evolution loop: Original -> Mutation -> Crossover -> Mutation -> ...\n
    Supports parallel execution per phase.

    Args:
        external_agents     : list of ExternalAgentBase instances (e.g. MacroExternalAgent).
                              Run before planning; their KV-cache goes to planning,
                              their summaries go to AlphaAgentLoop as external_context.
        llm_backend         : pre-built LocalLLMBackend instance.  Required when
                              external_agents is set, so planning can reuse the same
                              model that built the KV-cache.
        insight_orchestrator: pre-built InsightOrchestrator.  If provided, takes
                              precedence over external_agents + _run_external_agents.
        insight_mode        : "sequential" or "hierarchical" (for InsightOrchestrator).
    """
    quality_gate_cfg = quality_gate_cfg or {}
    from core.conf import RD_AGENT_SETTINGS
    RD_AGENT_SETTINGS.use_file_lock = False #* matikan file lock
    logger.info("Evolution mode: file lock disabled to avoid deadlock")

    # ── Initialize pipeline monitor for this evolution session ──
    if _HAS_MONITOR:
        _monitor = PipelineMonitor(
            session_name=f"evolution_{Path(str(logger.log_trace_path)).name}",
            log_dir=str(Path(str(logger.log_trace_path)) / "monitor"),
            console_echo=False,
            enabled=True,
        )
        set_monitor(_monitor)
        logger.info(f"[Monitor] Session started: {_monitor.session_id}")

    # ── Run external agents before planning ───────────────────────────────
    external_insights: List["ExternalInsight"] = []
    external_context: Optional[str] = None
    _insight_result: Optional["InsightResult"] = None
    _planning_kv: Optional[Any] = None  #* KV-cache from external/insight → AlphaAgentLoop

    if insight_orchestrator and _HAS_EXTERNAL:
        logger.info(
            f"[External] Running InsightOrchestrator (mode={insight_mode})..."
        )
        _insight_result = insight_orchestrator.run(mode=insight_mode) 
        #* InsightOrchestrator mengoordinasi semua external agents sekaligus (sequential atau hierarical)
        
        external_insights = _insight_result.as_external_insights_list()
        external_context = _insight_result.external_context
        _planning_kv = getattr(_insight_result, 'kv_cache', None)
        #* ambil context dan KV-cache dari orchestrator
        
        if external_context:
            logger.info(f"[External] Unified context ready ({len(external_context)} chars)")
    
    elif external_agents and _HAS_EXTERNAL:
        logger.info(f"[External] Running {len(external_agents)} external agent(s)...")
        external_insights, external_context, _planning_kv = _run_external_agents(
            external_agents, llm_backend
        ) #* jalankan semua external agents satu per satu
        
        if external_context:
            logger.info(f"[External] Context ready ({len(external_context)} chars)")

    # Parse config
    num_directions = int(planning_cfg.get("num_directions", 2))
    max_rounds = int(evolution_cfg.get("max_rounds", 10))
    crossover_size = int(evolution_cfg.get("crossover_size", 2))    #* berapa banyak parent untuk crossover (default 2 = crossover antara 2 trajectory)
    crossover_n = int(evolution_cfg.get("crossover_n", 3))          #* berapa banyak child yang dihasilkan dari satu set parent di crossover (default 3 = dari 2 parent bisa jadi 3 child dengan kombinasi berbeda)
    steps_per_loop = int(exec_cfg.get("steps_per_loop", 5))
    use_local = bool(exec_cfg.get("use_local", True))

    mutation_enabled = bool(evolution_cfg.get("mutation_enabled", True))
    crossover_enabled = bool(evolution_cfg.get("crossover_enabled", True))
    parent_selection_strategy = str(evolution_cfg.get("parent_selection_strategy", "best"))
    top_percent_threshold = float(evolution_cfg.get("top_percent_threshold", 0.3))
    log_root = str(logger.log_trace_path)
    parallel_enabled = bool(evolution_cfg.get("parallel_enabled", False))
    
    if parallel_enabled and llm_backend is not None:
        logger.warning(
            "[Evolution] parallel_enabled=True but llm_backend is set. "
            "LocalLLMBackend (GPU model) cannot be serialized for multiprocessing. "
            "Forcing parallel_enabled=False to use KV-cache latent pipeline."
        )
        parallel_enabled = False
        #* paksa parallel_enabled=False jika llm_backend disediakan, karena LocalLLMBackend tidak bisa di-serialize untuk multiprocessing. 
        #* Dalam mode ini, kita akan menggunakan pendekatan latent pipeline dengan KV-cache untuk sharing konteks antar proses.
        
    fresh_start = bool(evolution_cfg.get("fresh_start", True))
    cleanup_on_finish = bool(evolution_cfg.get("cleanup_on_finish", False))

    # Generate initial directions (with external KV-cache context if available)
    planning_enabled = bool(planning_cfg.get("enabled", False))
    prompt_file = planning_cfg.get("prompt_file") or "planning_prompts.yaml"
    prompt_path = Path(__file__).parent / "prompts" / str(prompt_file)

    if planning_enabled and initial_direction:
        directions = generate_parallel_directions(
            initial_direction=initial_direction,
            n=num_directions,
            prompt_file=prompt_path,
            max_attempts=int(planning_cfg.get("max_attempts", 5)),
            use_llm=bool(planning_cfg.get("use_llm", True)),
            allow_fallback=bool(planning_cfg.get("allow_fallback", True)),
            external_insights=external_insights or None,
            llm_backend=llm_backend,
        )
    elif planning_enabled:
        directions = [None] * num_directions
    else:
        #? initial direction ada berapa dan mengapa tidak perlu generate_parallel_directions?
        directions = [initial_direction] if initial_direction else [None]

    logger.info(f"Generated {len(directions)} exploration directions")
    for i, d in enumerate(directions):
        logger.info(f"  Direction {i}: {d}")

    pool_save_path = Path(log_root) / "trajectory_pool.json"
    mutation_prompt_path = Path(__file__).parent / "prompts" / "evolution_prompts.yaml"
    
    logger.info(f"Trajectory pool path: {pool_save_path} (fresh_start={fresh_start})")


    config = EvolutionConfig(
        num_directions=len(directions),
        steps_per_loop=steps_per_loop,
        max_rounds=max_rounds,
        mutation_enabled=mutation_enabled,
        crossover_enabled=crossover_enabled,
        crossover_size=crossover_size,
        crossover_n=crossover_n,
        prefer_diverse_crossover=bool(evolution_cfg.get("prefer_diverse_crossover", True)),
        parent_selection_strategy=parent_selection_strategy,
        top_percent_threshold=top_percent_threshold,
        parallel_enabled=parallel_enabled,
        pool_save_path=str(pool_save_path),
        mutation_prompt_path=str(mutation_prompt_path) if mutation_prompt_path.exists() else None,
        crossover_prompt_path=str(mutation_prompt_path) if mutation_prompt_path.exists() else None,
        fresh_start=fresh_start,
    )
    #*  controller sekarang punya: pool, mutation_op, crossover_op, state tracking
    #*  llm_backend diteruskan agar mutation/crossover operators bisa pakai
    #*  latent path (parent KV-cache → build_messages_and_run → kv_and_text)
    controller = EvolutionController(config, llm_backend=llm_backend)

    logger.info("="*40)
    logger.info("Starting evolution loop")
    logger.info(f"Config: directions={len(directions)}, max_rounds={max_rounds}, "
               f"crossover_size={crossover_size}, crossover_n={crossover_n}")
    logger.info(f"Phases: mutation={'on' if mutation_enabled else 'off'}, "
               f"crossover={'on' if crossover_enabled else 'off'}")
    if mutation_enabled and not crossover_enabled:
        logger.info("Mode: mutation only (Original -> Mutation -> ...)")
    elif crossover_enabled and not mutation_enabled:
        logger.info("Mode: crossover only (Original -> Crossover -> ...)")
    elif mutation_enabled and crossover_enabled:
        logger.info("Mode: full evolution (Original -> Mutation -> Crossover -> ...)")
    else:
        logger.info("Mode: original only (no evolution)")
    logger.info(f"Parent selection: {parent_selection_strategy}" +
               (f" (top_percent={top_percent_threshold})" if parent_selection_strategy == "top_percent_plus_random" else ""))
    logger.info(f"Parallel execution: {'on' if parallel_enabled else 'off'}")
    logger.info("="*60)

    if parallel_enabled:
        while not controller.is_complete():
            if stop_event and stop_event.is_set():
                logger.info("Stop signal received, ending evolution loop")
                break

            tasks = controller.get_all_tasks_for_current_phase()
            if not tasks:
                logger.info("Evolution complete: no more tasks")
                break

            current_phase = tasks[0]["phase"]
            current_round = tasks[0]["round_idx"]
            logger.info(f"Parallel phase: phase={current_phase.value}, round={current_round}, tasks={len(tasks)}")

            #* jalankan semua task secara parallel
            results = _run_tasks_parallel(
                tasks=tasks,
                directions=directions,
                step_n=steps_per_loop,
                use_local=use_local,
                user_direction=initial_direction,
                log_root=log_root,
                external_context=external_context,
            )
            
            completed_tasks = []
            for result in results:
                if result["success"]:
                    task = result["task"]
                    traj_data = result["traj_data"]

                    #* convert hasil loop -> StrategyTrajectory
                    trajectory = controller.create_trajectory_from_loop_result(
                        task=task,
                        hypothesis=traj_data.get("hypothesis"),
                        experiment=traj_data.get("experiment"),
                        feedback=traj_data.get("feedback"),
                        hypothesis_embedding=traj_data.get("hypothesis_embedding"),
                        kv_cache=traj_data.get("pipeline_kv"),
                    )
                    
                    #* simpan ke pool + update state
                    controller.report_task_complete(task, trajectory)
                    completed_tasks.append(task)
                    logger.info(f"Trajectory done: {trajectory.trajectory_id}, RankIC={trajectory.get_primary_metric()}")

            #* pindah phase: ORIGINAL->MUTATION, MUTATION->CROSSOVER, CROSSOVER->MUTATION (next round)
            controller.advance_phase_after_parallel_completion(completed_tasks)

    else: #* SEQUENTIAL MODE
        while not controller.is_complete():
            if stop_event and stop_event.is_set():
                logger.info("Stop signal received, ending evolution loop")
                break

            #* ambil SATU task berikutnya(bukan semua sekaligus)
            task = controller.get_next_task()
            if task is None:
                logger.info("Evolution complete: no more tasks")
                break

            logger.info(f"Running task: phase={task['phase'].value}, round={task['round_idx']}, direction={task['direction_id']}")

            try:
                # Gunakan parent KV-cache dari evolution task jika ada,
                # fallback ke planning KV dari external agent.
                task_kv = task.get("parent_kv") or _planning_kv

                traj_data = _run_evolution_task(
                    task=task,
                    directions=directions,
                    step_n=steps_per_loop,
                    use_local=use_local,
                    user_direction=initial_direction,
                    log_root=log_root,
                    stop_event=stop_event,
                    quality_gate_cfg=quality_gate_cfg,
                    external_context=external_context,
                    llm_backend=llm_backend,
                    past_kv=task_kv,
                )
                trajectory = controller.create_trajectory_from_loop_result(
                    task=task,
                    hypothesis=traj_data.get("hypothesis"),
                    experiment=traj_data.get("experiment"),
                    feedback=traj_data.get("feedback"),
                    hypothesis_embedding=traj_data.get("hypothesis_embedding"),
                    kv_cache=traj_data.get("pipeline_kv"),
                )
                controller.report_task_complete(task, trajectory)
                logger.info(f"Task done: trajectory_id={trajectory.trajectory_id}, RankIC={trajectory.get_primary_metric()}")

                # Monitor: record evolution round
                if _HAS_MONITOR:
                    try:
                        _em = get_monitor(auto_create=False)
                        if _em:
                            _em.track_evolution_round(
                                round_idx=task["round_idx"],
                                phase=task["phase"].value,
                                direction_id=task["direction_id"],
                                trajectory_id=trajectory.trajectory_id,
                                parent_ids=[p.trajectory_id for p in task.get("parent_trajectories", [])],
                                primary_metric=trajectory.get_primary_metric(),
                                is_successful=trajectory.is_successful(),
                            )
                    except Exception:
                        pass

            except Exception as e:
                logger.error(f"Task failed: {e}")
                import traceback
                logger.error(traceback.format_exc())
                continue
            #* satu task gagal -> log error tapi LANJUTKAN ke task berikutnya

    # simpan state di disk (bisa lanjut nanti)
    state_path = Path(log_root) / "evolution_state.json"
    controller.save_state(state_path)
    
    # ambil top Trajectory
    best_trajs = controller.get_best_trajectories(top_n=5)
    logger.info("="*40)
    logger.info(f"Evolution complete. Top {len(best_trajs)} trajectories:")
    
    # menampilkan trajectory terbaik
    for i, t in enumerate(best_trajs):
        metric = t.get_primary_metric()
        metric_str = f"{metric:.4f}" if metric is not None else "N/A"
        logger.info(f"  {i+1}. {t.trajectory_id}: phase={t.phase.value}, RankIC={metric_str}")
    
    # menampilkan statistik pool
    logger.info(f"Pool stats: {controller.pool.get_statistics()}")
    logger.info("="*40)
    
    # ── Finalize pipeline monitor ────────────────────────────────────────
    if _HAS_MONITOR:
        try:
            _em = get_monitor(auto_create=False)
            if _em:
                summary = _em.finalize()
                if summary:
                    logger.info(f"[Monitor] Session finalized: {_em.event_count} events recorded")
                    _anomaly_count = summary.get("anomalies", {}).get("total", 0)
                    if _anomaly_count > 0:
                        logger.warning(f"[Monitor] {_anomaly_count} anomalies detected — check monitor logs")
                    _bn = summary.get("pipeline", {}).get("bottleneck")
                    if _bn:
                        logger.info(f"[Monitor] Bottleneck: {_bn['step']} ({_bn['percentage']}% of pipeline time)")
        except Exception:
            pass

    # hapus file trajectory pool jika TRUE
    if cleanup_on_finish:
        logger.info("Cleaning up trajectory pool file...")
        controller.pool.cleanup_file()

    # ── Update external agent strategies (3.B) ────────────────────────────
    # Called AFTER all iterations complete. Agents adjust search weights
    # based on which topics/directions produced the best-performing factors.
    if _HAS_EXTERNAL and best_trajs:
        
        # buat feedback dari trajectory terbaik untuk update strategi eksternal agent
        feedback = _build_strategy_feedback(best_trajs, external_insights)
        
        # update semua agent via orchestrator
        if insight_orchestrator:
            insight_orchestrator.update_all_strategies(feedback)
        
        # setiap agent update weight
        elif external_agents:
            for agent in external_agents:
                try:
                    agent.update_strategy(feedback)
                    logger.info(
                        f"[External] Strategy updated for "
                        f"{getattr(agent, 'agent_name', type(agent).__name__)}"
                    )
                except Exception as e:
                    logger.warning(f"[External] Strategy update failed: {e}")


@force_timeout()
def main(
    path=None,
    step_n=100,
    direction=None,
    stop_event=None,
    config_path=None,
    evolution_mode=None,
    external_agents: Optional[List["ExternalAgentBase"]] = None,
    llm_backend: Optional[Any] = None,
    insight_orchestrator: Optional["InsightOrchestrator"] = None,
    insight_mode: str = "sequential",
):
    """
    Autonomous alpha factor mining with optional evolution support.

    Args:
        path: Session path (for resume)
        step_n: Number of steps (default 100 = 20 loops * 5 steps/loop)
        direction: Initial direction
        stop_event: Stop event
        config_path: Run config file path
        evolution_mode: Enable evolution (None=from config, True/False=override)
        insight_orchestrator: InsightOrchestrator instance (sequential/hierarchical).
                              Takes precedence over external_agents if provided.
        insight_mode: "sequential" or "hierarchical" (used by InsightOrchestrator).

    Evolution flow: Original -> Mutation -> Crossover -> Mutation -> ...

    You can continue running session by

    .. code-block:: python

        quantaalpha mine --direction "[Initial Direction]" --config_path configs/experiment.yaml

    """
    try:
        from core.conf import RD_AGENT_SETTINGS
        logger.info("="*60)
        logger.info("Experiment config")
        logger.info(f"  Workspace: {RD_AGENT_SETTINGS.workspace_path}")
        logger.info(f"  Cache dir: {RD_AGENT_SETTINGS.pickle_cache_folder_path_str}")
        logger.info(f"  Cache enabled: {RD_AGENT_SETTINGS.cache_with_pickle}")
        logger.info("="*60)

        # Config file default: project_root/configs/
        _project_root = Path(__file__).resolve().parents[2]
        config_default = _project_root / "configs" / "experiment.yaml"
        config_file = Path(config_path) if config_path else config_default
        run_cfg = load_run_config(config_file)
        planning_cfg = (run_cfg.get("planning") or {}) if isinstance(run_cfg, dict) else {}
        exec_cfg = (run_cfg.get("execution") or {}) if isinstance(run_cfg, dict) else {}
        evolution_cfg = (run_cfg.get("evolution") or {}) if isinstance(run_cfg, dict) else {}
        quality_gate_cfg = (run_cfg.get("quality_gate") or {}) if isinstance(run_cfg, dict) else {}
        external_cfg = (run_cfg.get("external") or {}) if isinstance(run_cfg, dict) else {}

        # YAML overrides env-var defaults for latent pipeline fields.
        # Must happen BEFORE create_llm_backend() so the backend factory
        # picks up the overridden values (latent_steps, use_realign, knn_*, ...).
        _apply_latent_overrides(run_cfg, ALPHA_AGENT_FACTOR_PROP_SETTING)

        # YAML override for insight_mode (external agent orchestration).
        # Caller-provided arg is overridden by YAML so the config file
        # stays the single source of truth for reproducibility.
        _yaml_insight_mode = external_cfg.get("insight_mode")
        if _yaml_insight_mode:
            insight_mode = str(_yaml_insight_mode)
            logger.info(f"[Config] insight_mode from YAML → {insight_mode}")

        if evolution_mode is not None:
            use_evolution = evolution_mode
        else:
            use_evolution = bool(evolution_cfg.get("enabled", False))

        # ── Auto-create llm_backend dari settings ────────────────────────
        # Jika latent_enabled=True di settings dan belum ada llm_backend
        # yang di-pass manual, buat LocalLLMBackend otomatis.
        # Model di-load sekali, di-share ke seluruh pipeline.
        if llm_backend is None and ALPHA_AGENT_FACTOR_PROP_SETTING.latent_enabled:
            logger.info("[LatentPipeline] latent_enabled=True, creating LocalLLMBackend...")
            llm_backend = ALPHA_AGENT_FACTOR_PROP_SETTING.create_llm_backend()
            _knn_status = (
                f"knn={ALPHA_AGENT_FACTOR_PROP_SETTING.knn_percentage:.0%}"
                f"/{ALPHA_AGENT_FACTOR_PROP_SETTING.knn_strategy}"
                if ALPHA_AGENT_FACTOR_PROP_SETTING.knn_enabled
                else "knn=off"
            )
            logger.info(
                f"[LatentPipeline] Backend ready: "
                f"model={ALPHA_AGENT_FACTOR_PROP_SETTING.latent_model_name}, "
                f"device={ALPHA_AGENT_FACTOR_PROP_SETTING.latent_device}, "
                f"latent_steps={ALPHA_AGENT_FACTOR_PROP_SETTING.latent_steps}, "
                f"use_realign={ALPHA_AGENT_FACTOR_PROP_SETTING.use_realign}, "
                f"{_knn_status}"
            )

        if step_n is None or step_n == 100:
            if exec_cfg.get("step_n") is not None:
                step_n = exec_cfg.get("step_n")
            else:
                max_loops = int(exec_cfg.get("max_loops", 10))
                steps_per_loop = int(exec_cfg.get("steps_per_loop", 5))
                step_n = max_loops * steps_per_loop

        use_local = os.getenv("USE_LOCAL", "True").lower()
        use_local = True if use_local in ["true", "1"] else False
        if exec_cfg.get("use_local") is not None:
            use_local = bool(exec_cfg.get("use_local"))
        exec_cfg["use_local"] = use_local
        
        logger.info(f"Use {'Local' if use_local else 'Docker container'} to execute factor backtest")
        
        if use_evolution and path is None:
            logger.info("="*60)
            logger.info("Evolution mode: Original -> Mutation -> Crossover loop")
            logger.info("="*60)
            
            run_evolution_loop(
                initial_direction=direction,
                evolution_cfg=evolution_cfg,
                exec_cfg=exec_cfg,
                planning_cfg=planning_cfg,
                stop_event=stop_event,
                quality_gate_cfg=quality_gate_cfg,
                external_agents=external_agents,
                llm_backend=llm_backend,
                insight_orchestrator=insight_orchestrator,
                insight_mode=insight_mode,
            )

        elif path is None:
            # ── Run external agents (non-evolution path) ───────────────────
            ext_insights: List["ExternalInsight"] = []
            ext_context: Optional[str] = None
            _planning_kv: Optional[Any] = None  # KV-cache → AlphaAgentLoop

            if insight_orchestrator and _HAS_EXTERNAL:
                logger.info(
                    f"[External] Running InsightOrchestrator "
                    f"(mode={insight_mode})..."
                )
                _insight_result = insight_orchestrator.run(mode=insight_mode)
                ext_insights = _insight_result.as_external_insights_list()
                ext_context = _insight_result.external_context
                _planning_kv = getattr(_insight_result, 'kv_cache', None)
            elif external_agents and _HAS_EXTERNAL:
                logger.info(f"[External] Running {len(external_agents)} agent(s)...")
                ext_insights, ext_context, _planning_kv = _run_external_agents(
                    external_agents, llm_backend
                )

            planning_enabled = bool(planning_cfg.get("enabled", False))
            n_dirs = int(planning_cfg.get("num_directions", 1))
            max_attempts = int(planning_cfg.get("max_attempts", 5))
            use_llm = bool(planning_cfg.get("use_llm", True))
            allow_fallback = bool(planning_cfg.get("allow_fallback", True))
            prompt_file = planning_cfg.get("prompt_file") or "planning_prompts.yaml"
            prompt_path = Path(__file__).parent / "prompts" / str(prompt_file)
            if planning_enabled and direction:
                directions = generate_parallel_directions(
                    initial_direction=direction,
                    n=n_dirs,
                    prompt_file=prompt_path,
                    max_attempts=max_attempts,
                    use_llm=use_llm,
                    allow_fallback=allow_fallback,
                    external_insights=ext_insights or None,
                    llm_backend=llm_backend,
                )
            else:
                directions = [direction] if direction else [None]

            log_root = exec_cfg.get("branch_log_root") or "log"
            log_prefix = exec_cfg.get("branch_log_prefix") or "branch"
            use_branch_logs = planning_enabled and len(directions) > 1
            parallel_execution = bool(exec_cfg.get("parallel_execution", False))
            if parallel_execution and llm_backend is not None:
                logger.warning(
                    "[Main] parallel_execution=True but llm_backend is set. "
                    "Forcing parallel_execution=False to use KV-cache latent pipeline."
                )
                parallel_execution = False

            if parallel_execution and len(directions) > 1:
                procs: list[Process] = []
                for idx, dir_text in enumerate(directions, start=1):
                    if dir_text:
                        logger.info(f"[Planning] Branch {idx}/{len(directions)} direction: {dir_text}")
                    p = Process(
                        target=_run_branch,
                        args=(dir_text, step_n, use_local, idx, log_root if use_branch_logs else "", log_prefix, quality_gate_cfg, ext_context),
                    )
                    p.start()
                    procs.append(p)
                for p in procs:
                    p.join()
            else:
                # SEQUENTIAL
                for idx, dir_text in enumerate(directions, start=1):
                    if dir_text:
                        logger.info(f"[Planning] Branch {idx}/{len(directions)} direction: {dir_text}")
                    if use_branch_logs:
                        branch_name = f"{log_prefix}_{idx:02d}"
                        branch_log = Path(log_root) / branch_name
                        branch_log.mkdir(parents=True, exist_ok=True)
                        logger.set_trace_path(branch_log)
                    model_loop = AlphaAgentLoop(
                        ALPHA_AGENT_FACTOR_PROP_SETTING,
                        potential_direction=dir_text,
                        stop_event=stop_event,
                        use_local=use_local,
                        quality_gate_config=quality_gate_cfg,
                        external_context=ext_context,
                        llm_backend=llm_backend,
                        past_kv=_planning_kv,
                    )
                    model_loop.user_initial_direction = direction
                    model_loop.run(step_n=step_n, stop_event=stop_event)
        else:
            # resume dari session yang sudah ada
            model_loop = AlphaAgentLoop.load(path, use_local=use_local)
            model_loop.run(step_n=step_n, stop_event=stop_event)
    except Exception as e:
        logger.error(f"Error during execution: {str(e)}")
        raise
    finally:
        logger.info("Run finished or terminated")

if __name__ == "__main__":
    fire.Fire(main)
