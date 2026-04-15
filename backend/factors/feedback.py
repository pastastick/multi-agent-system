"""
memformat hasil backtest (current vs SOTA) jadi string untuk prompt LLM
"""


import json
from pathlib import Path

import pandas as pd
from jinja2 import Environment, StrictUndefined

from core.experiment import Experiment
from core.prompts import Prompts
from core.proposal import (
    Hypothesis,
    HypothesisExperiment2Feedback,
    HypothesisFeedback,
    Trace,
)
from log import logger
from llm.client import LocalLLMBackend, robust_json_parse
from utils import convert2bool

# Max retries for JSON parsing
MAX_JSON_PARSE_RETRIES = 3

base_feedback_prompts = Prompts(file_path=Path(__file__).parent / "prompts" / "prompts.yaml")
DIRNAME = Path(__file__).absolute().resolve().parent

#! yang dipakai ini
def process_results(current_result, sota_result):
    # Convert the results to dataframes
    current_df = pd.DataFrame(current_result)
    
    # Handle case where sota_result might be None or empty
    if sota_result is None or (isinstance(sota_result, pd.DataFrame) and sota_result.empty):
        # If no SOTA result, return only current result
        current_df.index.name = "metric"
        if "0" in current_df.columns:
            current_df.rename(columns={"0": "Current Result"}, inplace=True)
        elif len(current_df.columns) > 0:
            # Use first column if "0" doesn't exist
            first_col = current_df.columns[0]
            current_df.rename(columns={first_col: "Current Result"}, inplace=True)
        
        # Select important metrics for comparison
        important_metrics = [
            "1day.excess_return_without_cost.max_drawdown",         # kerugian terbesar di puncak
            "1day.excess_return_without_cost.information_ratio",    # excess return/ tracking error
            "1day.excess_return_without_cost.annualized_return",    # return tahunan
            "IC",                                                   # korelasi faktor value dengan future return
        ]
        
        # Filter the DataFrame to retain only the important metrics that exist
        available_metrics = [m for m in important_metrics if m in current_df.index]
        if available_metrics:
            filtered_df = current_df.loc[available_metrics]
        else:
            filtered_df = current_df
        
        return filtered_df.to_string()
    
    sota_df = pd.DataFrame(sota_result)

    # Set the metric as the index
    current_df.index.name = "metric"
    sota_df.index.name = "metric"

    # Rename the value column to reflect the result type
    # Handle case where column "0" might not exist
    if "0" in current_df.columns:
        current_df.rename(columns={"0": "Current Result"}, inplace=True)
    elif len(current_df.columns) > 0:
        first_col = current_df.columns[0]
        current_df.rename(columns={first_col: "Current Result"}, inplace=True)
    else:
        # If no columns, create a dummy column
        current_df["Current Result"] = 0
    
    if "0" in sota_df.columns:
        sota_df.rename(columns={"0": "SOTA Result"}, inplace=True)
    elif len(sota_df.columns) > 0:
        first_col = sota_df.columns[0]
        sota_df.rename(columns={first_col: "SOTA Result"}, inplace=True)
    else:
        # If no columns, create a dummy column
        sota_df["SOTA Result"] = 0

    # Combine the dataframes on the Metric index
    combined_df = pd.concat([current_df, sota_df], axis=1)

    # Select important metrics for comparison
    important_metrics = [
        "1day.excess_return_without_cost.max_drawdown",
        "1day.excess_return_without_cost.information_ratio",
        "1day.excess_return_without_cost.annualized_return",
        "IC",
    ]

    # Filter the combined DataFrame to retain only the important metrics that exist
    available_metrics = [m for m in important_metrics if m in combined_df.index]
    if available_metrics:
        filtered_combined_df = combined_df.loc[available_metrics]
    else:
        filtered_combined_df = combined_df

    # Check if both columns exist before comparing
    #* menyuruh LLM untuk memutuskan "lebih baik" SOTA atau new faktor
    if "Current Result" in filtered_combined_df.columns and "SOTA Result" in filtered_combined_df.columns:
        filtered_combined_df[
            "Bigger columns name (Didn't consider the direction of the metric, you should judge it by yourself that bigger is better or smaller is better)"
        ] = filtered_combined_df.apply(
                lambda row: "Current Result" if pd.notna(row["Current Result"]) and pd.notna(row["SOTA Result"]) and row["Current Result"] > row["SOTA Result"] else "SOTA Result", axis=1
        )
    elif "Current Result" in filtered_combined_df.columns:
        # Only current result available
        filtered_combined_df[
            "Bigger columns name (Didn't consider the direction of the metric, you should judge it by yourself that bigger is better or smaller is better)"
        ] = "Current Result"
    elif "SOTA Result" in filtered_combined_df.columns:
        # Only SOTA result available
        filtered_combined_df[
            "Bigger columns name (Didn't consider the direction of the metric, you should judge it by yourself that bigger is better or smaller is better)"
        ] = "SOTA Result"

    return filtered_combined_df.to_string()


class QlibFactorHypothesisExperiment2Feedback(HypothesisExperiment2Feedback):
    def generate_feedback(self, exp: Experiment, hypothesis: Hypothesis, trace: Trace) -> HypothesisFeedback:
        """
        Generate feedback for the given experiment and hypothesis.

        Args:
            exp (QlibFactorExperiment): The experiment to generate feedback for.
            hypothesis (QlibFactorHypothesis): The hypothesis to generate feedback for.
            trace (Trace): The trace of the experiment.

        Returns:
            Any: The feedback generated for the given experiment and hypothesis.
        """
        logger.info("Generating feedback...")
        hypothesis_text = hypothesis.hypothesis
        current_result = exp.result
        tasks_factors = [task.get_task_information_and_implementation_result() for task in exp.sub_tasks]
        # Safely get SOTA result, handle case where based_experiments might be empty or result is None
        sota_result = None
        if exp.based_experiments and len(exp.based_experiments) > 0:
            sota_result = exp.based_experiments[-1].result

        # Process the results to filter important metrics
        combined_result = process_results(current_result, sota_result)

        # Generate the system prompt
        sys_prompt = (
            Environment(undefined=StrictUndefined)
            .from_string(base_feedback_prompts["factor_feedback_generation"]["system"])
            .render(scenario=self.scen.get_scenario_all_desc())
        )

        # Generate the user prompt
        usr_prompt = (
            Environment(undefined=StrictUndefined)
            .from_string(base_feedback_prompts["factor_feedback_generation"]["user"])
            .render(
                hypothesis_text=hypothesis_text,
                task_details=tasks_factors,
                combined_result=combined_result,
            )
        )

        # Call the APIBackend to generate the response for hypothesis feedback with retry
        response_json = None
        last_error = None
        
        for attempt in range(MAX_JSON_PARSE_RETRIES):
            try:
                response = self._call_llm(usr_prompt, sys_prompt, json_mode=True)
                # Parse the JSON response using robust parser
                response_json = robust_json_parse(response)
                break
            except json.JSONDecodeError as e:
                last_error = e
                logger.warning(f"[QuantaAlpha] JSON parse failed (attempt {attempt + 1}/{MAX_JSON_PARSE_RETRIES}): {e}")
                if attempt < MAX_JSON_PARSE_RETRIES - 1:
                    logger.info("[QuantaAlpha] Re-requesting LLM...")
                continue
        
        if response_json is None:
            logger.error(f"[QuantaAlpha] JSON parse still failed after {MAX_JSON_PARSE_RETRIES} attempts")
            return HypothesisFeedback(
                observations="JSON parse failed; could not extract feedback",
                hypothesis_evaluation="Unable to evaluate",
                new_hypothesis="",
                reason=f"JSON parse error: {last_error}",
                decision=False,
            )

        # Extract fields from JSON response
        observations = response_json.get("Observations", "No observations provided")
        hypothesis_evaluation = response_json.get("Feedback for Hypothesis", "No feedback provided")
        new_hypothesis = response_json.get("New Hypothesis", "No new hypothesis provided")
        reason = response_json.get("Reasoning", "No reasoning provided")
        decision = convert2bool(response_json.get("Replace Best Result", "no"))

        return HypothesisFeedback(
            observations=observations,
            hypothesis_evaluation=hypothesis_evaluation,
            new_hypothesis=new_hypothesis,
            reason=reason,
            decision=decision,
        )


#! ini yang dipakai.
qa_feedback_prompts = Prompts(file_path=Path(__file__).parent / "prompts" / "prompts.yaml")
class AlphaAgentQlibFactorHypothesisExperiment2Feedback(HypothesisExperiment2Feedback):

    def __init__(self, scen) -> None:
        super().__init__(scen)
        # ── KV-cache state (opsional) ─────────────────────────────
        self._past_kv = None        # KV-cache input untuk panggilan berikutnya
        self.last_result = None     # LLMResult lengkap dari panggilan terakhir

    def set_past_kv(self, kv) -> None:
        """Set KV-cache yang akan dipakai di _call_llm berikutnya."""
        self._past_kv = kv

    @property
    def last_kv(self):
        """KV-cache output dari panggilan _call_llm terakhir."""
        if self.last_result is not None:
            return getattr(self.last_result, "kv_cache", None)
        return None

    def _get_scenario_desc(self) -> str:
        """Scenario description untuk system prompt generate_feedback.

        Override di subclass Latent untuk pakai compact desc.
        Default: full scenario (semua section) karena feedback
        perlu konteks lengkap untuk mengevaluasi hasil backtest.
        """
        return self.scen.get_scenario_all_desc()

    def _call_llm(self, user_prompt: str, system_prompt: str, json_mode: bool = False) -> str:
        """
        Call LLM via self.llm_backend (LocalLLMBackend).

        self.llm_backend di-set oleh AlphaAgentLoop.__init__.
        Auto-detect: jika _past_kv tersedia dan backend mendukung
        build_messages_and_run, pakai latent path.
        Override di subclass Latent untuk parameter lanjutan (role, latent_steps).
        """
        # ── Latent path: auto-detect KV-cache support ──
        if self._past_kv is not None and hasattr(self.llm_backend, "build_messages_and_run"):
            result = self.llm_backend.build_messages_and_run(
                user_prompt=user_prompt,
                system_prompt=system_prompt,
                json_mode=json_mode,
                past_key_values=self._past_kv,
                mode="kv_and_text",
            )
            self.last_result = result
            logger.info(
                f"[Feedback] KV auto-detect: has_kv={result.has_kv}, "
                f"text_len={len(result.text or '')}"
            )
            return result.text or ""

        # ── Standard text-only path ──
        # Reset last_result agar last_kv tidak return KV stale
        # dari panggilan latent sebelumnya.
        self.last_result = None
        return self.llm_backend.build_messages_and_create_chat_completion(
            user_prompt=user_prompt, system_prompt=system_prompt, json_mode=json_mode
        )

    def generate_feedback(self, exp: Experiment, hypothesis: Hypothesis, trace: Trace) -> HypothesisFeedback:
        """
        Generate feedback for the given experiment and hypothesis.

        Args:
            exp (QlibFactorExperiment): The experiment to generate feedback for.
            hypothesis (QlibFactorHypothesis): The hypothesis to generate feedback for.
            trace (Trace): The trace of the experiment.

        Returns:
            Any: The feedback generated for the given experiment and hypothesis.
        """
        logger.info("Generating feedback...")
        hypothesis_text = hypothesis.hypothesis
        current_result = exp.result
        tasks_factors = [task.get_task_information_and_implementation_result() for task in exp.sub_tasks]
        # Safely get SOTA result, handle case where based_experiments might be empty or result is None
        sota_result = None
        if exp.based_experiments and len(exp.based_experiments) > 0:
            sota_result = exp.based_experiments[-1].result

        # Extract complexity information by directly calculating from factor expressions
        # Import complexity calculation functions
        try:
            from factors.coder.factor_ast import (
                calculate_symbol_length, count_base_features
            )
            from factors.coder.config import FACTOR_COSTEER_SETTINGS
            
            for idx, task_detail in enumerate(tasks_factors):
                if idx < len(exp.sub_tasks):
                    task = exp.sub_tasks[idx]
                    factor_expr = task_detail.get("factor_expression", "")
                    if factor_expr:
                        complexity_warnings = []
                        
                        # Calculate symbol length
                        symbol_length = calculate_symbol_length(factor_expr)
                        symbol_length_threshold = getattr(FACTOR_COSTEER_SETTINGS, 'symbol_length_threshold', 300)
                        if symbol_length > symbol_length_threshold:
                            complexity_warnings.append(
                                f"Symbol Length (SL) Check Failed: Symbol length ({symbol_length}) exceeds threshold ({symbol_length_threshold}). "
                                f"The factor expression is too complex and may lead to overfitting."
                            )
                        
                        # Calculate base features count
                        num_base_features = count_base_features(factor_expr)
                        base_features_threshold = getattr(FACTOR_COSTEER_SETTINGS, 'base_features_threshold', 6)
                        if num_base_features > base_features_threshold:
                            complexity_warnings.append(
                                f"Base Features Count (ER) Check Failed: Number of base features ({num_base_features}) exceeds threshold ({base_features_threshold}). "
                                f"The factor uses too many raw features, which may indicate over-engineering."
                            )
                        
                        if complexity_warnings:
                            task_detail["complexity_feedback"] = "\n".join(complexity_warnings)
        except Exception as e:
            logger.warning(f"Failed to calculate complexity info: {e}")

        # Process the results to filter important metrics
        combined_result = process_results(current_result, sota_result)

        # Generate the system prompt
        scenario_desc = self._get_scenario_desc()
        sys_prompt = (
            Environment(undefined=StrictUndefined)
            .from_string(qa_feedback_prompts["factor_feedback_generation"]["system"])
            .render(scenario=scenario_desc)
        )

        # Generate the user prompt
        usr_prompt = (
            Environment(undefined=StrictUndefined)
            .from_string(qa_feedback_prompts["factor_feedback_generation"]["user"])
            .render(
                hypothesis_text=hypothesis_text,
                task_details=tasks_factors,
                combined_result=combined_result,
            )
        )

        # Call the APIBackend to generate the response for hypothesis feedback with retry
        response_json = None
        last_error = None
        
        for attempt in range(MAX_JSON_PARSE_RETRIES):
            try:
                response = self._call_llm(usr_prompt, sys_prompt, json_mode=True)
                # Parse the JSON response using robust parser
                response_json = robust_json_parse(response)
                break
            except json.JSONDecodeError as e:
                last_error = e
                logger.warning(f"[AlphaAgent] JSON parse failed (attempt {attempt + 1}/{MAX_JSON_PARSE_RETRIES}): {e}")
                if attempt < MAX_JSON_PARSE_RETRIES - 1:
                    logger.info("[AlphaAgent] Re-requesting LLM...")
                continue
        
        if response_json is None:
            logger.error(f"[AlphaAgent] JSON parse still failed after {MAX_JSON_PARSE_RETRIES} attempts")
            return HypothesisFeedback(
                observations="JSON parse failed; could not extract feedback",
                hypothesis_evaluation="Unable to evaluate",
                new_hypothesis="",
                reason=f"JSON parse error: {last_error}",
                decision=False,
            )

        # Extract fields from JSON response
        observations = response_json.get("Observations", "No observations provided")
        hypothesis_evaluation = response_json.get("Feedback for Hypothesis", "No feedback provided")
        new_hypothesis = response_json.get("New Hypothesis", "No new hypothesis provided")
        reason = response_json.get("Reasoning", "No reasoning provided")
        decision = convert2bool(response_json.get("Replace Best Result", "no"))

        return HypothesisFeedback(
            observations=observations,
            hypothesis_evaluation=hypothesis_evaluation,
            new_hypothesis=new_hypothesis,
            reason=reason,
            decision=decision,
        )

#! untuk model experiment
class QlibModelHypothesisExperiment2Feedback(HypothesisExperiment2Feedback):
    """Generated feedbacks on the hypothesis from **Executed** Implementations of different tasks & their comparisons with previous performances"""

    def generate_feedback(self, exp: Experiment, hypothesis: Hypothesis, trace: Trace) -> HypothesisFeedback:
        """
        The `ti` should be executed and the results should be included, as well as the comparison between previous results (done by LLM).
        For example: `mlflow` of Qlib will be included.
        """

        logger.info("Generating feedback...")
        # Define the system prompt for hypothesis feedback
        system_prompt = feedback_prompts["model_feedback_generation"]["system"]

        # Define the user prompt for hypothesis feedback
        context = trace.scen
        SOTA_hypothesis, SOTA_experiment = trace.get_sota_hypothesis_and_experiment()

        user_prompt = (
            Environment(undefined=StrictUndefined)
            .from_string(feedback_prompts["model_feedback_generation"]["user"])
            .render(
                context=context,
                last_hypothesis=SOTA_hypothesis,
                last_task=SOTA_experiment.sub_tasks[0].get_task_information() if SOTA_hypothesis else None,
                last_code=SOTA_experiment.sub_workspace_list[0].code_dict.get("model.py") if SOTA_hypothesis else None,
                last_result=SOTA_experiment.result if SOTA_hypothesis else None,
                hypothesis=hypothesis,
                exp=exp,
            )
        )

        # Call the APIBackend to generate the response for hypothesis feedback with retry
        response_json_hypothesis = None
        last_error = None
        
        for attempt in range(MAX_JSON_PARSE_RETRIES):
            try:
                response_hypothesis = LocalLLMBackend().build_messages_and_create_chat_completion(
                    user_prompt=user_prompt,
                    system_prompt=system_prompt,
                    json_mode=True,
                )
                # Parse the JSON response using robust parser
                response_json_hypothesis = robust_json_parse(response_hypothesis)
                break
            except json.JSONDecodeError as e:
                last_error = e
                logger.warning(f"[Model] JSON parse failed (attempt {attempt + 1}/{MAX_JSON_PARSE_RETRIES}): {e}")
                if attempt < MAX_JSON_PARSE_RETRIES - 1:
                    logger.info("[Model] Re-requesting LLM...")
                continue
        
        if response_json_hypothesis is None:
            logger.error(f"[Model] JSON parse still failed after {MAX_JSON_PARSE_RETRIES} attempts")
            return HypothesisFeedback(
                observations="JSON parse failed; could not extract feedback",
                hypothesis_evaluation="Unable to evaluate",
                new_hypothesis="",
                reason=f"JSON parse error: {last_error}",
                decision=False,
            )
        
        return HypothesisFeedback(
            observations=response_json_hypothesis.get("Observations", "No observations provided"),
            hypothesis_evaluation=response_json_hypothesis.get("Feedback for Hypothesis", "No feedback provided"),
            new_hypothesis=response_json_hypothesis.get("New Hypothesis", "No new hypothesis provided"),
            reason=response_json_hypothesis.get("Reasoning", "No reasoning provided"),
            decision=convert2bool(response_json_hypothesis.get("Decision", "false")),
        )
