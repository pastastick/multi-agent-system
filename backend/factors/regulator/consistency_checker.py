"""
Static (no-LLM) consistency checks for factors.

Currently implements:
- Variable Declaration Consistency: every `$xxx` token in the expression
  must be in the allowed base set or explicitly declared in `variables`.

Complexity and redundancy checkers remain available for the quality gate.
"""

import re
from typing import Tuple, List, Dict, Any, Optional
from dataclasses import dataclass, field

from log import logger


ALLOWED_BASE_VARIABLES = frozenset({"$open", "$close", "$high", "$low", "$volume", "$return"})

_VAR_TOKEN_RE = re.compile(r"\$[A-Za-z_][A-Za-z0-9_]*")


@dataclass
class ConsistencyCheckResult:
    """Static consistency check outcome."""
    is_consistent: bool
    severity: str  # none, minor, critical
    overall_feedback: str
    undeclared_variables: List[str] = field(default_factory=list)
    unused_declared_variables: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_consistent": self.is_consistent,
            "severity": self.severity,
            "overall_feedback": self.overall_feedback,
            "undeclared_variables": list(self.undeclared_variables),
            "unused_declared_variables": list(self.unused_declared_variables),
        }


def _normalize_var(name: str) -> str:
    if not name:
        return name
    return name if name.startswith("$") else "$" + name


class StaticConsistencyChecker:
    """Pure-Python variable-declaration consistency check (no LLM calls)."""

    def __init__(self, enabled: bool = True):
        self.enabled = enabled

    def check(
        self,
        factor_name: str,
        factor_expression: str,
        variables: Optional[Dict[str, str]] = None,
    ) -> ConsistencyCheckResult:
        """Return result; severity=critical if undeclared vars used, minor if declared but unused."""
        if not self.enabled:
            return ConsistencyCheckResult(
                is_consistent=True,
                severity="none",
                overall_feedback="Consistency check disabled",
            )

        used_vars = set(_VAR_TOKEN_RE.findall(factor_expression or ""))
        declared_extra = {
            _normalize_var(k) for k in (variables or {}).keys() if k
        }
        allowed = ALLOWED_BASE_VARIABLES | declared_extra

        undeclared = sorted(used_vars - allowed)
        unused = sorted(declared_extra - used_vars)

        if undeclared:
            feedback = (
                f"Undeclared variable(s) in expression: {', '.join(undeclared)}. "
                f"Allowed base variables are {sorted(ALLOWED_BASE_VARIABLES)}; "
                f"any other variable must be declared in the factor's `variables` dict."
            )
            logger.warning(f"[Consistency] {factor_name}: undeclared={undeclared}")
            return ConsistencyCheckResult(
                is_consistent=False,
                severity="critical",
                overall_feedback=feedback,
                undeclared_variables=undeclared,
                unused_declared_variables=unused,
            )

        if unused:
            feedback = (
                f"Declared variable(s) not used in expression: {', '.join(unused)}. "
                f"Remove from `variables` or reference them in the expression."
            )
            logger.info(f"[Consistency] {factor_name}: unused declared={unused}")
            return ConsistencyCheckResult(
                is_consistent=True,
                severity="minor",
                overall_feedback=feedback,
                unused_declared_variables=unused,
            )

        return ConsistencyCheckResult(
            is_consistent=True,
            severity="none",
            overall_feedback="Variable declarations consistent.",
        )


class ComplexityChecker:
    """Factor complexity checker: validates expression complexity."""

    def __init__(
        self,
        enabled: bool = True,
        symbol_length_threshold: int = 250,
        base_features_threshold: int = 6,
        free_args_ratio_threshold: float = 0.5
    ):
        """Args: enabled, symbol_length_threshold, base_features_threshold, free_args_ratio_threshold."""
        self.enabled = enabled
        self.symbol_length_threshold = symbol_length_threshold
        self.base_features_threshold = base_features_threshold
        self.free_args_ratio_threshold = free_args_ratio_threshold

    def check(self, expression: str) -> Tuple[bool, str]:
        """Check expression complexity. Returns (passed, feedback)."""
        if not self.enabled:
            return True, "Complexity check disabled"

        try:
            from factors.coder.factor_ast import (
                calculate_symbol_length,
                count_base_features,
                count_free_args,
                count_all_nodes
            )

            feedback_parts = []
            passed = True

            symbol_length = calculate_symbol_length(expression)
            if symbol_length > self.symbol_length_threshold:
                passed = False
                feedback_parts.append(
                    f"Symbol Length (SL) Check Failed: {symbol_length} > {self.symbol_length_threshold}. "
                    f"Expression is too complex and may lead to overfitting."
                )

            num_base_features = count_base_features(expression)
            if num_base_features > self.base_features_threshold:
                passed = False
                feedback_parts.append(
                    f"Base Features (ER) Check Failed: {num_base_features} > {self.base_features_threshold}. "
                    f"Using too many raw features."
                )

            num_free_args = count_free_args(expression)
            num_all_nodes = count_all_nodes(expression)
            if num_all_nodes > 0:
                free_args_ratio = num_free_args / num_all_nodes
                if free_args_ratio > self.free_args_ratio_threshold:
                    passed = False
                    feedback_parts.append(
                        f"Free Args Ratio Check Failed: {free_args_ratio:.2%} > {self.free_args_ratio_threshold:.2%}. "
                        f"Factor is over-parameterized."
                    )

            if passed:
                return True, "Complexity check passed"
            else:
                return False, "\n".join(feedback_parts)

        except Exception as e:
            logger.warning(f"Complexity check failed with error: {e}")
            return True, f"Complexity check skipped due to error: {e}"


class RedundancyChecker:
    """Redundancy checker: detects duplication with existing factors."""

    def __init__(
        self,
        enabled: bool = True,
        duplication_threshold: int = 5,
        factor_zoo_path: str = None
    ):
        """Args: enabled, duplication_threshold, factor_zoo_path."""
        self.enabled = enabled
        self.duplication_threshold = duplication_threshold
        self.factor_zoo_path = factor_zoo_path
        self._factor_regulator = None

    @property
    def factor_regulator(self):
        """Lazy-load FactorRegulator."""
        if self._factor_regulator is None:
            from factors.regulator.factor_regulator import FactorRegulator
            self._factor_regulator = FactorRegulator(
                factor_zoo_path=self.factor_zoo_path,
                duplication_threshold=self.duplication_threshold
            )
        return self._factor_regulator

    def check(self, expression: str) -> Tuple[bool, str, Dict[str, Any]]:
        """Check expression redundancy. Returns (passed, feedback, details)."""
        if not self.enabled:
            return True, "Redundancy check disabled", {}

        try:
            if not self.factor_regulator.is_parsable(expression):
                return False, f"Expression cannot be parsed: {expression}", {}

            success, eval_dict = self.factor_regulator.evaluate(expression)
            if not success:
                return False, f"Failed to evaluate expression", {}

            duplicated_size = eval_dict.get('duplicated_subtree_size', 0)
            if duplicated_size > self.duplication_threshold:
                matched_alpha = eval_dict.get('matched_alpha', 'Unknown')
                duplicated_subtree = eval_dict.get('duplicated_subtree', '')
                return False, (
                    f"Redundancy Check Failed: Duplicated subtree size ({duplicated_size}) "
                    f"exceeds threshold ({self.duplication_threshold}). "
                    f"Matched with: {matched_alpha}. Duplicated subtree: {duplicated_subtree}"
                ), eval_dict

            return True, "Redundancy check passed", eval_dict

        except Exception as e:
            logger.warning(f"Redundancy check failed with error: {e}")
            return True, f"Redundancy check skipped due to error: {e}", {}


class FactorQualityGate:
    """Quality gate: integrates static consistency + complexity + redundancy checks."""

    def __init__(
        self,
        consistency_checker: StaticConsistencyChecker = None,
        complexity_checker: ComplexityChecker = None,
        redundancy_checker: RedundancyChecker = None,
        consistency_enabled: bool = True,
        complexity_enabled: bool = True,
        redundancy_enabled: bool = True
    ):
        """Args: optional checker instances; *_enabled flags toggle each check."""
        self.consistency_checker = consistency_checker or StaticConsistencyChecker(enabled=consistency_enabled)
        self.complexity_checker = complexity_checker or ComplexityChecker(enabled=complexity_enabled)
        self.redundancy_checker = redundancy_checker or RedundancyChecker(enabled=redundancy_enabled)

        self.consistency_checker.enabled = consistency_enabled
        self.complexity_checker.enabled = complexity_enabled
        self.redundancy_checker.enabled = redundancy_enabled

    def evaluate(
        self,
        factor_name: str,
        factor_expression: str,
        variables: Dict[str, str] = None,
    ) -> Tuple[bool, str, Dict[str, Any]]:
        """Evaluate factor against all enabled checks. Returns (passed, feedback, results)."""
        results = {
            "consistency": None,
            "complexity": None,
            "redundancy": None,
        }
        feedbacks = []
        all_passed = True

        if self.consistency_checker.enabled:
            consistency_result = self.consistency_checker.check(
                factor_name=factor_name,
                factor_expression=factor_expression,
                variables=variables,
            )
            results["consistency"] = consistency_result.to_dict()

            if not consistency_result.is_consistent:
                all_passed = False
                feedbacks.append(f"[Consistency] {consistency_result.overall_feedback}")

        if self.complexity_checker.enabled:
            complexity_passed, complexity_feedback = self.complexity_checker.check(factor_expression)
            results["complexity"] = {
                "passed": complexity_passed,
                "feedback": complexity_feedback
            }

            if not complexity_passed:
                all_passed = False
                feedbacks.append(f"[Complexity] {complexity_feedback}")

        if self.redundancy_checker.enabled:
            redundancy_passed, redundancy_feedback, redundancy_details = self.redundancy_checker.check(factor_expression)
            results["redundancy"] = {
                "passed": redundancy_passed,
                "feedback": redundancy_feedback,
                "details": redundancy_details
            }

            if not redundancy_passed:
                all_passed = False
                feedbacks.append(f"[Redundancy] {redundancy_feedback}")

        if all_passed:
            overall_feedback = f"Factor '{factor_name}' passed all quality gates."
            logger.info(overall_feedback)
        else:
            overall_feedback = f"Factor '{factor_name}' failed quality gates:\n" + "\n".join(feedbacks)
            logger.warning(overall_feedback)

        return all_passed, overall_feedback, results
