"""
Factor Regulator Module.

Provides:
- FactorRegulator: Factor duplication and complexity checking
- StaticConsistencyChecker: Pure-Python variable-declaration consistency
- FactorQualityGate: Integrated quality gate
"""

from factors.regulator.factor_regulator import FactorRegulator

try:
    from factors.regulator.consistency_checker import (
        StaticConsistencyChecker,
        ConsistencyCheckResult,
        ComplexityChecker,
        RedundancyChecker,
        FactorQualityGate,
    )
    CONSISTENCY_CHECKER_AVAILABLE = True
except ImportError:
    CONSISTENCY_CHECKER_AVAILABLE = False
    StaticConsistencyChecker = None
    ConsistencyCheckResult = None
    ComplexityChecker = None
    RedundancyChecker = None
    FactorQualityGate = None


__all__ = [
    'FactorRegulator',
    'StaticConsistencyChecker',
    'ConsistencyCheckResult',
    'ComplexityChecker',
    'RedundancyChecker',
    'FactorQualityGate',
    'CONSISTENCY_CHECKER_AVAILABLE',
]
