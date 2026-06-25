"""
nyambung ke factors/proposal.py karena membuat faktor perlu regulatornya
"""

import inspect
import math
import re
from functools import lru_cache
import pandas as pd
import numpy as np
from typing import Tuple, List, Dict, Any, Optional
from core.evaluation import Evaluator
from log import logger
from core.scenario import Scenario
from factors.coder.factor_ast import (
    match_alphazoo, count_free_args, count_unique_vars, count_all_nodes,
    calculate_symbol_length, count_base_features,
    parse_expression as parse_ast,
    FunctionNode, BinaryOpNode, ConditionalNode, UnaryOpNode, VarNode, NumberNode,
)
from factors.coder.expr_parser import parse_expression


# ── Validasi arity fungsi DSL ────────────────────────────────────────────
# Faktor bisa lolos parsing tapi tetap gagal saat backtest kalau memanggil
# fungsi dengan jumlah argumen yang salah, mis. `RANK($volume, 7)` padahal
# RANK cross-sectional hanya menerima 1 argumen (yang berperiode adalah
# TS_RANK). Map arity dibangun langsung dari signature asli di function_lib
# (runtime namespace, jadi redefinisi seperti MAX/MIN versi element-wise
# yang menang otomatis terbaca benar).

# Map cross-sectional → time-series untuk kesalahan paling sering: LLM memberi
# periode ke fungsi cross-sectional 1-argumen (mis. RANK(A, 7)). Dipakai untuk
# (a) hint pesan error dan (b) auto-repair deterministik (lihat
# auto_repair_function_arity). Hanya nama dengan padanan TS_(A, n) yang jelas;
# MAX/MIN sengaja TIDAK di sini karena overloaded (cross-sectional/pairwise/TS_).
_CS_TO_TS = {
    "RANK": "TS_RANK", "ZSCORE": "TS_ZSCORE", "MEAN": "TS_MEAN",
    "STD": "TS_STD", "MEDIAN": "TS_MEDIAN", "SKEW": "TS_SKEW",
    "KURT": "TS_KURT", "SUM": "TS_SUM",
}

# Override arity untuk fungsi yang sengaja variadic (*args) sehingga signature
# tidak memberi batas berguna. MAX/MIN menerima 1 (cross-sectional) s/d 3
# (element-wise) argumen.
_ARITY_OVERRIDES = {
    "MAX": (1, 3),
    "MIN": (1, 3),
}


@lru_cache(maxsize=1)
def _build_arity_map() -> Dict[str, Tuple[int, float]]:
    """Bangun {NAMA_FUNGSI: (min_args, max_args)} dari function_lib.

    max_args = math.inf bila fungsi punya *args. Hanya nama DSL (UPPERCASE,
    tidak diawali '_') yang diambil; helper internal & alias modul diabaikan.
    """
    from factors.coder import function_lib

    arity: Dict[str, Tuple[int, float]] = {}
    for name in dir(function_lib):
        if not re.fullmatch(r"[A-Z][A-Z0-9_]*", name):
            continue
        fn = getattr(function_lib, name)
        if not callable(fn):
            continue
        try:
            sig = inspect.signature(fn)  # follow_wrapped=True default → signature asli
        except (TypeError, ValueError):
            continue
        min_args = 0
        max_args = 0
        has_var = False
        for p in sig.parameters.values():
            if p.kind in (inspect.Parameter.VAR_POSITIONAL,):
                has_var = True
            elif p.kind in (inspect.Parameter.POSITIONAL_ONLY,
                            inspect.Parameter.POSITIONAL_OR_KEYWORD):
                max_args += 1
                if p.default is inspect.Parameter.empty:
                    min_args += 1
            # VAR_KEYWORD / KEYWORD_ONLY tidak bisa diisi posisional di DSL → diabaikan
        arity[name] = (min_args, math.inf if has_var else max_args)
    arity.update(_ARITY_OVERRIDES)
    return arity


def _iter_function_nodes(node):
    """Yield setiap FunctionNode dalam AST factor_ast secara rekursif."""
    if isinstance(node, FunctionNode):
        yield node
        for arg in node.args:
            yield from _iter_function_nodes(arg)
    elif isinstance(node, BinaryOpNode):
        yield from _iter_function_nodes(node.left)
        yield from _iter_function_nodes(node.right)
    elif isinstance(node, UnaryOpNode):
        yield from _iter_function_nodes(node.operand)
    elif isinstance(node, ConditionalNode):
        yield from _iter_function_nodes(node.condition)
        yield from _iter_function_nodes(node.true_expr)
        yield from _iter_function_nodes(node.false_expr)
    # VarNode / NumberNode: tidak ada anak


def validate_function_arity(expression: str) -> Tuple[bool, List[str]]:
    """Cek setiap pemanggilan fungsi pada expression melawan arity asli.

    Returns:
        (ok, errors). ok=True bila semua pemanggilan valid. errors berisi
        pesan ramah-LLM untuk tiap pelanggaran (fungsi tak dikenal / arity salah).
    """
    arity_map = _build_arity_map()
    try:
        tree = parse_ast(expression)
    except Exception as e:
        # Biarkan jalur is_parsable yang menangani error parse; di sini skip.
        logger.debug(f"validate_function_arity: parse skipped for {expression!r}: {e}")
        return True, []

    errors: List[str] = []
    for fn in _iter_function_nodes(tree):
        # FunctionNode.name bisa berupa VarNode (hasil parse action var) → ambil str-nya.
        name = fn.name.name if isinstance(fn.name, VarNode) else str(fn.name)
        n = len(fn.args)
        if name not in arity_map:
            errors.append(
                f"`{name}` is not a known function. Use only functions from the allowed list."
            )
            continue
        lo, hi = arity_map[name]
        if n < lo or n > hi:
            expected = f"{lo}" if lo == hi else (
                f"{lo}+" if hi == math.inf else f"{lo}-{hi}"
            )
            msg = f"`{name}` takes {expected} argument(s) but got {n} (`{fn}`)."
            # Hint cross-sectional → time-series bila kelebihan argumen.
            if name in _CS_TO_TS and n > hi:
                ts = _CS_TO_TS[name]
                msg += (
                    f" `{name}` is CROSS-SECTIONAL (1 arg, no period). "
                    f"For a rolling/windowed version use `{ts}(A, n)` instead."
                )
            errors.append(msg)
    return (len(errors) == 0), errors


def _node_name(fn: "FunctionNode") -> str:
    """Ambil nama fungsi (string) dari FunctionNode (name bisa berupa VarNode)."""
    return fn.name.name if isinstance(fn.name, VarNode) else str(fn.name)


def auto_repair_function_arity(expression: str) -> Tuple[Optional[str], List[str]]:
    """Coba perbaiki otomatis kesalahan arity cross-sectional→time-series.

    Hanya menangani kasus yang TIDAK AMBIGU: fungsi cross-sectional 1-argumen
    (RANK/ZSCORE/MEAN/STD/MEDIAN/SKEW/KURT/SUM) yang dipanggil dengan tepat 2
    argumen di mana argumen ke-2 adalah konstanta numerik (periode). Pada kasus
    itu intent LLM hampir pasti versi windowed → tambahkan prefiks TS_.

    MAX/MIN sengaja TIDAK diperbaiki (overloaded), begitu pula error arity lain
    (kurang argumen, 3+ argumen, argumen ke-2 bukan angka) → biar di-repair LLM.

    Returns:
        (repaired_expression | None, applied): expression baru bila ada perbaikan
        yang berhasil & tetap valid; None bila tak ada yang bisa diperbaiki aman.
        `applied` = daftar deskripsi perbaikan untuk logging.
    """
    try:
        tree = parse_ast(expression)
    except Exception:
        return None, []

    applied: List[str] = []

    def _rewrite(node) -> None:
        if isinstance(node, FunctionNode):
            name = _node_name(node)
            if (name in _CS_TO_TS
                    and len(node.args) == 2
                    and isinstance(node.args[1], NumberNode)):
                ts = _CS_TO_TS[name]
                applied.append(f"{name}(.., n) → {ts}(.., n)")
                # name bisa VarNode → set .name; selain itu ganti string langsung.
                if isinstance(node.name, VarNode):
                    node.name.name = ts
                else:
                    node.name = ts
            for arg in node.args:
                _rewrite(arg)
        elif isinstance(node, BinaryOpNode):
            _rewrite(node.left)
            _rewrite(node.right)
        elif isinstance(node, UnaryOpNode):
            _rewrite(node.operand)
        elif isinstance(node, ConditionalNode):
            _rewrite(node.condition)
            _rewrite(node.true_expr)
            _rewrite(node.false_expr)

    _rewrite(tree)
    if not applied:
        return None, []

    repaired = str(tree)

    # str(AST) merender angka sebagai float (7 → "7.0"). Periode windowing harus
    # int (rolling(7.0) → ValueError: window must be an integer), jadi normalkan
    # float bernilai bulat kembali ke int. Aman untuk DSL: angka non-bulat
    # (0.2, 1e-8) tidak tersentuh.
    repaired = re.sub(r"\b(\d+)\.0\b", r"\1", repaired)

    # Pastikan hasil repair benar-benar valid: arity bersih DAN tetap bisa
    # diparse oleh runtime parser. Kalau round-trip merusak sesuatu, batalkan.
    ok, _ = validate_function_arity(repaired)
    if not ok:
        return None, []
    try:
        parse_expression(repaired)
    except Exception:
        return None, []

    return repaired, applied


# ── Validasi variabel & degenerate-args (gate deterministik #1-#2) ────────────
# Faktor bisa lolos parser + arity tapi tetap rusak saat eksekusi karena:
#   (1) leaf $var yang TAK ADA di data (mis. $return_1d) → KeyError/NaN; dan
#   (2) fungsi 2-deret dengan dua argumen IDENTIK (mis. REGRESI(x, x)) → residual
#       ≈ 0 / korelasi ≡ 1 (faktor mati). Keduanya sering muncul saat model 4B
#       dipaksa memakai operator regresi/teknikal yang tak ia pahami.

# Kolom runtime yang sah (daily_pv.h5: OHLCV + $return turunan). Leaf $xxx di luar
# set ini = halusinasi → tolak deterministik (parser/arity tak menangkapnya).
_KNOWN_VARS = {"open", "high", "low", "close", "volume", "return"}


def validate_known_variables(expression: str) -> Tuple[bool, List[str]]:
    """Cek tiap leaf `$var` terhadap kolom data yang tersedia. Deterministik (regex,
    tak perlu AST). Returns (ok, errors) ramah-LLM."""
    errors: List[str] = []
    seen = set()
    for m in re.finditer(r"\$([A-Za-z_][A-Za-z0-9_]*)", expression or ""):
        v = m.group(1)
        if v not in _KNOWN_VARS and v not in seen:
            seen.add(v)
            errors.append(
                f"`${v}` is not a known variable. Use only: "
                f"$open $high $low $close $volume $return."
            )
    return (len(errors) == 0), errors


# Fungsi 2-deret yang degenerate bila kedua deret identik.
_PAIRWISE_SERIES_FUNCS = {"REGBETA", "REGRESI", "TS_CORR", "TS_COVARIANCE"}


def validate_no_degenerate_args(expression: str) -> Tuple[bool, List[str]]:
    """Tolak REGBETA/REGRESI/TS_CORR/TS_COVARIANCE dengan dua deret IDENTIK
    (arg0≡arg1) → faktor mati (residual≈0 / corr≡1). Pakai AST agar tahan nesting."""
    try:
        tree = parse_ast(expression)
    except Exception:
        return True, []  # serahkan ke is_parsable
    errors: List[str] = []
    for fn in _iter_function_nodes(tree):
        name = _node_name(fn)
        if name in _PAIRWISE_SERIES_FUNCS and len(fn.args) >= 2:
            if str(fn.args[0]) == str(fn.args[1]):
                errors.append(
                    f"`{name}` got two identical series (`{fn.args[0]}`) → "
                    f"degenerate (residual≈0 / corr≡1). Use two DIFFERENT series."
                )
    return (len(errors) == 0), errors


# TODO harusnya bisa dioptimalkan dalam penentuan parameter dianggap "berlebihan" atau tidak
# [terjawab — skripsi Bab 2 §Kerangka QuantaAlpha (rumus C(f), S(f)) & Bab 4 §Kendali Mutu
#   Faktor (gate SL/PC/ER + redundansi + kesesuaian kode vs formalisme paper)].
class FactorRegulator(Evaluator):
    """
    FactorRegulator class to evaluate expressions for duplication and manage the factor zoo database.
    This class provides functionality to detect duplicated subtrees in factor expressions
    and ensure new factors maintain appropriate originality.
    """

    def __init__(self, factor_zoo_path: str = None, duplication_threshold: int = 8,
                 symbol_length_threshold: int = 300, base_features_threshold: int = 6):
        """
        Initialize the FactorRegulator with a reference to the factor zoo.

        Args:
            factor_zoo_path (str): Path to the CSV file containing the factor zoo database.
            duplication_threshold (int): Threshold for duplication detection.
            symbol_length_threshold (int): Maximum allowed symbol length (SL) for expressions.
            base_features_threshold (int): Maximum allowed number of unique base features (ER).
        """
        super().__init__(None)
        self.factor_zoo_path = factor_zoo_path

        #* load CSV berisi semua faktor yang sudah diketahui ("alpha zoo")
        #* dipakai untuk cek apakah faktor baru terlalu mirip
        if factor_zoo_path:
            self.alphazoo = pd.read_csv(factor_zoo_path, index_col=None)
        else:
            self.alphazoo = pd.DataFrame()

        #* threshold: kalau subtree yang sama > 8 nodes => duplikat
        self.duplication_threshold = duplication_threshold

        #* max panjang simbol ekspresi(300) -> terlalu panjang => terlalu kompleks
        self.symbol_length_threshold = symbol_length_threshold

        #* max jumlah fitur dasar unik(6) -> terlalu banyak => overfitted
        self.base_features_threshold = base_features_threshold

        #* buffer faktor baru yang ditambahkan dalam run ini
        self.new_factors = []



    def is_parsable(self, expression: str) -> bool:
        """
        Checks if an expression can be successfully parsed.

        Args:
            expression (str): The factor expression to check.

        Returns:
            bool: True if the expression can be parsed, False otherwise.
        """
        try:
            parse_expression(expression)    #* AST parser untuk ekspresi faktor
            return True
        except Exception as e:
            logger.error(f"Failed to parse expression: {expression}. Error: {str(e)}")
            return False

    def validate_signatures(self, expression: str) -> Tuple[bool, List[str]]:
        """Validasi arity tiap pemanggilan fungsi DSL terhadap function_lib.

        Menangkap kesalahan yang lolos parsing tapi gagal saat backtest,
        mis. `RANK($volume, 7)` (RANK cross-sectional hanya 1 argumen).

        Returns:
            (ok, errors): ok=True bila semua valid; errors = pesan ramah-LLM.
        """
        return validate_function_arity(expression)

    def auto_repair_signatures(self, expression: str) -> Tuple[Optional[str], List[str]]:
        """Coba perbaiki otomatis arity cross-sectional→time-series yang tak ambigu.

        Returns:
            (repaired | None, applied). None bila tak ada perbaikan aman.
        """
        return auto_repair_function_arity(expression)

    def evaluate(self, expression: str) -> Tuple[int, str, Optional[str]]:
        """
        Evaluates an expression for duplication with existing factors in the factor zoo.

        Args:
            expression (str): The factor expression to evaluate.

        Returns:
            Tuple containing:
                - duplicated_subtree_size (int): Size of the duplicated subtree
                - duplicated_subtree (str): The duplicated subtree expression
                - matched_alpha (str or None): Name of the matched alpha if available
        """
        try:
            # Check for duplication
            duplicated_subtree_size, duplicated_subtree, matched_alpha = match_alphazoo(
                expression, self.alphazoo
            )

            num_free_args = count_free_args(expression)
            num_unique_vars = count_unique_vars(expression)
            num_all_nodes = count_all_nodes(expression)
            symbol_length = calculate_symbol_length(expression)
            num_base_features = count_base_features(expression)

            logger.info(f"""
                        Evaluated expr: {expression}
                        Duplicated Size: {duplicated_subtree_size}
                        Duplicated Subtree: {duplicated_subtree}
                        # Free Args: {num_free_args}
                        # Unique Vars: {num_unique_vars}
                        Symbol Length (SL): {symbol_length}
                        # Base Features (ER): {num_base_features}
                        """)

            eval_dict = {
                "expr": expression,
                "duplicated_subtree_size": duplicated_subtree_size,
                "duplicated_subtree": duplicated_subtree,
                "matched_alpha": matched_alpha,
                "num_free_args": num_free_args,
                "num_unique_vars": num_unique_vars,
                "num_all_nodes": num_all_nodes,
                "symbol_length": symbol_length,
                "num_base_features": num_base_features
                }

            return True, eval_dict

        except Exception as e:
            logger.error(f"Failed to evaluate expression: {expression}. Error: {str(e)}")
            return False, None


    def is_expression_acceptable(self, eval_dict) -> bool:
        """
        Determines if an expression is acceptable based on the duplication threshold,
        the ratio of num_free_args and num_unique_vars to the total number of nodes,
        symbol length (SL), and base features count (ER).

        This implements the complexity regularization R_g(f, h) from the paper:
        R_g(f, h) = α₁·SL(f) + α₂·PC(f) + α₃·ER(f, h)

        Args:
            eval_dict (dict): Dictionary containing evaluation results of the expression.

        Returns:
            bool: True if the expression is acceptable, False otherwise.
        """
        # Condition 1: Check if the duplicated subtree size is within the threshold
        cond1 = eval_dict['duplicated_subtree_size'] <= self.duplication_threshold

        # Get the number of free arguments, unique variables, and total nodes
        num_free_args = eval_dict['num_free_args']
        num_unique_vars = eval_dict['num_unique_vars']
        num_all_nodes = eval_dict['num_all_nodes']
        symbol_length = eval_dict.get('symbol_length', 0)
        num_base_features = eval_dict.get('num_base_features', 0)

        # Avoid division by zero and invalid ratios
        if num_all_nodes == 0:
            logger.warning(f"Expression has no nodes: {eval_dict['expr']}")
            return False

        # Calculate ratios
        free_args_ratio = float(num_free_args) / float(num_all_nodes)
        unique_vars_ratio = float(num_unique_vars) / float(num_all_nodes)

        # Ensure ratios are within valid range (0 <= ratio < 1)
        if free_args_ratio >= 1 or unique_vars_ratio >= 1:
            logger.warning(f"Invalid ratio detected: free_args_ratio={free_args_ratio}, unique_vars_ratio={unique_vars_ratio}")
            return False

        # Condition 2: Ensure the ratio of num_free_args to total nodes is not too high using -log(1 - ratio)
        # -log(1 - x) increases as x increases, so we set a threshold (e.g., -log(1 - 0.5) ≈ 0.693)
        # This ensures the ratio is not too high (e.g., x < 0.5)
        cond2 = -np.log(1 - free_args_ratio) < 0.693  # Threshold for x < 0.5

        # Condition 3: Ensure the ratio of num_unique_vars to total nodes is not too high using -log(1 - ratio)
        cond3 = -np.log(1 - unique_vars_ratio) < 0.693  # Threshold for x < 0.5

        # Condition 4: Check symbol length (SL) - expression should not be too long
        cond4 = symbol_length <= self.symbol_length_threshold

        # Condition 5: Check base features count (ER) - should not use too many raw features
        # Using log(1 + |F_f|) penalty as in the paper
        cond5 = num_base_features <= self.base_features_threshold

        # The expression is acceptable if all conditions are met
        return cond1 and cond2 and cond3 and cond4 and cond5


    def add_factor(self, factor_name: str, factor_expression: str) -> bool:
        """
        Adds a new factor to the in-memory factor zoo if it passes the duplication check.

        Args:
            factor_name (str): Name of the new factor.
            factor_expression (str): Expression of the new factor.

        Returns:
            bool: True if the factor was added, False otherwise.
        """
        new_factor = pd.DataFrame({
                'factor_name': factor_name,
                'factor_expression': factor_expression
                })

        self.alphazoo = pd.concat([self.alphazoo, new_factor])
        self.new_factors.append((factor_name, factor_expression))
        logger.info(f"Added new factor: {factor_name} with expression: {factor_expression}")

    def save_factor_zoo(self, output_path: Optional[str] = None) -> None:
        """
        Saves the updated factor zoo to a CSV file.

        Args:
            output_path (str, optional): Path to save the updated factor zoo.
                                         If None, updates the original file.
        """
        save_path = output_path if output_path else self.factor_zoo_path
        self.alphazoo.to_csv(save_path, index=False)
        logger.info(f"Saved updated factor zoo to {save_path}")

    def get_new_factors(self) -> List[Tuple[str, str]]:
        """
        Returns the list of new factors added during this session.

        Returns:
            List[Tuple[str, str]]: List of (factor_name, factor_expression) tuples.
        """
        return self.new_factors
