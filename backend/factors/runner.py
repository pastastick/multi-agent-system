import pickle
import sys
from pathlib import Path
from typing import List
import os
import pandas as pd
from pandarallel import pandarallel

from core.conf import RD_AGENT_SETTINGS
from core.utils import cache_with_pickle, multiprocessing_wrapper
from factors.coder.config import FACTOR_COSTEER_SETTINGS

pandarallel.initialize(verbose=1)

from components.runner import CachedRunner
from core.exception import FactorEmptyError
from log import logger
from factors.experiment import QlibFactorExperiment

DIRNAME = Path(__file__).absolute().resolve().parent
DIRNAME_local = Path.cwd()

# class QlibFactorExpWorkspace:

#     def prepare():
#         # create a folder;
#         # copy template
#         # place data inside the folder `combined_factors`
#         #
#     def execute():
#         de = DockerEnv()
#         de.run(local_path=self.ws_path, entry="qrun conf.yaml")

# TODO: supporting multiprocessing and keep previous results


class QlibFactorRunner(CachedRunner[QlibFactorExperiment]):
    """
    Docker run
    Everything in a folder
    - config.yaml
    - price-volume data dumper
    - `data.py` + Adaptor to Factor implementation
    - results in `mlflow`
    """

    def _local_execute(self, workspace_path: Path, config_name: str) -> tuple:
        """
        Jalankan qlib backtest secara lokal tanpa conda/docker.
        Dipanggil oleh develop() saat use_local=True.
        """
        import subprocess as _sp
        env = os.environ.copy()
        project_root = Path(__file__).resolve().parent.parent
        env["PYTHONPATH"] = str(project_root) + os.pathsep + env.get("PYTHONPATH", "")

        # Step 1: qrun
        try:
            r = _sp.run(
                ["qrun", config_name],
                cwd=str(workspace_path),
                env=env,
                capture_output=True,
                text=True,
                timeout=3600,
            )
            if r.returncode != 0:
                logger.error(f"[LocalExec] qrun failed:\n{r.stderr[-2000:]}")
                return None, r.stderr
            logger.info(f"[LocalExec] qrun OK: {r.stdout[-500:]}")
        except Exception as e:
            logger.error(f"[LocalExec] qrun exception: {e}")
            return None, str(e)

        # Step 2: read_exp_res.py
        read_script = workspace_path / "read_exp_res.py"
        if read_script.exists():
            try:
                r2 = _sp.run(
                    [sys.executable, "read_exp_res.py"],
                    cwd=str(workspace_path),
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=120,
                )
                if r2.returncode != 0:
                    logger.warning(f"[LocalExec] read_exp_res.py failed:\n{r2.stderr[-1000:]}")
            except Exception as e:
                logger.warning(f"[LocalExec] read_exp_res.py exception: {e}")

        # Step 3: baca hasil
        qlib_res = workspace_path / "qlib_res.csv"
        if qlib_res.exists():
            df = pd.read_csv(qlib_res, index_col=0).iloc[:, 0]
            return df, "local_execute_ok"
        logger.warning("[LocalExec] qlib_res.csv tidak ditemukan setelah qrun.")
        return None, "qlib_res.csv not found"

    #* Hitung IC antara setiap kolom SOTA factor dan new factor
    def calculate_information_coefficient(
        self, concat_feature: pd.DataFrame, SOTA_feature_column_size: int, new_feature_columns_size: int
    ) -> pd.DataFrame:
        
        # buat series kosong
        res = pd.Series(index=range(SOTA_feature_column_size * new_feature_columns_size))
        
        #* hitung pearson correlation antara kolom SOTA dan new -> simpan di res dengan index linear
        for col1 in range(SOTA_feature_column_size):
            for col2 in range(SOTA_feature_column_size, SOTA_feature_column_size + new_feature_columns_size):
                res.loc[col1 * new_feature_columns_size + col2 - SOTA_feature_column_size] = concat_feature.iloc[
                    :, col1
                ].corr(concat_feature.iloc[:, col2])
        return res

    #* buang new faktor yang terlalu mirip faktor SOTA
    def deduplicate_new_factors(self, SOTA_feature: pd.DataFrame, new_feature: pd.DataFrame) -> pd.DataFrame:
        # calculate the IC between each column of SOTA_feature and new_feature
        # if the IC is larger than a threshold, remove the new_feature column
        # return the new_feature

        # gabung semua kolom
        concat_feature = pd.concat([SOTA_feature, new_feature], axis=1)
        
        #* per tanggal: hitung IC semua pasangan (SOTA,new)
        IC_max = (
            concat_feature.groupby("datetime")
            .parallel_apply(
                lambda x: self.calculate_information_coefficient(x, SOTA_feature.shape[1], new_feature.shape[1])
            )
            .mean()
        )
        
        # reshape index 
        IC_max.index = pd.MultiIndex.from_product([range(SOTA_feature.shape[1]), range(new_feature.shape[1])])
        
        #* ambil max IC => untuk setiap faktor baru, berapa IC tertingginya dengan SEMUA faktor SOTA
        IC_max = IC_max.unstack().max(axis=0)
        
        # new faktor yang kurang dari threshold
        return new_feature.iloc[:, IC_max[IC_max < 0.99].index]

    
    # ── HYBRID: standalone per-factor RankIC (reward `L` paper) ───────────────
    # Pelengkap LightGBM combined (metrik portofolio). Per-factor RankIC = sinyal
    # fitness/seleksi evolution + evaluatif feedback. Dihitung dari sumber yang
    # SAMA dengan factor (daily_pv.h5) → index otomatis align, tanpa qlib.init.
    @staticmethod
    def _oos_window() -> tuple:
        """(start, end) Timestamp segmen TEST dari template config combined-factors,
        agar RankIC standalone out-of-sample & sebanding dgn RankIC LightGBM.
        Fallback (None, None) bila gagal parse → IC dihitung full-range."""
        try:
            import yaml as _yaml
            cfg = Path(__file__).resolve().parent / "factor_template" / "conf_combined_factors.yaml"
            seg = _yaml.safe_load(cfg.read_text())["task"]["dataset"]["kwargs"]["segments"]["test"]
            return pd.Timestamp(seg[0]), pd.Timestamp(seg[1])
        except Exception as e:  # noqa: BLE001
            logger.warning(f"[FactorIC] gagal baca segmen test dari config: {e}; pakai full-range")
            return None, None

    def _factor_label(self) -> "pd.Series | None":
        """Label = Ref($close,-2)/Ref($close,-1)-1 (= LABEL0 config), dihitung dari
        daily_pv.h5 (sumber yang sama dgn factor). Series ber-index (datetime,
        instrument). None bila sumber/kolom tak tersedia."""
        data_source = Path(FACTOR_COSTEER_SETTINGS.data_folder)
        if not data_source.is_absolute():
            data_source = Path(__file__).resolve().parent.parent / FACTOR_COSTEER_SETTINGS.data_folder
        pv_path = data_source / "daily_pv.h5"
        if not pv_path.exists():
            return None
        pv = pd.read_hdf(pv_path, key="data")
        if "$close" not in getattr(pv, "columns", []):
            return None
        close = pv["$close"].sort_index()
        g = close.groupby(level="instrument")          # cegah bocor antar simbol
        label = g.shift(-2) / g.shift(-1) - 1.0        # = Ref($close,-2)/Ref($close,-1)-1
        label.name = "label"
        return label

    def _compute_factor_ic(self, new_factors: pd.DataFrame) -> dict:
        """RankIC cross-sectional per factor pada segmen TEST (OOS):
        mean_t spearman(factor_t, label_t). Return {factor_name: rank_ic|None}."""
        label = self._factor_label()
        if label is None:
            logger.warning("[FactorIC] daily_pv.h5/$close tak tersedia → lewati per-factor RankIC")
            return {}
        # samakan urutan level index dgn new_factors (alignment by tuple)
        names = list(new_factors.index.names)
        if set(label.index.names) == set(names) and list(label.index.names) != names:
            label = label.reorder_levels(names)
        start, end = self._oos_window()
        ic_out: dict = {}
        for col in dict.fromkeys(new_factors.columns):   # unik, jaga urutan
            s = new_factors[col]
            if isinstance(s, pd.DataFrame):              # nama kolom dobel → ambil pertama
                s = s.iloc[:, 0]
            df = pd.DataFrame({"f": s, "y": label}).dropna()
            if start is not None and not df.empty:
                dts = df.index.get_level_values("datetime")
                df = df[(dts >= start) & (dts <= end)]
            if df.empty:
                ic_out[str(col)] = None
                continue
            per_day = df.groupby(level="datetime").apply(
                lambda x: x["f"].corr(x["y"], method="spearman") if len(x) > 2 else float("nan")
            )
            ic = per_day.mean()
            ic_out[str(col)] = float(ic) if pd.notna(ic) else None
        return ic_out

    #* dipanggil di AlphaAgentLoop -> factor_backtest
    # gabung semua faktor value
    # jalankan backtest (Qlib)
    # simpan result di exp.result
    #* kalau experiment yang persis sama pernah dijalankan, load dari pickle cache
    @cache_with_pickle(CachedRunner.get_cache_key, CachedRunner.assign_cached_result)
    def develop(self, exp: QlibFactorExperiment, use_local: bool = True) -> QlibFactorExperiment:
        """Process new factors and run backtest. Each round uses ONLY the new factors
        so metrics are strictly comparable across iterations (no SOTA dilution)."""

        #* Process the new factors data
        try:
            new_factors = self.process_factor_data(exp)
        except FactorEmptyError as e:
            logger.error(f"Failed to process new factors: {e}")

            #* Try manual factor execution
            logger.info("Attempting to manually execute factors...")
            for ws in exp.sub_workspace_list:
                if not (ws.workspace_path / "result.h5").exists():
                    try:
                        data_source = Path(FACTOR_COSTEER_SETTINGS.data_folder).absolute()
                        if not data_source.is_absolute():
                            data_source = Path(__file__).resolve().parent.parent / FACTOR_COSTEER_SETTINGS.data_folder
                        daily_pv_link = ws.workspace_path / "daily_pv.h5"
                        if not daily_pv_link.exists() and (data_source / "daily_pv.h5").exists():
                            os.symlink(str(data_source / "daily_pv.h5"), str(daily_pv_link))
                        import subprocess
                        env = os.environ.copy()
                        project_root = Path(__file__).resolve().parent.parent
                        env['PYTHONPATH'] = str(project_root) + os.pathsep + env.get('PYTHONPATH', '')
                        subprocess.check_output(
                            [sys.executable, str(ws.workspace_path / 'factor.py')],
                            cwd=str(ws.workspace_path),
                            stderr=subprocess.STDOUT,
                            env=env,
                            timeout=1200,
                        )
                    except Exception as exec_e:
                        logger.warning(f"Failed to manually execute factor {ws.workspace_path}: {exec_e}")

            try:
                new_factors = self.process_factor_data(exp)
            except FactorEmptyError:
                raise FactorEmptyError("No valid factor data found to merge after manual execution attempt.")

        if new_factors.empty:
            raise FactorEmptyError("No valid factor data found to merge.")

        # ── per-factor RankIC (OOS, standalone) ──────────────────────────────
        # Hitung sebelum di-nest ke MultiIndex "feature" (nama kolom masih = factor name).
        try:
            exp.factor_ic = self._compute_factor_ic(new_factors)
            logger.info(f"Per-factor RankIC (OOS): {exp.factor_ic}")
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Per-factor RankIC computation failed: {e}")
            exp.factor_ic = {}

        if len(new_factors.columns) >= 2:
            pd.set_option('display.width', 1000)
            logger.info(f"Factor correlation: \n\n{new_factors.corr()}\n")

        # Sort, deduplicate, and nest under 'feature' for Qlib compatibility
        combined_factors = new_factors.sort_index()
        combined_factors = combined_factors.loc[:, ~combined_factors.columns.duplicated(keep="last")]
        combined_factors.columns = pd.MultiIndex.from_product([["feature"], combined_factors.columns])

        logger.info(f"Factor values this round: \n\n{combined_factors.tail()}\n\n")

        parquet_path = exp.experiment_workspace.workspace_path / "combined_factors_df.parquet"
        combined_factors.to_parquet(parquet_path, engine="pyarrow")
        logger.info(f"Saved combined factors to {parquet_path}")

        # Always use conf_combined_factors.yaml (new-factors-only mode)
        config_name = "conf_combined_factors.yaml"
        logger.info(f"Execute factor backtest (Use {'Local' if use_local else 'Docker container'}): {config_name}")
        
        # Ensure workspace and config are ready (execute() does not call before_execute()).
        exp.experiment_workspace.before_execute()

        # Gunakan local execution (subprocess langsung) agar tidak butuh conda.
        # Rdagent workspace.execute() default pakai conda yang tidak tersedia.
        if use_local:
            result_tuple = self._local_execute(
                workspace_path=exp.experiment_workspace.workspace_path,
                config_name=config_name,
            )
        else:
            # execute() returns (result_df, execute_qlib_log) or (None, execute_qlib_log)
            result_tuple = exp.experiment_workspace.execute(
                qlib_config_name=config_name,
                run_env={}
            )
        
        # Unpack tuple; take first element (DataFrame)
        result = result_tuple[0] if isinstance(result_tuple, tuple) else result_tuple
        
        if result is not None:
            logger.info(f"Backtesting results: \n{result.iloc[2:] if hasattr(result, 'iloc') else result}")
        else:
            logger.warning("Backtesting result is None. Check the execution logs above for errors.")
            if isinstance(result_tuple, tuple) and len(result_tuple) > 1:
                logger.info(f"Execution log: {result_tuple[1][:500]}...")
        
        #* SIMPAN HASIL KE EXPERIMENT — ini yang nanti dibaca di feedback step
        exp.result = result

        return exp

    def process_factor_data(self, exp_or_list: List[QlibFactorExperiment] | QlibFactorExperiment) -> pd.DataFrame:
        """
        Process and combine factor data from experiment implementations.

        Args:
            exp (ASpecificExp): The experiment containing factor data.

        Returns:
            pd.DataFrame: Combined factor data without NaN values.
        """
        if isinstance(exp_or_list, QlibFactorExperiment):
            exp_or_list = [exp_or_list]
        factor_dfs = []

        # Collect all exp's dataframes
        for exp in exp_or_list:
            # Iterate over sub-implementations and execute them to get each factor data
            message_and_df_list = multiprocessing_wrapper(
                [(implementation.execute, ("All",)) for implementation in exp.sub_workspace_list],
                n=RD_AGENT_SETTINGS.multi_proc_n,
            )
            
            for idx, (message, df) in enumerate(message_and_df_list):
                # Check if factor generation was successful
                if df is not None and "datetime" in df.index.names:
                    # Convert Series to DataFrame if needed
                    if isinstance(df, pd.Series):
                        # Get factor name from the corresponding workspace (order should match)
                        if idx < len(exp.sub_workspace_list):
                            factor_name = getattr(exp.sub_workspace_list[idx].target_task, 'factor_name', None)
                            if factor_name:
                                df = df.to_frame(name=factor_name)
                            else:
                                df = df.to_frame(name=df.name if df.name else f'factor_{idx}')
                        else:
                            df = df.to_frame(name=df.name if df.name else f'factor_{idx}')
                    
                    time_diff = df.index.get_level_values("datetime").to_series().diff().dropna().unique()
                    if pd.Timedelta(minutes=1) not in time_diff:       # filter hanya terima daily
                        factor_dfs.append(df)

        # Combine all successful factor data
        if factor_dfs:
            return pd.concat(factor_dfs, axis=1)
        else:
            raise FactorEmptyError("No valid factor data found to merge.")
