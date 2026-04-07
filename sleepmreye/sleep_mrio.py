import logging

import numpy as np
import re
import pandas as pd
from nilearn import signal
from nilearn.image import resample_to_img
from nilearn.masking import apply_mask
from scipy.stats import zscore

from mreyemove.constants import COMMON_CONFOUNDS, EYE_SIGNAL_CONFOUNDS
from mreyemove.data.fmri_mixin import TissueType
from mreyemove.data.mrio import MRIO
from mreyemove.preprocessing.eye_move import construct_dissimilarity_signal

logger = logging.getLogger(__name__)


class SleepMRIO(MRIO):
    DEFAULT_TASK = "sleep"
    VALID_EVENT_STAGES = ["W", "1", "2"]
    MASK_COLS = ["W", "1", "2", "S"]

    def __init__(self, load_eog: bool = True, **kwargs):
        self.load_eog = load_eog
        super().__init__(**kwargs)

    def run_inclusion_condition(self, _run_events: pd.DataFrame) -> bool:
        return (_run_events["state"].isin(self.VALID_EVENT_STAGES)).all()

    def _load_events_impl(
        self,
        subjects: list[str],
        exclude_subjects: list[str] | None,
        load_signal: bool = True,
    ):
        events_epoch = self._load_sleep_epochs(
            subjects=subjects, exclude_subjects=exclude_subjects
        )
        events_epoch = events_epoch[events_epoch["task"] == "sleep"]
        events_tr = self.build_tr_level_events(
            epoch_events=events_epoch,
            n_scans=self.n_scans(subject="03"),
            load_signal=load_signal,
        )
        return events_tr

    def _load_sleep_epochs(
        self, subjects: list[str] | None, exclude_subjects: list[str] | None
    ) -> pd.DataFrame:
        """
        Load sourcedata epoch files and return a long DataFrame with one row per 30s epoch.

        Adds:
        - subject (string)
        - TR (int): TR index at which the epoch starts (0-based, ceil of time/TR)
        - task, run (parsed from 'session' column)
        - dummy columns: awake, nrem1, nrem2, nrem3 (booleans)
        """
        event_dir = self.bids_root / "sourcedata"

        all_events: list[pd.DataFrame] = []

        for subject_event_path in event_dir.rglob("sub-*.tsv"):
            m = re.search(r"sub-(\d+)", subject_event_path.name)
            if m is None:
                continue
            subject = m.group(1)

            if (subjects is not None and subject not in subjects) or (
                exclude_subjects is not None and subject in exclude_subjects
            ):
                logger.info("Skipping subject %s", subject)
                continue

            df = pd.read_csv(subject_event_path, sep="\t")

            # Rename to something less horrible
            df = df.rename(columns={"30-sec_epoch_sleep_stage": "state"})

            # Attach subject
            df["subject"] = subject

            # TRs don't fit perfectly: ~14.29 TR per epoch.
            # We ceil so that once we cross into the 15th TR, it's the next epoch.
            df["TR"] = np.ceil(
                df["epoch_start_time_sec"].astype(float) / self.tr
            ).astype(int)

            # Parse task and run from "session" column
            # (adapt regex if your filenames differ)
            df[["task", "run"]] = df["session"].str.extract(r"task-(.*?)_.*run-(\w+)")
            df["run"] = pd.to_numeric(df["run"])

            # Drop raw session col
            df = df.drop(columns=["session"])

            # Keep only sessions that actually contain sleep
            if (df["task"] == "sleep").any():
                all_events.append(df)
            else:
                logger.info(f"Removing subject {subject} without sleep sessions")

        if not all_events:
            raise RuntimeError("No sleep epoch files found under sourcedata.")

        events = pd.concat(all_events, ignore_index=True)

        # Clean / collapse sleep stage labels
        events["state"] = events["state"].str.replace(
            r".*\(uncertain\)", "NS", regex=True
        )
        events["state"] = events["state"].str.replace(
            r".*[Uu]nscorable.*", "A", regex=True
        )

        # Convenience one-hot-ish columns
        events["W"] = events["state"] == "W"
        events["1"] = events["state"] == "1"
        events["2"] = events["state"] == "2"
        events["3"] = events["state"] == "3"
        events["S"] = events["state"].isin(["1", "2"])

        return events

    def _expand_epoch_to_trs(
        self,
        run_events: pd.DataFrame,
        n_scans: int,
    ) -> pd.DataFrame:
        """
        Expand epoch-level state labels to TR-level labels for a single run.

        Parameters
        ----------
        run_events : DataFrame
            Rows = epochs for a single (subject, task, run),
            must contain:
                - "TR" (int): epoch start TR index
                - one boolean column per state in `states`
        n_scans : int
            Total number of TRs in this run (before trimming).

        Returns
        -------
        pd.DataFrame
        """
        df = run_events.copy()

        df = df.sort_values("TR").reset_index(drop=True)

        TR_starts = df["TR"].clip(lower=0)
        next_starts = TR_starts.shift(-1, fill_value=n_scans)

        df["repeat"] = (next_starts - TR_starts).clip(lower=0).astype(int)

        df_repeated = df.loc[df.index.repeat(df["repeat"])].reset_index(drop=True)

        df_repeated = df_repeated.drop(
            columns=["epoch_start_time_sec", "repeat"], errors="ignore"
        )

        df_repeated["TR"] = df_repeated.index

        return df_repeated

    def build_tr_level_events(
        self,
        epoch_events: pd.DataFrame,
        n_scans: int,
        load_signal: bool = True,
    ) -> pd.DataFrame:
        """
        Expand epoch-level sleep events to TR-level labels for all runs.

        Parameters
        ----------
        epoch_events : DataFrame
            Output of `load_sleep_epochs`, or equivalent.
            Must contain columns:
                - 'subject' (str)
                - 'task' (str)
                - 'run' (int)
                - 'TR' (int) epoch start TR
                - state columns in `states` (bool)
        states : iterable of str
            State columns to expand.

        Returns
        -------
        DataFrame
            One row per TR, with columns:
                - subject, task, run
                - tr (int): TR index in original run
                - state label (str, may be empty/None if no state)
                - one column per state in `states` as float mask (0/1)
        """
        tr_dfs: list[pd.DataFrame] = []

        grouped = epoch_events.groupby(["subject", "task", "run"], sort=False)

        eog_data = self.load_eog_df()

        for (subject, task, run), run_events in grouped:
            # Expand into per-TR masks
            run_tr_df = self._expand_epoch_to_trs(
                run_events=run_events,
                n_scans=n_scans,
            )

            if load_signal:
                run_tr_df = self._add_mreyemove_to_events(
                    df=run_tr_df,
                    subject=subject,
                    run=run,
                    mask_cols=self.MASK_COLS,
                )

            run_tr_df = run_tr_df.reset_index(drop=True)
            framewise_displacement = (
                self.load_confounds(
                    subject=subject, run=run, confound_names=["framewise_displacement"]
                )
                .copy()
                .reset_index(drop=True)
            )
            run_tr_df["conv_framewise_displacement"] = framewise_displacement
            if self.load_eog:
                eog_run_df = eog_data[
                    (eog_data["subject"] == subject) & (eog_data["run"] == run)
                ].reset_index(drop=True)
                eog_run = eog_run_df["signal"].to_numpy()
                eog_run = eog_run[self.tr_slice()]

                if len(eog_run) == len(run_tr_df):
                    confounds = self.load_confounds(
                        subject=subject, run=run, confound_names=COMMON_CONFOUNDS
                    )
                    eog_run_2d = eog_run.reshape(-1, 1)
                    eog_run = signal.clean(
                        eog_run_2d,
                        confounds=confounds,
                        standardize=False,
                        detrend=False,
                        filter=False,
                        t_r=None,
                    ).ravel()
                    eog_run = np.maximum(eog_run, 0)
                else:
                    logger.warning("EOG data not full for %s, %s (%d/%d). Skipping", subject, run, len(eog_run), len(run_tr_df))
                    eog_run = [np.nan for _ in range(len(run_tr_df))]

                if len(eog_run) != len(run_tr_df):
                    eog_run = [np.nan for _ in range(len(run_tr_df))]
                run_tr_df["eog"] = eog_run
                self.split_regressor(
                    run_df=run_tr_df,
                    regressor_name="eog",
                    state_cols=self.MASK_COLS,
                    fill_value=0.0,
                )

            tr_dfs.append(run_tr_df)
        tr_events = pd.concat(tr_dfs, ignore_index=True)

        tr_events.to_csv("/Users/zach/2025_RA/matthias/X/events.csv")
        return tr_events

    def load_eog_df(self, rectify: bool = True, aggr: str = "mean") -> pd.DataFrame:
        dir = self.mreyemove_dir / "group" / "eeg"
        fname = ""
        if rectify:
            fname = "rectify_"
        fname = f"{fname}{aggr}_eog_reg.p"

        df = pd.read_pickle(dir / fname)
        return df
