import argparse
import logging
import os
from functools import partial
from pathlib import Path
from typing import cast

from joblib import Parallel, delayed
import pandas as pd
from sklearn import clone

from nilearn.connectome import ConnectivityMeasure
from nilearn.maskers import NiftiLabelsMasker
from nilearn import datasets

import numpy as np

from threadpoolctl import threadpool_limits

from .sleep_mrio import SleepMRIO
from mreyemove import enable_logging, _ensure_worker_logging, ContextAdapter
from mreyemove.analysis.glm.config import GLMConfig
from mreyemove.cli.common_args import add_io_args, add_logger_args, build_mrio_from_args
from mreyemove.constants import COMMON_CONFOUNDS
from mreyemove.data.mrio import MRIO
from mreyemove.preprocessing.eye_move import convolve_signal

logger = logging.getLogger(__name__)


def compute_fc_matrix(
    data,
    confounds: pd.DataFrame,
    masker: NiftiLabelsMasker,
    state: str,
    state_mask: np.ndarray,
    subj_logger: ContextAdapter,
    run: str,
    connectivity_kind: str = "correlation",
    standardize: str = "zscore_sample",
):
    run_logger = subj_logger.with_ctx(run=run)
    run_logger.info("Fitting state %s", state)

    # Resample to atlas (TRs, n_labels), regressing out the confounds
    time_series = masker.transform(
        data, confounds=confounds, sample_mask=state_mask.astype(bool)
    )

    # Prepare a correlation connectivity measure object and compute the connectivity matrix
    # for the time series
    correlation_measure = ConnectivityMeasure(
        kind=connectivity_kind,
        standardize=standardize,
    )
    correlation_matrix = correlation_measure.fit_transform([time_series])[0]
    np.fill_diagonal(correlation_matrix, 0)

    return correlation_matrix


def fisher_z(r):
    r = np.clip(r, -0.999999, 0.999999)
    return 0.5 * np.log((1 + r) / (1 - r))


def compute_subject_matrix(
    subject: str,
    include_eye_diss: bool,
    mrio: MRIO,
    masker: NiftiLabelsMasker,
    config: GLMConfig,
    convolve: bool = True,
    connectivity_kind: str = "correlation",
    standardize: str = "zscore_sample",
    log_level: int = logging.INFO,
) -> dict[str, np.ndarray]:
    _ensure_worker_logging(log_level=log_level)
    subj_logger = ContextAdapter(logging.getLogger(__name__), {"ctx": {}}).with_ctx(
        subj=subject
    )
    subj_logger.info("PID %s with include_eye_diss %s", os.getpid(), include_eye_diss)
    state_matrices = {}

    for _, run, run_df in mrio.iter_run_events(subject=subject):
        img, _ = mrio.load_functional_image(subject=subject, run=run)
        subj_masker = cast(NiftiLabelsMasker, clone(masker))
        subj_masker.fit(
            img
        )  # Only fit the masker geometry once per subject (same for all runs)

        confound_names = list(set(COMMON_CONFOUNDS) - {"global_signal"})
        confounds = mrio.load_confounds(
            subject=subject, run=run, confound_names=confound_names
        )

        if include_eye_diss:
            eye_confound, _ = mrio.load_eye_move(subject=subject, run=run)
            if convolve:
                eye_confound = convolve_signal(signal=eye_confound, tr=mrio.tr)
            confounds["eye_dissimilarity"] = eye_confound

        state_masks = {}
        for regressor in config.regressor_names:
            state = regressor.split("_")[1]
            state_mask = run_df[state].reset_index(drop=True).to_numpy()
            state_masks[state] = state_mask

        # Add "all" after individual states are collected
        state_masks["all"] = np.ones(len(run_df), dtype=bool)

        for state, state_mask in state_masks.items():
            if state_mask.sum() < 30:
                continue
            matrix = compute_fc_matrix(
                img,
                confounds,
                subj_masker,
                state_mask=state_mask,
                connectivity_kind=connectivity_kind,
                standardize=standardize,
                subj_logger=subj_logger,
                run=run,
                state=state,
            )
            transformed_matrix = fisher_z(matrix)
            state_matrices.setdefault(state, []).append(transformed_matrix)

    state_means = {}
    for state, matrix in state_matrices.items():
        state_means[state] = np.mean(matrix, axis=0)

    return state_means


def compute_all_subjects(
    mrio: MRIO, masker, convolve: bool, config: GLMConfig, n_jobs: int, log_level: int
):
    subjects = mrio.get_subjects()

    run_func_with = partial(
        compute_subject_matrix,
        include_eye_diss=True,
        masker=masker,
        mrio=mrio,
        convolve=convolve,
        config=config,
        log_level=log_level,
    )

    run_func_without = partial(
        compute_subject_matrix,
        include_eye_diss=False,
        masker=masker,
        mrio=mrio,
        convolve=convolve,
        config=config,
        log_level=log_level,
    )

    logger.info("Running subjects in parallel")
    with threadpool_limits(limits=1):
        with_eds = Parallel(n_jobs=n_jobs, backend="loky", verbose=10)(
            delayed(run_func_with)(subject) for subject in subjects
        )
        without_eds = Parallel(n_jobs=n_jobs, backend="loky", verbose=10)(
            delayed(run_func_without)(subject) for subject in subjects
        )

    # else:
    #     logger.info("Running subjects in parallel")
    #     with_eds = Parallel(n_jobs=n_jobs, backend="loky")(delayed(run_func_with)(subject) for subject in subjects)
    #     without_eds = Parallel(n_jobs=n_jobs, verbose=10)(
    #         delayed(run_func_without)(subject) for subject in subjects)
    #

    # with_eds = Parallel(n_jobs=n_jobs, verbose=10)(delayed(run_func_with)(subject) for subject in subjects)
    #
    # without_eds = Parallel(n_jobs=n_jobs, verbose=10)(delayed(run_func_without)(subject) for subject in subjects)

    with_state_arrays = {}
    without_state_arrays = {}
    for i in range(len(subjects)):
        for state, with_matrix in with_eds[i].items():
            with_state_arrays.setdefault(state, []).append(with_matrix)

        for state, without_matrix in without_eds[i].items():
            without_state_arrays.setdefault(state, []).append(without_matrix)

    return with_state_arrays, without_state_arrays


def extract_with_atlas(
    atlas_name: str,
    atlas,
    convolve: bool,
    mrio: MRIO,
    config: GLMConfig,
    output_dir: Path,
    njobs: int,
    log_level: int,
):
    file_name = atlas_name
    file_name += "_conv" if convolve else ""

    masker = NiftiLabelsMasker(
        labels_img=atlas.maps,
        lut=atlas.lut,
        standardize="zscore_sample",
        standardize_confounds=True,
        memory="nilearn_cache",
        verbose=0,
    )

    with_state_eds, without_state_eds = compute_all_subjects(
        mrio=mrio,
        masker=masker,
        convolve=convolve,
        config=config,
        n_jobs=njobs,
        log_level=log_level,
    )

    for state in with_state_eds:
        with_eds = with_state_eds[state]
        without_eds = without_state_eds[state]

        logger.info(
            f"{state} - without: {np.mean(without_eds)}, with: {np.mean(with_eds)}"
        )

        output_dir.mkdir(parents=True, exist_ok=True)
        np.save(output_dir / f"{file_name}_{state}_with_eds.npy", with_eds)
        np.save(output_dir / f"{file_name}_{state}_without_eds.npy", without_eds)

        logger.info(f"Saved FC matrices at {output_dir} / {file_name}")


def cli_main() -> None:
    """
    CLI entrypoint
    """
    parser = argparse.ArgumentParser(
        prog="sleepmreye-fc-extract", description="Extract FC Matrices."
    )

    parser.add_argument(
        "--subjects",
        help="Which subjects to run. If omitted, all subjects will be processed",
        nargs="*",
    )

    parser.add_argument(
        "--njobs",
        type=int,
        help="Number of cpus to use. If omitted, will not run in parallel",
        default=None,
    )

    parser.add_argument(
        "--config",
        help="Path to YAML file describing the FirstLevelConfig",
        required=True,
    )

    parser.add_argument(
        "--atlases",
        help="Which atlases to run. If omitted, all will be processed",
        nargs="*",
    )

    parser.add_argument(
        "--convolve",
        action="store_true",
        help="Convolve the signal before saving",
    )

    parser.add_argument(
        "--output-dir",
        help="Specify a directory to save extracted FC matrices",
        required=True,
    )

    add_io_args(parser)
    add_logger_args(parser)

    args = parser.parse_args()
    mrio = build_mrio_from_args(args)
    config = GLMConfig.from_yaml(args.config)

    enable_logging(level=args.log_level.upper())

    atlases = {
        # "ho": datasets.fetch_atlas_harvard_oxford("cort-maxprob-thr25-2mm"),
        "juelich": datasets.fetch_atlas_juelich("maxprob-thr25-2mm"),
        # "ho_sub": datasets.fetch_atlas_harvard_oxford("sub-maxprob-thr25-2mm"),
        "schaefer500": datasets.fetch_atlas_schaefer_2018(n_rois=500),
        "aal": datasets.fetch_atlas_aal(),
    }

    for name, atlas in atlases.items():
        if args.atlases is not None and name not in args.atlases:
            continue

        extract_with_atlas(
            atlas_name=name,
            atlas=atlas,
            mrio=mrio,
            config=config,
            convolve=args.convolve,
            output_dir=Path(args.output_dir),
            njobs=args.njobs,
            log_level=args.log_level.upper(),
        )
