import argparse
import logging
from functools import partial
from typing import Callable

import mne
import numpy as np
import pandas as pd
from pathlib import Path
from joblib import Parallel, delayed

from mreyemove import enable_logging, ContextAdapter
from mreyemove.cli.common_args import add_logger_args, add_io_args, build_mrio_from_args
from mreyemove.data.mrio import MRIO

logger = logging.getLogger(__name__)


def extract_eog(eeg_dir: Path, subject: str, run: str, subj_logger: ContextAdapter, l_filter: float = None, h_filter: float = None, ):
    path = Path(eeg_dir) / f"sub-{subject}" / f"sub-{subject}_task-sleep_run-{run}_eeg_desc-gacbcg_eeg.set"
    try:
        raw = mne.io.read_raw_eeglab(str(path), preload=True)
        if 'EOG' in raw.ch_names:
            raw.set_channel_types({'EOG': 'eog'})
        raw = raw.filter(l_freq=l_filter, h_freq=h_filter, picks='EOG')
    except FileNotFoundError:
        subj_logger.error("Could not find file %s ", path)
        return None
    return raw


def extract_TR_epochs(raw, picks=None, annotation: str = 'R128', offset_fraction=0.0):
    events, event_ids = mne.events_from_annotations(raw)
    r128_id = {annotation: event_ids[annotation]}
    r128_events = events[events[:, 2] == r128_id[annotation]]
    new_event_id = 999
    sfreq = raw.info['sfreq']

    # Compute shifted onsets (drop last base event: no following interval)
    onsets = r128_events[:, 0]
    win_len = np.median(np.diff(onsets))
    if offset_fraction > 0:
        shifted_onsets = onsets[:-1] + offset_fraction * win_len
    else:
        shifted_onsets = onsets
    tmax = win_len / sfreq

    # Build synthetic events at shifted onsets
    samples = shifted_onsets.astype(int)
    events_shifted = np.c_[samples,
                           np.zeros_like(samples),
                           np.full_like(samples, new_event_id, dtype=int)]

    if picks is None:
        picks = mne.pick_types(raw.info, meg=False, eeg=False, stim=False, eog=True)

    epochs = mne.Epochs(
        raw,
        events_shifted,
        event_id=new_event_id,
        tmin=0.0,
        tmax=tmax,
        picks=picks,
        baseline=None,
        preload=True
    )
    return epochs


def run_subject(subject: str, mrio: MRIO, eeg_dir: Path, aggregator: Callable, rectify: bool = False, log_level: int = logging.INFO, ):
    enable_logging(level=log_level)
    subj_logger = ContextAdapter(logging.getLogger(__name__), {"ctx": {}}).with_ctx(
        subj=subject
    )

    mne.set_log_level("WARNING")

    signals = []
    subjects = []
    runs = []
    TRs = []
    subj_logger.info("Running subject %s ", subject)

    for _, run, _ in mrio.iter_run_events(subject=subject, run_exclusion=False):
        raw = extract_eog(subject=subject, eeg_dir=eeg_dir, run=run, l_filter=0.1, subj_logger=subj_logger)
        if raw is None:
            mrio.qc_logger.log(
                subject=subject,
                run=run,
                excluded=True,
                stage="eeg_read",
                no_eeg_found=True
            )
            continue

        channel_raw = raw.copy().pick(["EOG"])
        epochs = extract_TR_epochs(channel_raw, picks=['EOG'], annotation="R128", offset_fraction=0.0)

        signal = []
        for epoch in epochs:
            if rectify:
                epoch = np.abs(epoch)
            signal.append(aggregator(epoch))

        signals.extend(signal)
        TRs.extend(range(0, len(signal)))
        subjects.extend([subject for _ in range(len(signal))])
        runs.extend([run for _ in range(len(signal))])

    df = pd.DataFrame({
        "subject": subjects,
        "run": runs,
        "TR": TRs,
        "signal": signals
    })
    return df


def main(mrio: MRIO, eeg_dir: Path, rectify: bool, aggregator: Callable):
    subjects = mrio.subjects
    logger.info("Running EOG extractor for all subjects with rectify = %s", rectify)

    mrio_qc_path = eeg_dir / "qc"
    mrio_qc_path.mkdir(parents=True, exist_ok=True)
    mrio.qc_logger.set_path(mrio_qc_path, desc_add="eeg_extraction")
    logger.debug(mrio.qc_logger.path)

    run_func = partial(run_subject, eeg_dir=eeg_dir, mrio=mrio, rectify=rectify, aggregator=aggregator)
    results = Parallel(n_jobs=6)(delayed(run_func)(subject) for subject in subjects)
    _df = pd.concat(results, ignore_index=True)
    file_name = "rectify_" if rectify else ""
    file_name += f"{aggregator.__name__}_eog_reg.p"

    result_dir = mrio.mreyemove_dir / 'group' / 'eeg'
    result_dir.mkdir(parents=True, exist_ok=True)
    path = result_dir / file_name
    logger.info("Saving to %s", path)
    _df.to_pickle(path)


def cli_main():
    parser = argparse.ArgumentParser(
        prog="sleepmreye-extract-eog", description="Extract EOG signals."
    )

    parser.add_argument(
        "--rectify",
        help="Rectify the signal",
        action=argparse.BooleanOptionalAction,
    )

    parser.add_argument(
        "--aggregator",
        help="numpy function used to aggregate EOG signal within a TR (mean|median|max)",
        required=True,
    )

    parser.add_argument(
        "--eeg-derivatives",
        help="Path containing EEG signals of all subjects",
        required=True,
    )

    add_logger_args(parser)
    add_io_args(parser)

    args = parser.parse_args()

    enable_logging(level=args.log_level)

    if hasattr(args, "io_kwarg"):
        args.io_kwarg.append("load_eog=False")
    else:
        args.io_kwarg = ["load_eog=False"]
    args.load_signal = False

    mrio = build_mrio_from_args(args)

    if not hasattr(np, args.aggregator):
        raise argparse.ArgumentTypeError(f"Aggregator must be a numpy function")
    agg = getattr(np, args.aggregator)

    eeg_dir = mrio.bids_root / "derivatives" / args.eeg_derivatives
    if not eeg_dir.exists():
        raise ValueError(f"Could not find EEG directory {eeg_dir}")

    main(mrio=mrio, eeg_dir=eeg_dir, rectify=args.rectify, aggregator=agg)
