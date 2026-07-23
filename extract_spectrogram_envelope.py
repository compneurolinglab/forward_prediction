#!/usr/bin/env python3
"""
Extract spectrogram and Hilbert envelope on a fixed 10 ms time grid.

Given mono audio, this script computes:

  - 129-bin log-power spectrogram (STFT, nperseg=256, PSD in dB)
  - 1 Hilbert amplitude envelope channel

Both are linearly interpolated to a uniform grid (default 100 Hz, 10 ms).
Output shape: (n_timepoints, n_spectrogram_bins + 1) = (T, 130).

This is the acoustic feature construction used upstream of PCA in the
analysis pipeline; PCA and sentence-prefix logic are not included here.

Example:
  python extract_spectrogram_envelope.py \\
    --audio conversation.wav \\
    --output-dir ./out \\
    --tag sentence01

"""

from __future__ import annotations

import argparse
import json
import os
from functools import lru_cache

import numpy as np
import soundfile as sf
from scipy import signal


DEFAULT_GRID_INTERVAL_S = 0.010
DEFAULT_NPERSEG = 256
DEFAULT_N_SPECTROGRAM_BINS = 129


@lru_cache(maxsize=512)
def file_peak(audio_path: str) -> float:
    """Peak absolute amplitude of a WAV file (for optional normalization)."""
    peak = 0.0
    with sf.SoundFile(audio_path) as audio_file:
        for block in audio_file.blocks(blocksize=1_000_000, dtype="float64", always_2d=True):
            mono_block = block.mean(axis=1)
            peak = max(peak, float(np.max(np.abs(mono_block))))
    return peak


def load_mono_wav(
    audio_path: str,
    *,
    start_seconds: float | None = None,
    end_seconds: float | None = None,
    normalize_peak: bool = True,
) -> tuple[np.ndarray, int]:
    """
    Load mono audio from WAV, optionally clipped to [start, end) in seconds.

    When normalize_peak is True, divide by the file's peak amplitude.
    """
    with sf.SoundFile(audio_path) as audio_file:
        sample_rate = int(audio_file.samplerate)
        if start_seconds is None and end_seconds is None:
            audio = audio_file.read(dtype="float64", always_2d=True)
        else:
            start = 0.0 if start_seconds is None else float(start_seconds)
            end = float(audio_file.frames) / sample_rate if end_seconds is None else float(end_seconds)
            start_sample = max(0, int(np.floor(start * sample_rate)))
            end_sample = min(audio_file.frames, int(np.ceil(end * sample_rate)))
            if end_sample <= start_sample:
                raise ValueError("Audio segment bounds select no samples.")
            audio_file.seek(start_sample)
            audio = audio_file.read(end_sample - start_sample, dtype="float64", always_2d=True)

    mono = np.asarray(audio, dtype=np.float64).mean(axis=1)
    if normalize_peak:
        peak = file_peak(os.path.abspath(audio_path))
        if peak > 0.0:
            mono /= peak
    return mono, sample_rate


def extract_spectrogram_envelope(
    audio: np.ndarray,
    sample_rate: int,
    *,
    grid_interval_seconds: float = DEFAULT_GRID_INTERVAL_S,
    nperseg: int = DEFAULT_NPERSEG,
    expected_n_bins: int | None = DEFAULT_N_SPECTROGRAM_BINS,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """
    Compute spectrogram + envelope on a uniform time grid.

    Parameters
    ----------
    audio : 1D mono waveform
    sample_rate : Hz
    grid_interval_seconds : target grid step (default 0.01 s → 100 Hz)

    Returns
    -------
    features : (T, n_bins + 1) float64 — spectrogram dB then envelope
    time_grid : (T,) seconds relative to segment start
    meta : extraction parameters and shapes
    """
    waveform = np.asarray(audio, dtype=np.float64).ravel()
    if waveform.size == 0:
        raise ValueError("Empty audio waveform.")
    if sample_rate <= 0:
        raise ValueError(f"Invalid sample_rate: {sample_rate}")

    grid_step = float(grid_interval_seconds)
    if grid_step <= 0.0:
        raise ValueError("grid_interval_seconds must be positive.")

    duration = waveform.size / float(sample_rate)
    time_grid = np.arange(0.0, duration, grid_step, dtype=np.float64)
    if time_grid.size < 2:
        raise ValueError(f"Audio too short for grid extraction ({duration:.4f} s).")

    nperseg = int(nperseg)
    frequencies, spectrogram_times, spectrogram_power = signal.spectrogram(
        waveform,
        fs=sample_rate,
        nperseg=nperseg,
        scaling="density",
        mode="psd",
    )
    n_bins = int(len(frequencies))
    if expected_n_bins is not None and n_bins != int(expected_n_bins):
        raise RuntimeError(
            f"Expected {expected_n_bins} spectrogram bins (nperseg={nperseg}), got {n_bins}."
        )

    spectrogram_db = 10.0 * np.log10(spectrogram_power + 1e-10)
    if spectrogram_times.size == 1:
        spectrogram_at_grid = np.repeat(spectrogram_db.T, time_grid.size, axis=0)
    else:
        spectrogram_at_grid = np.column_stack(
            [
                np.interp(time_grid, spectrogram_times, spectrogram_db[fi])
                for fi in range(n_bins)
            ]
        )

    envelope = np.abs(signal.hilbert(waveform))
    sample_times = np.arange(waveform.size, dtype=np.float64) / float(sample_rate)
    envelope_at_grid = np.interp(time_grid, sample_times, envelope)

    features = np.column_stack((spectrogram_at_grid, envelope_at_grid)).astype(np.float64)
    meta: dict[str, object] = {
        "sample_rate_hz": int(sample_rate),
        "duration_seconds": float(duration),
        "grid_interval_seconds": grid_step,
        "grid_hz": 1.0 / grid_step,
        "n_timepoints": int(time_grid.size),
        "n_spectrogram_bins": n_bins,
        "n_features": int(n_bins + 1),
        "nperseg": nperseg,
        "spectrogram_scale": "dB PSD",
        "feature_layout": "columns 0:n_bins-1 = spectrogram dB; column n_bins = Hilbert envelope",
    }
    return features, time_grid, meta


def slice_time_range(
    features: np.ndarray,
    time_grid: np.ndarray,
    *,
    start_seconds: float,
    end_seconds: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Subselect grid samples with start <= t < end (relative to segment origin).

    Endpoints follow np.arange-style half-open intervals used in the full pipeline.
    """
    t0 = float(start_seconds)
    t1 = float(end_seconds)
    mask = (time_grid >= t0) & (time_grid < t1)
    if not bool(np.any(mask)):
        raise ValueError(
            f"No grid samples in [{t0}, {t1}); grid spans "
            f"[{time_grid[0]:.4f}, {time_grid[-1]:.4f}] with step {time_grid[1]-time_grid[0]:.4f}."
        )
    return features[mask], time_grid[mask]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio", type=str, required=True, help="Input WAV path.")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--tag", type=str, default="audio")
    parser.add_argument(
        "--start",
        type=float,
        default=None,
        help="Optional clip start in seconds (within the WAV).",
    )
    parser.add_argument(
        "--end",
        type=float,
        default=None,
        help="Optional clip end in seconds (within the WAV).",
    )
    parser.add_argument(
        "--slice-start",
        type=float,
        default=None,
        help="After extraction, keep grid times >= this (relative to clipped segment).",
    )
    parser.add_argument(
        "--slice-end",
        type=float,
        default=None,
        help="After extraction, keep grid times < this (relative to clipped segment).",
    )
    parser.add_argument(
        "--grid-interval",
        type=float,
        default=DEFAULT_GRID_INTERVAL_S,
        help="Uniform grid step in seconds (default: 0.01).",
    )
    parser.add_argument("--nperseg", type=int, default=DEFAULT_NPERSEG)
    parser.add_argument(
        "--no-normalize-peak",
        action="store_true",
        help="Do not divide waveform by file peak before STFT/envelope.",
    )
    return parser.parse_args()


def save_outputs(
    *,
    tag: str,
    output_dir: str,
    features: np.ndarray,
    time_grid: np.ndarray,
    meta: dict[str, object],
    audio_path: str,
    clip_start: float | None,
    clip_end: float | None,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, f"{tag}_acoustic.npy"), features)
    np.save(os.path.join(output_dir, f"{tag}_time_grid.npy"), time_grid)

    summary = {
        "tag": tag,
        "audio_path": audio_path,
        "clip_start_seconds": clip_start,
        "clip_end_seconds": clip_end,
        "shape": list(features.shape),
        **meta,
    }
    with open(os.path.join(output_dir, f"{tag}_acoustic_summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)


def main() -> None:
    args = parse_args()
    audio, sample_rate = load_mono_wav(
        args.audio,
        start_seconds=args.start,
        end_seconds=args.end,
        normalize_peak=not bool(args.no_normalize_peak),
    )
    features, time_grid, meta = extract_spectrogram_envelope(
        audio,
        sample_rate,
        grid_interval_seconds=float(args.grid_interval),
        nperseg=int(args.nperseg),
    )

    if args.slice_start is not None or args.slice_end is not None:
        if args.slice_start is None or args.slice_end is None:
            raise SystemExit("Provide both --slice-start and --slice-end.")
        features, time_grid = slice_time_range(
            features,
            time_grid,
            start_seconds=float(args.slice_start),
            end_seconds=float(args.slice_end),
        )
        meta = dict(meta)
        meta["n_timepoints"] = int(time_grid.size)
        meta["slice_start_seconds"] = float(args.slice_start)
        meta["slice_end_seconds"] = float(args.slice_end)

    save_outputs(
        tag=str(args.tag),
        output_dir=str(args.output_dir),
        features=features,
        time_grid=time_grid,
        meta=meta,
        audio_path=str(args.audio),
        clip_start=args.start,
        clip_end=args.end,
    )
    print(
        f"[{args.tag}] saved acoustic features shape={features.shape} "
        f"-> {args.output_dir}"
    )


if __name__ == "__main__":
    main()
