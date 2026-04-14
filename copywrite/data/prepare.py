"""Data preparation pipeline for RAVE and MusicGen training."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

from copywrite.config import CopywriteConfig
from copywrite.utils.audio import (
    load_audio,
    save_audio,
    beat_aligned_split,
    split_audio,
    get_audio_duration,
)

AUDIO_EXTENSIONS = {".mp3", ".wav", ".flac"}


def _discover_audio_files(directory: Path) -> list[Path]:
    """Recursively find all supported audio files in a directory."""
    files: list[Path] = []
    if not directory.exists():
        return files
    for ext in AUDIO_EXTENSIONS:
        files.extend(directory.rglob(f"*{ext}"))
    files.sort()
    return files


def prepare_for_rave(
    audio_path: Path,
    output_dir: Path,
    sr: int = 48000,
    segment_duration: float = 4.0,
) -> list[Path]:
    """Prepare a single audio file for RAVE training.

    Loads audio as mono at the target sample rate, splits into segments
    of ``segment_duration`` seconds with 50% overlap, and saves each
    segment as a WAV file.

    Returns:
        List of paths to the created WAV segments.
    """
    audio, sr = load_audio(audio_path, sr=sr, mono=True)

    overlap_seconds = segment_duration * 0.5
    segments = split_audio(audio, sr, segment_duration, overlap=overlap_seconds)

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = audio_path.stem
    created: list[Path] = []

    for idx, seg in enumerate(segments):
        out_path = output_dir / f"{stem}_rave_{idx:04d}.wav"
        save_audio(out_path, seg, sr)
        created.append(out_path)

    return created


def prepare_for_musicgen(
    audio_path: Path,
    output_dir: Path,
    sr: int = 32000,
    segment_duration: float = 20.0,
) -> list[Path]:
    """Prepare a single audio file for MusicGen training.

    Loads audio as stereo at the target sample rate, performs beat-aligned
    splitting into segments of approximately ``segment_duration`` seconds
    (targeting 15-30s), and saves each segment as a WAV file.

    Returns:
        List of paths to the created WAV segments.
    """
    audio, sr = load_audio(audio_path, sr=sr, mono=False)

    # beat_aligned_split expects a 1-D array, so use mono for beat detection
    if audio.ndim == 2:
        mono_audio = np.mean(audio, axis=0)
    else:
        mono_audio = audio

    beat_segments = beat_aligned_split(mono_audio, sr, segment_duration)

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = audio_path.stem
    created: list[Path] = []

    for idx, (seg_mono, start_t, end_t) in enumerate(beat_segments):
        seg_duration = end_t - start_t
        if seg_duration < 15.0 or seg_duration > 30.0:
            # Skip segments outside the 15-30s target range
            continue

        # Extract the corresponding stereo segment
        start_sample = int(start_t * sr)
        end_sample = int(end_t * sr)
        if audio.ndim == 2:
            seg = audio[:, start_sample:end_sample]
            # soundfile expects (samples, channels)
            seg = seg.T
        else:
            seg = audio[start_sample:end_sample]

        out_path = output_dir / f"{stem}_mg_{idx:04d}.wav"
        save_audio(out_path, seg, sr)
        created.append(out_path)

    return created


def prepare_all(
    config: CopywriteConfig,
    input_dir: Path | None = None,
) -> dict:
    """Discover and prepare all audio files for RAVE and MusicGen training.

    Args:
        config: Project configuration.
        input_dir: Directory containing source audio files.
            Defaults to ``config.reference_dir``.

    Returns:
        Stats dict with keys: rave_files, musicgen_files,
        total_source_files, rave_duration, musicgen_duration.
    """
    source_dir = input_dir or config.reference_dir
    audio_files = _discover_audio_files(source_dir)

    config.ensure_dirs()
    rave_out = config.rave_preprocessed_dir
    mg_out = config.musicgen_preprocessed_dir

    rave_count = 0
    mg_count = 0
    rave_duration = 0.0
    mg_duration = 0.0

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
    ) as progress:
        task = progress.add_task("Preparing audio", total=len(audio_files))

        for audio_path in audio_files:
            progress.update(task, description=f"Processing {audio_path.name}")

            # Prepare for RAVE
            rave_segments = prepare_for_rave(
                audio_path,
                rave_out,
                sr=config.rave_sample_rate,
                segment_duration=config.rave_segment_duration,
            )
            rave_count += len(rave_segments)
            for seg_path in rave_segments:
                rave_duration += get_audio_duration(seg_path)

            # Prepare for MusicGen
            mg_segments = prepare_for_musicgen(
                audio_path,
                mg_out,
                sr=config.musicgen_sample_rate,
                segment_duration=config.musicgen_segment_duration,
            )
            mg_count += len(mg_segments)
            for seg_path in mg_segments:
                mg_duration += get_audio_duration(seg_path)

            progress.advance(task)

    return {
        "total_source_files": len(audio_files),
        "rave_files": rave_count,
        "musicgen_files": mg_count,
        "rave_duration": rave_duration,
        "musicgen_duration": mg_duration,
    }


def get_preparation_stats(config: CopywriteConfig) -> dict:
    """Count files and total duration in preprocessed directories.

    Returns:
        Stats dict with keys: rave_files, musicgen_files,
        rave_duration, musicgen_duration.
    """
    rave_files = list(config.rave_preprocessed_dir.glob("*.wav"))
    mg_files = list(config.musicgen_preprocessed_dir.glob("*.wav"))

    rave_duration = 0.0
    for f in rave_files:
        rave_duration += get_audio_duration(f)

    mg_duration = 0.0
    for f in mg_files:
        mg_duration += get_audio_duration(f)

    return {
        "rave_files": len(rave_files),
        "musicgen_files": len(mg_files),
        "rave_duration": rave_duration,
        "musicgen_duration": mg_duration,
    }
