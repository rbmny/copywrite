"""Shared audio I/O utilities for copywrite."""

from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np
import soundfile as sf


def load_audio(
    path: Path, sr: int = 44100, mono: bool = True
) -> tuple[np.ndarray, int]:
    """Load an audio file and resample to the target sample rate.

    Returns (audio_array, sample_rate).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")

    audio, orig_sr = librosa.load(str(path), sr=sr, mono=mono)
    if audio.size == 0:
        raise ValueError(f"Audio file is empty: {path}")
    return audio, sr


def save_audio(path: Path, audio: np.ndarray, sr: int) -> Path:
    """Save a numpy array as a WAV file, creating parent directories."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio, sr)
    return path


def convert_to_wav(
    input_path: Path, output_path: Path, sr: int, mono: bool = True
) -> Path:
    """Convert any supported audio format to WAV at the given sample rate."""
    audio, sr = load_audio(input_path, sr=sr, mono=mono)
    return save_audio(output_path, audio, sr)


def get_audio_duration(path: Path) -> float:
    """Return the duration of an audio file in seconds."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")
    return float(librosa.get_duration(path=str(path)))


def split_audio(
    audio: np.ndarray,
    sr: int,
    segment_duration: float,
    overlap: float = 0.0,
) -> list[np.ndarray]:
    """Split audio into fixed-length segments.

    Args:
        audio: Audio samples as a 1-D numpy array.
        sr: Sample rate.
        segment_duration: Length of each segment in seconds.
        overlap: Overlap between consecutive segments in seconds.

    Returns:
        List of audio segments. The last segment is included only if it is
        at least half of ``segment_duration``.
    """
    if audio.size == 0:
        return []

    seg_samples = int(segment_duration * sr)
    hop_samples = int((segment_duration - overlap) * sr)

    if seg_samples <= 0 or hop_samples <= 0:
        raise ValueError(
            f"Invalid segment_duration ({segment_duration}) or "
            f"overlap ({overlap})"
        )

    segments: list[np.ndarray] = []
    start = 0
    min_length = seg_samples // 2

    while start < len(audio):
        end = start + seg_samples
        chunk = audio[start:end]
        if len(chunk) >= min_length:
            segments.append(chunk)
        start += hop_samples

    return segments


def beat_aligned_split(
    audio: np.ndarray,
    sr: int,
    target_duration: float,
) -> list[tuple[np.ndarray, float, float]]:
    """Split audio at beat boundaries into segments of roughly *target_duration*.

    Uses ``librosa.beat.beat_track`` to find beat positions, then groups
    consecutive beats so that each segment is as close to *target_duration*
    as possible.

    Returns:
        List of ``(segment_array, start_time, end_time)`` tuples.
    """
    if audio.size == 0:
        return []

    total_duration = len(audio) / sr
    if total_duration <= target_duration:
        return [(audio, 0.0, total_duration)]

    # Detect beats
    _tempo, beat_frames = librosa.beat.beat_track(y=audio, sr=sr)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr).tolist()

    # Ensure we have start and end boundaries
    if not beat_times or beat_times[0] > 0.01:
        beat_times.insert(0, 0.0)
    if beat_times[-1] < total_duration - 0.01:
        beat_times.append(total_duration)

    # Group beats into segments of approximately target_duration
    segments: list[tuple[np.ndarray, float, float]] = []
    seg_start_idx = 0

    for i in range(1, len(beat_times)):
        span = beat_times[i] - beat_times[seg_start_idx]
        if span >= target_duration:
            start_t = beat_times[seg_start_idx]
            end_t = beat_times[i]
            start_sample = int(start_t * sr)
            end_sample = int(end_t * sr)
            segments.append((audio[start_sample:end_sample], start_t, end_t))
            seg_start_idx = i

    # Handle remaining samples
    if seg_start_idx < len(beat_times) - 1:
        start_t = beat_times[seg_start_idx]
        end_t = beat_times[-1]
        start_sample = int(start_t * sr)
        end_sample = int(end_t * sr)
        remainder = audio[start_sample:end_sample]
        if len(remainder) > 0:
            # Merge short remainders into the last segment
            if segments and (end_t - start_t) < target_duration * 0.5:
                prev_audio, prev_start, _prev_end = segments.pop()
                merged = np.concatenate([prev_audio, remainder])
                segments.append((merged, prev_start, end_t))
            else:
                segments.append((remainder, start_t, end_t))

    return segments
