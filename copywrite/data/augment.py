"""Audio augmentation for training data expansion."""

from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from scipy.signal import butter, sosfilt

from copywrite.utils.audio import load_audio, save_audio

DEFAULT_AUGMENTATIONS = ["pitch_up", "pitch_down", "stretch_fast", "stretch_slow"]


def pitch_shift(audio: np.ndarray, sr: int, semitones: float) -> np.ndarray:
    """Shift pitch by the given number of semitones."""
    if audio.size == 0:
        return audio
    return librosa.effects.pitch_shift(y=audio, sr=sr, n_steps=semitones)


def time_stretch(audio: np.ndarray, sr: int, rate: float) -> np.ndarray:
    """Time-stretch audio by the given rate (>1 = faster, <1 = slower)."""
    if audio.size == 0:
        return audio
    return librosa.effects.time_stretch(y=audio, rate=rate)


def eq_variation(audio: np.ndarray, sr: int, seed: int = 0) -> np.ndarray:
    """Apply a subtle random EQ variation.

    Picks 3 random frequency bands and boosts or cuts each by up to +/-2 dB
    using second-order Butterworth bandpass filters.
    """
    if audio.size == 0:
        return audio

    rng = np.random.default_rng(seed)
    nyquist = sr / 2.0

    # Define reasonable frequency band boundaries (Hz)
    band_edges = [80, 250, 600, 1200, 3000, 6000, 12000]
    # Filter out bands above nyquist
    band_edges = [f for f in band_edges if f < nyquist]

    if len(band_edges) < 2:
        return audio

    # Pick 3 random bands (or fewer if not enough edges)
    num_bands = min(3, len(band_edges) - 1)
    band_indices = rng.choice(len(band_edges) - 1, size=num_bands, replace=False)

    result = audio.copy().astype(np.float64)

    for idx in band_indices:
        low = band_edges[idx]
        high = band_edges[idx + 1]

        # Random gain between -2 and +2 dB
        gain_db = rng.uniform(-2.0, 2.0)
        gain_linear = 10.0 ** (gain_db / 20.0)

        # Design a bandpass filter
        low_norm = low / nyquist
        high_norm = high / nyquist
        # Clamp to valid range
        low_norm = max(low_norm, 0.001)
        high_norm = min(high_norm, 0.999)

        if low_norm >= high_norm:
            continue

        sos = butter(2, [low_norm, high_norm], btype="band", output="sos")
        band_signal = sosfilt(sos, result)

        # Apply gain to the band and mix back
        result = result + band_signal * (gain_linear - 1.0)

    # Prevent clipping
    peak = np.max(np.abs(result))
    if peak > 1.0:
        result /= peak

    return result.astype(np.float32)


def hpss_isolate(
    audio: np.ndarray, sr: int, component: str = "harmonic"
) -> np.ndarray:
    """Isolate harmonic or percussive component using HPSS.

    Args:
        audio: Input audio array.
        sr: Sample rate (unused but kept for consistent signature).
        component: Either "harmonic" or "percussive".
    """
    if audio.size == 0:
        return audio

    harmonic, percussive = librosa.effects.hpss(y=audio)

    if component == "harmonic":
        return harmonic
    elif component == "percussive":
        return percussive
    else:
        raise ValueError(f"Unknown HPSS component: {component!r}. "
                         f"Use 'harmonic' or 'percussive'.")


# Map augmentation name -> (function, kwargs)
_AUGMENTATION_REGISTRY: dict[str, tuple] = {
    "pitch_up": (pitch_shift, {"semitones": 2.0}),
    "pitch_down": (pitch_shift, {"semitones": -2.0}),
    "stretch_fast": (time_stretch, {"rate": 1.1}),
    "stretch_slow": (time_stretch, {"rate": 0.9}),
    "eq": (eq_variation, {}),
    "harmonic": (hpss_isolate, {"component": "harmonic"}),
    "percussive": (hpss_isolate, {"component": "percussive"}),
}


def augment_directory(
    input_dir: Path,
    output_dir: Path,
    augmentations: list[str] | None = None,
    sr: int = 48000,
) -> list[Path]:
    """Apply augmentations to every WAV file in a directory.

    Args:
        input_dir: Directory containing source WAV files.
        output_dir: Directory to write augmented files.
        augmentations: List of augmentation names to apply.
            Defaults to pitch_up, pitch_down, stretch_fast, stretch_slow.
        sr: Sample rate for loading audio.

    Returns:
        List of paths to all created augmented files.
    """
    if augmentations is None:
        augmentations = DEFAULT_AUGMENTATIONS

    wav_files = sorted(input_dir.glob("*.wav"))
    if not wav_files:
        return []

    output_dir.mkdir(parents=True, exist_ok=True)
    created: list[Path] = []
    total_ops = len(wav_files) * len(augmentations)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
    ) as progress:
        task = progress.add_task("Augmenting audio", total=total_ops)

        for wav_path in wav_files:
            audio, loaded_sr = load_audio(wav_path, sr=sr, mono=True)

            if audio.size == 0:
                progress.advance(task, advance=len(augmentations))
                continue

            stem = wav_path.stem

            for aug_name in augmentations:
                progress.update(
                    task,
                    description=f"{stem} [{aug_name}]",
                )

                if aug_name not in _AUGMENTATION_REGISTRY:
                    progress.advance(task)
                    continue

                func, kwargs = _AUGMENTATION_REGISTRY[aug_name]
                augmented = func(audio, loaded_sr, **kwargs)

                out_path = output_dir / f"{stem}_{aug_name}.wav"
                save_audio(out_path, augmented, loaded_sr)
                created.append(out_path)

                progress.advance(task)

    return created
