"""Auto-captioning for MusicGen training data."""

from __future__ import annotations

import json
from pathlib import Path

from rich.progress import Progress

from copywrite.config import CopywriteConfig
from copywrite.scoring.features import extract_features, AudioFeatures


# ---------------------------------------------------------------------------
# Album-specific descriptors
# ---------------------------------------------------------------------------

_ALBUM_DESCRIPTORS: dict[str, str] = {
    "Homework": "acid house, distorted bass, raw analog synths",
    "Discovery": "disco samples, vocoder, filtered funk",
    "Human After All": "heavy distortion, repetitive vocals, industrial textures",
    "Random Access Memories": "live instrumentation, polished disco, lush production",
    "Alive 2007": "mashup, live energy, layered transitions",
}


# ---------------------------------------------------------------------------
# Feature-to-description mapping
# ---------------------------------------------------------------------------

def _tempo_word(bpm: float) -> str:
    if bpm < 90:
        return "slow"
    if bpm < 115:
        return "moderate"
    if bpm < 135:
        return "upbeat"
    return "fast"


def _energy_word(rms: float) -> str:
    if rms < 0.05:
        return "soft"
    if rms < 0.15:
        return "moderate energy"
    return "high energy"


def _brightness_word(centroid: float) -> str:
    if centroid < 1500:
        return "dark"
    if centroid < 3000:
        return "warm"
    return "bright"


def _sidechain_description(depth: float) -> str:
    if depth < 1.0:
        return ""
    if depth < 3.0:
        return "subtle sidechain compression"
    return "heavy sidechain pumping"


def _features_to_description(
    features: AudioFeatures, metadata: dict | None = None
) -> str:
    """Convert extracted audio features into a natural-language caption."""
    parts: list[str] = []

    # Core style tags
    parts.append("Daft Punk")
    parts.append("french house")
    parts.append("electronic")

    # Tempo
    bpm = features.rhythm.tempo
    parts.append(f"{_tempo_word(bpm)} tempo {int(round(bpm))} BPM")

    # Key
    key = features.harmony.key
    mode = "minor" if "m" in key else "major"
    parts.append(f"{key} {mode}")

    # Energy
    parts.append(_energy_word(features.dynamics.rms_mean))

    # Brightness
    parts.append(_brightness_word(features.spectral.spectral_centroid_mean))

    # Sidechain
    sc = _sidechain_description(features.dynamics.sidechain_depth)
    if sc:
        parts.append(sc)

    # Filter house tag based on spectral flatness
    if features.spectral.spectral_flatness_mean < 0.1:
        parts.append("filter house")

    # Album-specific descriptors
    if metadata and "album" in metadata:
        album = metadata["album"]
        desc = _ALBUM_DESCRIPTORS.get(album)
        if desc:
            parts.append(desc)

    return ", ".join(parts)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def caption_segment(
    audio_path: Path, track_metadata: dict | None = None
) -> str:
    """Extract audio features and return a natural-language caption."""
    features = extract_features(audio_path)
    return _features_to_description(features, track_metadata)


def caption_all(
    config: CopywriteConfig, track_metadata: dict | None = None
) -> dict[str, str]:
    """Caption all WAV files in the preprocessed directory.

    Saves individual .txt caption files alongside the WAVs and a combined
    captions.json in config.captions_dir.  Returns a dict mapping filename
    to caption string.
    """
    wav_dir = config.musicgen_preprocessed_dir
    wav_files = sorted(wav_dir.glob("*.wav"))
    if not wav_files:
        raise FileNotFoundError(
            f"No WAV files found in {wav_dir}. "
            "Run preprocessing before captioning."
        )

    config.captions_dir.mkdir(parents=True, exist_ok=True)
    captions: dict[str, str] = {}

    with Progress() as progress:
        task = progress.add_task("Captioning segments", total=len(wav_files))
        for wav_path in wav_files:
            caption = caption_segment(wav_path, track_metadata)
            captions[wav_path.name] = caption

            # Save individual caption file
            txt_path = config.captions_dir / f"{wav_path.stem}.txt"
            txt_path.write_text(caption, encoding="utf-8")

            progress.advance(task)

    # Save combined JSON
    json_path = config.captions_dir / "captions.json"
    json_path.write_text(
        json.dumps(captions, indent=2), encoding="utf-8"
    )

    return captions
