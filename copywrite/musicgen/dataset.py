"""Dataset preparation for MusicGen fine-tuning."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from copywrite.config import CopywriteConfig
from copywrite.utils.audio import get_audio_duration


def prepare_dataset(config: CopywriteConfig) -> Path:
    """Create a MusicGen training dataset directory.

    Reads WAV files from config.musicgen_preprocessed_dir and captions from
    config.captions_dir, then writes:
      - metadata.jsonl  (one JSON line per sample)
      - copies of the WAV files

    Returns the dataset directory path.
    """
    wav_dir = config.musicgen_preprocessed_dir
    captions_json = config.captions_dir / "captions.json"

    if not captions_json.exists():
        raise FileNotFoundError(
            f"Captions file not found: {captions_json}. "
            "Run captioning before dataset preparation."
        )

    captions: dict[str, str] = json.loads(
        captions_json.read_text(encoding="utf-8")
    )

    dataset_dir = config.musicgen_dataset_dir
    dataset_dir.mkdir(parents=True, exist_ok=True)

    entries: list[dict[str, str]] = []
    for wav_name, caption in sorted(captions.items()):
        src = wav_dir / wav_name
        if not src.exists():
            raise FileNotFoundError(
                f"WAV file referenced in captions not found: {src}"
            )
        dest = dataset_dir / wav_name
        if not dest.exists() or dest.stat().st_size != src.stat().st_size:
            shutil.copy2(src, dest)
        entries.append({"file_name": wav_name, "text": caption})

    metadata_path = dataset_dir / "metadata.jsonl"
    with open(metadata_path, "w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")

    return dataset_dir


def validate_dataset(dataset_dir: Path) -> dict:
    """Validate that every entry in metadata.jsonl has a matching WAV file
    and a non-empty caption.

    Returns {"valid": bool, "errors": [...], "file_count": int}.
    """
    metadata_path = dataset_dir / "metadata.jsonl"
    if not metadata_path.exists():
        return {
            "valid": False,
            "errors": ["metadata.jsonl not found"],
            "file_count": 0,
        }

    errors: list[str] = []
    count = 0

    for line_num, line in enumerate(
        metadata_path.read_text(encoding="utf-8").strip().splitlines(), start=1
    ):
        entry = json.loads(line)
        count += 1
        wav_path = dataset_dir / entry["file_name"]
        if not wav_path.exists():
            errors.append(f"Line {line_num}: WAV missing — {entry['file_name']}")
        if not entry.get("text", "").strip():
            errors.append(f"Line {line_num}: empty caption — {entry['file_name']}")

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "file_count": count,
    }


def get_dataset_stats(dataset_dir: Path) -> dict:
    """Return summary statistics for the dataset.

    Returns {"file_count": int, "total_duration": float,
             "avg_caption_length": float}.
    """
    metadata_path = dataset_dir / "metadata.jsonl"
    if not metadata_path.exists():
        return {
            "file_count": 0,
            "total_duration": 0.0,
            "avg_caption_length": 0.0,
        }

    total_duration = 0.0
    total_caption_len = 0
    count = 0

    for line in metadata_path.read_text(encoding="utf-8").strip().splitlines():
        entry = json.loads(line)
        count += 1
        wav_path = dataset_dir / entry["file_name"]
        if wav_path.exists():
            total_duration += get_audio_duration(wav_path)
        total_caption_len += len(entry.get("text", ""))

    return {
        "file_count": count,
        "total_duration": round(total_duration, 2),
        "avg_caption_length": round(total_caption_len / max(count, 1), 1),
    }
