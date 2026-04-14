"""Optional CLAP-based semantic scoring for audio-text and audio-audio similarity."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def clap_available() -> bool:
    """Check if laion-clap is installed."""
    try:
        import laion_clap  # noqa: F401
        return True
    except ImportError:
        return False


def _load_model():
    """Load the CLAP model (cached on first call)."""
    import laion_clap
    model = laion_clap.CLAP_Module(enable_fusion=False)
    model.load_ckpt()
    return model


_cached_model = None


def _get_model():
    global _cached_model
    if _cached_model is None:
        _cached_model = _load_model()
    return _cached_model


def clap_similarity(audio_path: Path, text_description: str) -> float:
    """Score audio against a text description using CLAP.

    Returns 0.0 - 1.0 similarity.
    Raises ImportError if laion-clap not installed.
    """
    if not clap_available():
        raise ImportError(
            "laion-clap is not installed. "
            "Install with: pip install laion-clap"
        )

    model = _get_model()
    audio_embed = model.get_audio_embedding_from_filelist(
        x=[str(audio_path)], use_tensor=False
    )
    text_embed = model.get_text_embedding(
        [text_description], use_tensor=False
    )

    # Cosine similarity
    audio_vec = audio_embed[0]
    text_vec = text_embed[0]
    norm_a = np.linalg.norm(audio_vec)
    norm_t = np.linalg.norm(text_vec)
    if norm_a < 1e-10 or norm_t < 1e-10:
        return 0.0
    cos_sim = float(np.dot(audio_vec, text_vec) / (norm_a * norm_t))
    # Map from [-1, 1] to [0, 1]
    return float(np.clip((cos_sim + 1.0) / 2.0, 0.0, 1.0))


def clap_audio_similarity(path_a: Path, path_b: Path) -> float:
    """Score similarity between two audio files using CLAP embeddings.

    Returns 0.0 - 1.0 similarity.
    Raises ImportError if laion-clap not installed.
    """
    if not clap_available():
        raise ImportError(
            "laion-clap is not installed. "
            "Install with: pip install laion-clap"
        )

    model = _get_model()
    embeddings = model.get_audio_embedding_from_filelist(
        x=[str(path_a), str(path_b)], use_tensor=False
    )

    vec_a = embeddings[0]
    vec_b = embeddings[1]
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    if norm_a < 1e-10 or norm_b < 1e-10:
        return 0.0
    cos_sim = float(np.dot(vec_a, vec_b) / (norm_a * norm_b))
    return float(np.clip((cos_sim + 1.0) / 2.0, 0.0, 1.0))
