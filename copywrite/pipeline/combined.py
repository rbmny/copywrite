"""Combined generation pipeline: MusicGen + RAVE."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from copywrite.config import CopywriteConfig


_STRATEGIES = ("texture", "rave_seed", "rave_resynthesis")


def generate_combined(
    config: CopywriteConfig,
    prompt: str,
    duration: float = 30.0,
    strategy: str = "texture",
) -> np.ndarray:
    """Generate audio using a combined MusicGen + RAVE strategy.

    Strategies:
        texture          — layer RAVE texture under MusicGen composition
        rave_seed        — use a short RAVE clip as melody conditioning
        rave_resynthesis — re-synthesise MusicGen output through RAVE

    Returns a 1-D numpy array at 32 kHz.
    """
    if strategy not in _STRATEGIES:
        raise ValueError(
            f"Unknown strategy {strategy!r}. Choose from {_STRATEGIES}"
        )

    if strategy == "texture":
        return texture_and_structure(config, prompt, duration)
    if strategy == "rave_seed":
        return rave_seed_musicgen(config, prompt, duration)
    return musicgen_through_rave(config, prompt, duration)


# ---------------------------------------------------------------------------
# Strategy: texture + structure layering
# ---------------------------------------------------------------------------

def texture_and_structure(
    config: CopywriteConfig, prompt: str, duration: float
) -> np.ndarray:
    """Layer RAVE texture underneath MusicGen compositional output.

    MusicGen provides structure and melody at ~80 % volume.
    RAVE provides textural richness at ~30 % volume.
    """
    from copywrite.musicgen.generate import load_model as load_musicgen, generate
    from copywrite.rave import load_model as load_rave, generate_random

    # MusicGen
    mg_model, mg_processor = load_musicgen(config)
    mg_audio = generate(mg_model, mg_processor, prompt, duration=duration)

    # RAVE
    rave_model = load_rave(config)
    rave_audio = generate_random(rave_model, duration=duration)

    # Align lengths
    target_len = min(len(mg_audio), len(rave_audio))
    mg_audio = mg_audio[:target_len]
    rave_audio = rave_audio[:target_len]

    # Mix
    mixed = 0.8 * mg_audio + 0.3 * rave_audio

    # Normalise to prevent clipping
    peak = np.abs(mixed).max()
    if peak > 1.0:
        mixed = mixed / peak

    return mixed


# ---------------------------------------------------------------------------
# Strategy: RAVE seed -> MusicGen melody conditioning
# ---------------------------------------------------------------------------

def rave_seed_musicgen(
    config: CopywriteConfig, prompt: str, duration: float
) -> np.ndarray:
    """Generate a short RAVE clip and use it as melody conditioning for
    MusicGen.
    """
    from copywrite.musicgen.generate import (
        load_model as load_musicgen,
        generate_with_melody,
    )
    from copywrite.rave import load_model as load_rave, generate_random

    # Generate a 4-second RAVE seed
    rave_model = load_rave(config)
    rave_seed = generate_random(rave_model, duration=4.0)

    # Use as melody conditioning
    mg_model, mg_processor = load_musicgen(config)
    audio = generate_with_melody(
        mg_model, mg_processor, prompt,
        melody_audio=rave_seed,
        duration=duration,
        sr=config.musicgen_sample_rate,
    )
    return audio


# ---------------------------------------------------------------------------
# Strategy: MusicGen -> RAVE re-synthesis
# ---------------------------------------------------------------------------

def musicgen_through_rave(
    config: CopywriteConfig, prompt: str, duration: float
) -> np.ndarray:
    """Generate with MusicGen, then re-synthesise through RAVE.

    Encoding MusicGen output into RAVE's latent space and decoding back
    applies RAVE's learned timbre characteristics to the generated audio.
    """
    from copywrite.musicgen.generate import load_model as load_musicgen, generate
    from copywrite.rave import load_model as load_rave, generate_from_audio

    # Generate with MusicGen
    mg_model, mg_processor = load_musicgen(config)
    mg_audio = generate(mg_model, mg_processor, prompt, duration=duration)

    # Re-synthesise through RAVE
    rave_model = load_rave(config)
    resynthesised = generate_from_audio(rave_model, mg_audio)

    return resynthesised
