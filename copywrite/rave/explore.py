"""Latent-space exploration utilities for exported RAVE models."""

from __future__ import annotations

import numpy as np
import torch

from .generate import _get_latent_dim, _model_device


def random_walk(
    model: torch.jit.ScriptModule,
    duration: float,
    step_size: float = 0.1,
    sr: int = 48000,
) -> np.ndarray:
    """Generate audio by walking through the latent space.

    Starts at a random latent point and takes small random steps over time,
    producing a smoothly evolving sound.

    Returns a 1-D numpy array of audio samples.
    """
    device = _model_device(model)
    latent_dim = _get_latent_dim(model)

    # Discover the audio-to-latent ratio
    dummy = torch.randn(1, 1, 2048, device=device)
    with torch.no_grad():
        z_dummy = model.encode(dummy)
    ratio = 2048 / z_dummy.shape[2]

    n_samples = int(duration * sr)
    n_latent_frames = int(n_samples / ratio) + 1

    # Build the latent trajectory via cumulative random steps
    z = torch.zeros(1, latent_dim, n_latent_frames, device=device)
    z[:, :, 0] = torch.randn(1, latent_dim, device=device)

    for t in range(1, n_latent_frames):
        step = torch.randn(1, latent_dim, device=device) * step_size
        z[:, :, t] = z[:, :, t - 1] + step

    with torch.no_grad():
        audio = model.decode(z)

    audio = audio.squeeze().cpu().numpy()
    return audio[:n_samples]


def latent_sweep(
    model: torch.jit.ScriptModule,
    dimension: int,
    start: float = -3.0,
    end: float = 3.0,
    steps: int = 20,
    sr: int = 48000,
) -> list[np.ndarray]:
    """Sweep a single latent dimension while holding all others at zero.

    Generates *steps* short audio clips, each decoded from a latent vector
    where ``dimension`` varies linearly from *start* to *end*.

    Returns a list of 1-D numpy arrays (one per step).
    """
    device = _model_device(model)
    latent_dim = _get_latent_dim(model)

    if dimension < 0 or dimension >= latent_dim:
        raise ValueError(
            f"dimension {dimension} out of range for latent_dim {latent_dim}"
        )

    # Use a short latent sequence for each sample (~0.5s of audio)
    dummy = torch.randn(1, 1, 2048, device=device)
    with torch.no_grad():
        z_dummy = model.encode(dummy)
    ratio = 2048 / z_dummy.shape[2]
    n_latent_frames = max(int(sr * 0.5 / ratio), 1)

    values = np.linspace(start, end, steps)
    results: list[np.ndarray] = []

    for val in values:
        z = torch.zeros(1, latent_dim, n_latent_frames, device=device)
        z[:, dimension, :] = val

        with torch.no_grad():
            audio = model.decode(z)

        results.append(audio.squeeze().cpu().numpy())

    return results


def batch_generate(
    model: torch.jit.ScriptModule,
    count: int,
    duration: float,
    sr: int = 48000,
    temperature: float = 1.0,
) -> list[np.ndarray]:
    """Generate multiple random audio samples.

    Each sample is decoded from an independent random latent vector sampled
    from N(0, temperature).

    Returns a list of *count* 1-D numpy arrays.
    """
    device = _model_device(model)
    latent_dim = _get_latent_dim(model)

    # Discover the audio-to-latent ratio
    dummy = torch.randn(1, 1, 2048, device=device)
    with torch.no_grad():
        z_dummy = model.encode(dummy)
    ratio = 2048 / z_dummy.shape[2]

    n_samples = int(duration * sr)
    n_latent_frames = int(n_samples / ratio) + 1

    results: list[np.ndarray] = []

    for _ in range(count):
        z = torch.randn(1, latent_dim, n_latent_frames, device=device) * temperature

        with torch.no_grad():
            audio = model.decode(z)

        audio_np = audio.squeeze().cpu().numpy()
        results.append(audio_np[:n_samples])

    return results
