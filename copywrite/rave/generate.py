"""Load exported RAVE models and generate audio from the latent space."""

from __future__ import annotations

import numpy as np
import torch


def _get_device() -> torch.device:
    """Return the best available device (cuda if present, else cpu)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_model(model_path: "Path") -> torch.jit.ScriptModule:
    """Load an exported RAVE ``.ts`` model and move it to the best device."""
    from pathlib import Path

    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"RAVE model not found: {model_path}")

    device = _get_device()
    model = torch.jit.load(str(model_path), map_location=device)
    model.eval()
    return model


def _get_latent_dim(model: torch.jit.ScriptModule) -> int:
    """Infer the latent dimension by encoding a short dummy signal."""
    device = next(model.parameters()).device
    dummy = torch.randn(1, 1, 2048, device=device)
    with torch.no_grad():
        z = model.encode(dummy)
    # z shape: [batch, latent_dim, time]
    return z.shape[1]


def _model_device(model: torch.jit.ScriptModule) -> torch.device:
    """Return the device the model parameters live on."""
    return next(model.parameters()).device


def generate_random(
    model: torch.jit.ScriptModule,
    duration: float,
    sr: int = 48000,
    temperature: float = 1.0,
) -> np.ndarray:
    """Generate audio by decoding random latent vectors.

    Samples from N(0, temperature) in the model's latent space and decodes
    into a waveform.

    Returns a 1-D numpy array of audio samples.
    """
    device = _model_device(model)
    latent_dim = _get_latent_dim(model)

    # RAVE's decode expects latent frames; the ratio between audio samples
    # and latent frames is discovered from the dummy encode above.
    dummy = torch.randn(1, 1, 2048, device=device)
    with torch.no_grad():
        z_dummy = model.encode(dummy)
    latent_length = z_dummy.shape[2]
    ratio = 2048 / latent_length

    n_samples = int(duration * sr)
    n_latent_frames = int(n_samples / ratio) + 1

    z = torch.randn(1, latent_dim, n_latent_frames, device=device) * temperature

    with torch.no_grad():
        audio = model.decode(z)

    audio = audio.squeeze().cpu().numpy()
    return audio[:n_samples]


def generate_from_audio(
    model: torch.jit.ScriptModule,
    input_audio: np.ndarray,
    sr: int = 48000,
) -> np.ndarray:
    """Encode audio into the latent space and decode it back.

    Useful as a reconstruction test to evaluate model quality.

    Returns a 1-D numpy array of reconstructed audio.
    """
    device = _model_device(model)
    audio_tensor = torch.from_numpy(input_audio).float().unsqueeze(0).unsqueeze(0)
    audio_tensor = audio_tensor.to(device)

    with torch.no_grad():
        z = model.encode(audio_tensor)
        reconstructed = model.decode(z)

    out = reconstructed.squeeze().cpu().numpy()
    return out[: len(input_audio)]


def interpolate(
    model: torch.jit.ScriptModule,
    audio_a: np.ndarray,
    audio_b: np.ndarray,
    steps: int = 10,
    sr: int = 48000,
) -> list[np.ndarray]:
    """Linearly interpolate between two audio clips in latent space.

    Encodes both clips, creates *steps* evenly spaced interpolations from
    ``audio_a`` to ``audio_b``, and decodes each one.

    Returns a list of 1-D numpy arrays (one per interpolation step).
    """
    device = _model_device(model)

    ta = torch.from_numpy(audio_a).float().unsqueeze(0).unsqueeze(0).to(device)
    tb = torch.from_numpy(audio_b).float().unsqueeze(0).unsqueeze(0).to(device)

    # Pad the shorter tensor to match the longer one
    max_len = max(ta.shape[-1], tb.shape[-1])
    if ta.shape[-1] < max_len:
        ta = torch.nn.functional.pad(ta, (0, max_len - ta.shape[-1]))
    if tb.shape[-1] < max_len:
        tb = torch.nn.functional.pad(tb, (0, max_len - tb.shape[-1]))

    with torch.no_grad():
        z_a = model.encode(ta)
        z_b = model.encode(tb)

    results: list[np.ndarray] = []
    for i in range(steps):
        alpha = i / max(steps - 1, 1)
        z_interp = z_a * (1.0 - alpha) + z_b * alpha

        with torch.no_grad():
            audio = model.decode(z_interp)

        results.append(audio.squeeze().cpu().numpy())

    return results
