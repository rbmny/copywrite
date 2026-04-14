"""MusicGen audio generation with optional LoRA adapter."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from transformers import MusicgenForConditionalGeneration, AutoProcessor

from copywrite.config import CopywriteConfig


def load_model(config: CopywriteConfig) -> tuple:
    """Load MusicGen base model with optional LoRA adapter.

    Returns (model, processor).  If config.musicgen_adapter_path is set and
    the directory contains adapter weights, the LoRA adapter is merged on top
    of the base model.  Otherwise the un-fine-tuned base model is returned.
    """
    processor = AutoProcessor.from_pretrained(config.musicgen_base_model)
    model = MusicgenForConditionalGeneration.from_pretrained(
        config.musicgen_base_model
    )

    adapter_path = config.musicgen_adapter_path
    if adapter_path is None:
        adapter_path = config.musicgen_checkpoint_dir

    adapter_config = Path(adapter_path) / "adapter_config.json"
    if adapter_config.exists():
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(adapter_path))
        model = model.merge_and_unload()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    return model, processor


def generate(
    model,
    processor,
    prompt: str,
    duration: float = 15.0,
    temperature: float = 1.0,
    top_k: int = 250,
    guidance_scale: float = 3.0,
) -> np.ndarray:
    """Generate audio from a text prompt.

    Returns a 1-D numpy array at 32 kHz.
    """
    device = next(model.parameters()).device
    inputs = processor(text=[prompt], padding=True, return_tensors="pt").to(device)

    # MusicGen generates at 32 kHz; max_new_tokens controls length
    # ~50 tokens per second of audio for MusicGen
    max_tokens = int(duration * 50)

    with torch.no_grad():
        audio_values = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=temperature,
            top_k=top_k,
            guidance_scale=guidance_scale,
        )

    # audio_values shape: (batch, channels, samples)
    audio = audio_values[0, 0].cpu().numpy()
    return audio


def generate_with_melody(
    model,
    processor,
    prompt: str,
    melody_audio: np.ndarray,
    duration: float = 15.0,
    sr: int = 32000,
) -> np.ndarray:
    """Generate audio conditioned on a melody.

    Uses the melody-conditioned generation path if the model supports it
    (e.g. musicgen-melody).  Falls back to text-only generation otherwise.

    Returns a 1-D numpy array at 32 kHz.
    """
    device = next(model.parameters()).device
    inputs = processor(text=[prompt], padding=True, return_tensors="pt").to(device)

    max_tokens = int(duration * 50)

    # Prepare melody tensor
    melody_tensor = torch.tensor(melody_audio, dtype=torch.float32)
    if melody_tensor.dim() == 1:
        melody_tensor = melody_tensor.unsqueeze(0)  # (1, samples)
    melody_tensor = melody_tensor.unsqueeze(0).to(device)  # (1, 1, samples)

    try:
        with torch.no_grad():
            audio_values = model.generate(
                **inputs,
                audio=melody_tensor,
                max_new_tokens=max_tokens,
            )
    except TypeError:
        # Model doesn't support melody conditioning — fall back
        with torch.no_grad():
            audio_values = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=True,
                guidance_scale=3.0,
            )

    audio = audio_values[0, 0].cpu().numpy()
    return audio


def batch_generate(
    model,
    processor,
    prompt: str,
    count: int = 4,
    duration: float = 15.0,
    temperature: float = 1.0,
) -> list[np.ndarray]:
    """Generate multiple variations from the same prompt.

    Returns a list of 1-D numpy arrays at 32 kHz.
    """
    results: list[np.ndarray] = []
    for _ in range(count):
        audio = generate(
            model, processor, prompt,
            duration=duration,
            temperature=temperature,
        )
        results.append(audio)
    return results
