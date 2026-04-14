#!/usr/bin/env python3
"""Generate 'Neverender' on EC2 after training completes."""

import torch
import soundfile as sf
from pathlib import Path

print("Loading model...")
from transformers import MusicgenForConditionalGeneration, AutoProcessor
from peft import PeftModel

model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-medium")
adapter_path = Path.home() / "musicgen_output" / "final_adapter"
if adapter_path.exists():
    print(f"Loading LoRA adapter from {adapter_path}")
    model = PeftModel.from_pretrained(model, str(adapter_path))
    model = model.merge_and_unload()
else:
    print("WARNING: No adapter found, using base model")

model.to("cuda").eval()
processor = AutoProcessor.from_pretrained("facebook/musicgen-medium")
print(f"Model loaded on GPU ({torch.cuda.get_device_name(0)})")

# Generate 3 variations of "Neverender"
prompts = [
    "Daft Punk style french filter house, neverender, driving distorted bassline, "
    "vocoder melody, 124 BPM, heavy sidechain compression, analog synths, "
    "resonant filter sweep build, four on the floor kick, 12-bit grit, "
    "repetitive hypnotic groove, electronic dance music",

    "Daft Punk Homework era, neverender, acid house bassline, "
    "TB-303 squelch, TR-909 drums, deep lowpass filter sweeps, "
    "123 BPM, minor key, dark underground club, compressed pumping mix, "
    "raw analog distortion, french touch",

    "Daft Punk Discovery style, neverender, disco sample chops, "
    "vocoder robot vocals, funky filtered synths, 122 BPM, "
    "warm Rhodes chords, sidechain pumping bass, uplifting build, "
    "four on the floor, retro futuristic electronic",
]

output_dir = Path.home() / "neverender_output"
output_dir.mkdir(exist_ok=True)

for i, prompt in enumerate(prompts):
    print(f"\n=== Generating variation {i+1}/3 ===")
    print(f"  Prompt: {prompt[:80]}...")

    inputs = processor(text=[prompt], padding=True, return_tensors="pt").to("cuda")

    with torch.no_grad():
        audio = model.generate(
            **inputs,
            max_new_tokens=int(90 * 50),  # 90 seconds
            do_sample=True,
            temperature=0.9,
            top_k=250,
            guidance_scale=3.5,
        )

    out_path = output_dir / f"neverender_v{i+1}.wav"
    sf.write(str(out_path), audio[0, 0].cpu().numpy(), 32000)
    print(f"  Saved: {out_path}")

print("\n=== All 3 variations generated! ===")
