#!/usr/bin/env python3
"""MusicGen LoRA fine-tuning - self-contained EC2 script."""

import json
import math
import os
from pathlib import Path

import torch
import torchaudio
from peft import LoraConfig, get_peft_model, TaskType
from transformers import (
    MusicgenForConditionalGeneration,
    AutoProcessor,
    get_cosine_schedule_with_warmup,
)

# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------
DATASET_DIR = Path("~/musicgen_dataset").expanduser()
OUTPUT_DIR = Path.home() / "musicgen_output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BASE_MODEL = "facebook/musicgen-medium"
LORA_RANK = 16
LORA_ALPHA = 32
TARGET_MODULES = ['q_proj', 'v_proj']
LEARNING_RATE = 0.0001
TRAIN_STEPS = 1000
BATCH_SIZE = 2
GRADIENT_ACCUMULATION = 8
SAVE_EVERY = 200
WARMUP_STEPS = 50
SAMPLE_RATE = 32000
MAX_AUDIO_LENGTH = int(20.0 * SAMPLE_RATE)

# -------------------------------------------------------------------
# Load dataset
# -------------------------------------------------------------------
def load_dataset():
    metadata_path = DATASET_DIR / "metadata.jsonl"
    samples = []
    for line in metadata_path.read_text().strip().splitlines():
        entry = json.loads(line)
        wav_path = DATASET_DIR / entry["file_name"]
        if wav_path.exists():
            samples.append((str(wav_path), entry["text"]))
    print(f"Loaded {len(samples)} training samples")
    return samples

# -------------------------------------------------------------------
# Load and prepare audio
# -------------------------------------------------------------------
def load_audio(path: str):
    waveform, sr = torchaudio.load(path)
    # Mix to mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    # Resample if needed
    if sr != SAMPLE_RATE:
        waveform = torchaudio.functional.resample(waveform, sr, SAMPLE_RATE)
    # Truncate or pad to fixed length
    if waveform.shape[1] > MAX_AUDIO_LENGTH:
        waveform = waveform[:, :MAX_AUDIO_LENGTH]
    elif waveform.shape[1] < MAX_AUDIO_LENGTH:
        pad = MAX_AUDIO_LENGTH - waveform.shape[1]
        waveform = torch.nn.functional.pad(waveform, (0, pad))
    return waveform.squeeze(0)

# -------------------------------------------------------------------
# Training loop
# -------------------------------------------------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model and processor
    print(f"Loading {BASE_MODEL} ...")
    processor = AutoProcessor.from_pretrained(BASE_MODEL)
    model = MusicgenForConditionalGeneration.from_pretrained(BASE_MODEL)

    # Apply LoRA
    lora_config = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        target_modules=TARGET_MODULES,
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    model.to(device)

    # Enable gradient checkpointing
    model.gradient_checkpointing_enable()

    # Load data
    samples = load_dataset()
    if not samples:
        raise RuntimeError("No training samples found")

    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=0.01
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=WARMUP_STEPS,
        num_training_steps=TRAIN_STEPS,
    )

    # Mixed precision scaler
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    # Training
    model.train()
    global_step = 0
    sample_idx = 0
    accumulated = 0
    running_loss = 0.0

    print(f"Starting training for {TRAIN_STEPS} steps ...")
    optimizer.zero_grad()

    while global_step < TRAIN_STEPS:
        # Get next sample (cycle through dataset)
        audio_path, caption = samples[sample_idx % len(samples)]
        sample_idx += 1

        try:
            audio = load_audio(audio_path)
        except Exception as e:
            print(f"Skipping {audio_path}: {e}")
            continue

        # Tokenize caption
        inputs = processor(
            text=[caption],
            padding=True,
            return_tensors="pt",
        ).to(device)

        # Encode audio to codes via the audio encoder
        audio_tensor = audio.unsqueeze(0).to(device)  # (1, samples)
        with torch.no_grad():
            # Encode audio to discrete codes
            audio_input = audio_tensor.unsqueeze(1)  # (1, 1, samples)
            encoder_outputs = model.audio_encoder.encode(
                audio_input, bandwidth=2.2
            )
            audio_codes = encoder_outputs.audio_codes  # (1, 1, codebooks, seq)
            audio_codes = audio_codes.squeeze(1)  # (1, codebooks, seq)

            # Build decoder_input_ids: shift codes right with BOS
            # MusicGen decoder expects: (batch, codebooks, seq)
            num_codebooks = audio_codes.shape[1]
            bos = torch.full(
                (1, num_codebooks, 1),
                model.config.decoder.pad_token_id,
                dtype=audio_codes.dtype,
                device=device,
            )
            decoder_input_ids = torch.cat([bos, audio_codes[:, :, :-1]], dim=-1)
            labels = audio_codes.clone()

        # Forward pass with mixed precision
        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            outputs = model.decoder(
                input_ids=decoder_input_ids,
                encoder_hidden_states=model.enc_to_dec_proj(
                    model.text_encoder(
                        input_ids=inputs.input_ids,
                        attention_mask=inputs.attention_mask,
                    ).last_hidden_state
                ),
                encoder_attention_mask=inputs.attention_mask,
            )
            # Compute cross-entropy loss per codebook
            logits = outputs.logits  # (batch, codebooks, seq, vocab)
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1),
                ignore_index=model.config.decoder.pad_token_id,
            ) / GRADIENT_ACCUMULATION

        scaler.scale(loss).backward()
        accumulated += 1
        running_loss += loss.item() * GRADIENT_ACCUMULATION

        if accumulated >= GRADIENT_ACCUMULATION:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()
            accumulated = 0
            global_step += 1

            if global_step % 10 == 0:
                avg_loss = running_loss / 10
                lr = scheduler.get_last_lr()[0]
                print(
                    f"Step {global_step}/{TRAIN_STEPS} | "
                    f"Loss: {avg_loss:.4f} | LR: {lr:.2e}"
                )
                running_loss = 0.0

            if global_step % SAVE_EVERY == 0:
                ckpt_dir = OUTPUT_DIR / f"checkpoint-{global_step}"
                model.save_pretrained(ckpt_dir)
                print(f"Saved checkpoint: {ckpt_dir}")

    # Save final adapter
    model.save_pretrained(OUTPUT_DIR)
    processor.save_pretrained(OUTPUT_DIR)
    print(f"Training complete. Adapter saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
