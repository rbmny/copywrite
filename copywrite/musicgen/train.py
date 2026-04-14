"""MusicGen LoRA fine-tuning (EC2 and local)."""

from __future__ import annotations

import textwrap
from pathlib import Path

from rich.console import Console

from copywrite.config import CopywriteConfig
from copywrite.utils.ec2 import (
    launch_instance,
    wait_for_instance,
    upload_files,
    download_files,
    run_remote_command,
    terminate_instance,
)

console = Console()


# ---------------------------------------------------------------------------
# Self-contained training script (runs on EC2)
# ---------------------------------------------------------------------------

def _generate_training_script(
    config: CopywriteConfig, remote_dataset_dir: str
) -> str:
    """Return a complete, self-contained Python training script.

    The script is written to a file on the EC2 instance and executed there.
    It depends only on pip-installable packages.
    """
    return textwrap.dedent(f"""\
        #!/usr/bin/env python3
        \"\"\"MusicGen LoRA fine-tuning — self-contained EC2 script.\"\"\"

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
        DATASET_DIR = Path("{remote_dataset_dir}").expanduser()
        OUTPUT_DIR = Path.home() / "musicgen_output"
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        BASE_MODEL = "{config.musicgen_base_model}"
        LORA_RANK = {config.musicgen_lora_rank}
        LORA_ALPHA = {config.musicgen_lora_alpha}
        TARGET_MODULES = {config.musicgen_lora_target_modules!r}
        LEARNING_RATE = {config.musicgen_learning_rate}
        TRAIN_STEPS = {config.musicgen_train_steps}
        BATCH_SIZE = 2
        GRADIENT_ACCUMULATION = 8
        SAVE_EVERY = 200
        WARMUP_STEPS = 50
        SAMPLE_RATE = {config.musicgen_sample_rate}
        MAX_AUDIO_LENGTH = int({config.musicgen_segment_duration} * SAMPLE_RATE)

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
            print(f"Loaded {{len(samples)}} training samples")
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
            print(f"Using device: {{device}}")

            # Load model and processor
            print(f"Loading {{BASE_MODEL}} ...")
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

            print(f"Starting training for {{TRAIN_STEPS}} steps ...")
            optimizer.zero_grad()

            while global_step < TRAIN_STEPS:
                # Get next sample (cycle through dataset)
                audio_path, caption = samples[sample_idx % len(samples)]
                sample_idx += 1

                try:
                    audio = load_audio(audio_path)
                except Exception as e:
                    print(f"Skipping {{audio_path}}: {{e}}")
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
                            f"Step {{global_step}}/{{TRAIN_STEPS}} | "
                            f"Loss: {{avg_loss:.4f}} | LR: {{lr:.2e}}"
                        )
                        running_loss = 0.0

                    if global_step % SAVE_EVERY == 0:
                        ckpt_dir = OUTPUT_DIR / f"checkpoint-{{global_step}}"
                        model.save_pretrained(ckpt_dir)
                        print(f"Saved checkpoint: {{ckpt_dir}}")

            # Save final adapter
            model.save_pretrained(OUTPUT_DIR)
            processor.save_pretrained(OUTPUT_DIR)
            print(f"Training complete. Adapter saved to {{OUTPUT_DIR}}")

        if __name__ == "__main__":
            main()
    """)


# ---------------------------------------------------------------------------
# EC2 training
# ---------------------------------------------------------------------------

def train_ec2(config: CopywriteConfig, dataset_dir: Path) -> Path:
    """Fine-tune MusicGen with LoRA on an EC2 GPU instance.

    1. Generate the training script
    2. Launch EC2 instance
    3. Upload dataset + script
    4. Install dependencies and run training
    5. Download trained adapter
    6. Terminate instance

    Returns the path to the local adapter directory.
    """
    console.print("[bold]Starting MusicGen LoRA training on EC2[/bold]")

    # Generate training script
    remote_dataset = "~/musicgen_dataset"
    script_content = _generate_training_script(config, remote_dataset)

    # Write script to a temp file for upload
    script_path = dataset_dir / "train_musicgen.py"
    script_path.write_text(script_content, encoding="utf-8")

    # Launch instance
    instance = launch_instance(
        instance_type=config.ec2_instance_type,
        ami_id=config.ec2_ami_id,
        key_name=config.ec2_key_name,
        security_group=config.ec2_security_group,
        region=config.ec2_region,
    )

    try:
        instance = wait_for_instance(instance.instance_id, config.ec2_region)
        ip = instance.public_ip
        key = config.ec2_key_path

        # Create remote directories
        run_remote_command(ip, key, f"mkdir -p {remote_dataset}")

        # Upload dataset files
        files_to_upload = list(dataset_dir.glob("*.wav")) + [
            dataset_dir / "metadata.jsonl",
            script_path,
        ]
        upload_files(ip, key, files_to_upload, remote_dataset)

        # Install dependencies
        console.print("[bold]Installing dependencies on EC2...[/bold]")
        run_remote_command(
            ip, key,
            "pip install torch torchaudio transformers peft "
            "accelerate datasets audiocraft soundfile",
            timeout=600,
        )

        # Run training
        console.print("[bold]Running training script...[/bold]")
        run_remote_command(
            ip, key,
            f"cd {remote_dataset} && python train_musicgen.py",
            timeout=7200,
        )

        # Download adapter
        config.musicgen_checkpoint_dir.mkdir(parents=True, exist_ok=True)
        download_files(
            ip, key,
            [
                "~/musicgen_output/adapter_config.json",
                "~/musicgen_output/adapter_model.safetensors",
            ],
            config.musicgen_checkpoint_dir,
        )

        console.print(
            f"[green bold]Training complete.[/green bold] "
            f"Adapter saved to {config.musicgen_checkpoint_dir}"
        )
    finally:
        terminate_instance(instance.instance_id, config.ec2_region)

    # Clean up temp script
    script_path.unlink(missing_ok=True)

    return config.musicgen_checkpoint_dir


# ---------------------------------------------------------------------------
# Local training
# ---------------------------------------------------------------------------

def train_local(config: CopywriteConfig, dataset_dir: Path) -> Path:
    """Fine-tune MusicGen with LoRA locally.

    Uses conservative settings for 12 GB VRAM: batch size 1, gradient
    accumulation 16, gradient checkpointing, fp16.

    Returns the path to the adapter directory.
    """
    import json
    import math

    import torch
    import torchaudio
    from peft import LoraConfig, get_peft_model, TaskType
    from transformers import (
        MusicgenForConditionalGeneration,
        AutoProcessor,
        get_cosine_schedule_with_warmup,
    )

    console.print("[bold]Starting local MusicGen LoRA training[/bold]")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    console.print(f"Device: {device}")

    # Load model
    processor = AutoProcessor.from_pretrained(config.musicgen_base_model)
    model = MusicgenForConditionalGeneration.from_pretrained(
        config.musicgen_base_model
    )

    lora_config = LoraConfig(
        r=config.musicgen_lora_rank,
        lora_alpha=config.musicgen_lora_alpha,
        target_modules=config.musicgen_lora_target_modules,
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    model.to(device)
    model.gradient_checkpointing_enable()

    # Load dataset
    metadata_path = dataset_dir / "metadata.jsonl"
    samples: list[tuple[str, str]] = []
    for line in metadata_path.read_text(encoding="utf-8").strip().splitlines():
        entry = json.loads(line)
        wav_path = dataset_dir / entry["file_name"]
        if wav_path.exists():
            samples.append((str(wav_path), entry["text"]))

    if not samples:
        raise RuntimeError("No training samples found in dataset")

    console.print(f"Loaded {len(samples)} training samples")

    max_audio_length = int(config.musicgen_segment_duration * config.musicgen_sample_rate)
    batch_size = 1
    grad_accum = 16
    warmup_steps = 50

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.musicgen_learning_rate, weight_decay=0.01
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=config.musicgen_train_steps,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    output_dir = config.musicgen_checkpoint_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    model.train()
    global_step = 0
    sample_idx = 0
    accumulated = 0
    running_loss = 0.0
    optimizer.zero_grad()

    from rich.progress import Progress

    with Progress() as progress:
        task = progress.add_task("Training", total=config.musicgen_train_steps)

        while global_step < config.musicgen_train_steps:
            audio_path, caption = samples[sample_idx % len(samples)]
            sample_idx += 1

            try:
                waveform, sr = torchaudio.load(audio_path)
                if waveform.shape[0] > 1:
                    waveform = waveform.mean(dim=0, keepdim=True)
                if sr != config.musicgen_sample_rate:
                    waveform = torchaudio.functional.resample(
                        waveform, sr, config.musicgen_sample_rate
                    )
                if waveform.shape[1] > max_audio_length:
                    waveform = waveform[:, :max_audio_length]
                elif waveform.shape[1] < max_audio_length:
                    pad = max_audio_length - waveform.shape[1]
                    waveform = torch.nn.functional.pad(waveform, (0, pad))
                audio = waveform.squeeze(0)
            except Exception as e:
                console.print(f"[yellow]Skipping {audio_path}: {e}[/yellow]")
                continue

            inputs = processor(
                text=[caption], padding=True, return_tensors="pt"
            ).to(device)

            audio_tensor = audio.unsqueeze(0).to(device)
            with torch.no_grad():
                encoder_outputs = model.audio_encoder.encode(
                    audio_tensor.unsqueeze(1), bandwidth=6.0
                )
                labels = encoder_outputs.audio_codes.squeeze(0)

            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                outputs = model(
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    labels=labels,
                )
                loss = outputs.loss / grad_accum

            scaler.scale(loss).backward()
            accumulated += 1
            running_loss += loss.item() * grad_accum

            if accumulated >= grad_accum:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
                accumulated = 0
                global_step += 1
                progress.advance(task)

                if global_step % 200 == 0:
                    ckpt_dir = output_dir / f"checkpoint-{global_step}"
                    model.save_pretrained(ckpt_dir)

    # Save final adapter
    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)
    console.print(
        f"[green bold]Training complete.[/green bold] "
        f"Adapter saved to {output_dir}"
    )
    return output_dir


# ---------------------------------------------------------------------------
# Status check
# ---------------------------------------------------------------------------

def get_training_status(config: CopywriteConfig) -> dict:
    """Check the checkpoint directory for training artifacts.

    Returns {"adapter_exists": bool, "latest_checkpoint": str | None,
             "step_count": int}.
    """
    ckpt_dir = config.musicgen_checkpoint_dir
    adapter_exists = (ckpt_dir / "adapter_config.json").exists()

    # Find numbered checkpoints
    checkpoints = sorted(
        (d for d in ckpt_dir.iterdir() if d.is_dir() and d.name.startswith("checkpoint-")),
        key=lambda p: int(p.name.split("-")[1]),
    ) if ckpt_dir.exists() else []

    latest = checkpoints[-1].name if checkpoints else None
    step_count = int(latest.split("-")[1]) if latest else 0

    return {
        "adapter_exists": adapter_exists,
        "latest_checkpoint": latest,
        "step_count": step_count,
    }
