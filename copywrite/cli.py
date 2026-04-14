"""CLI entry point for copywrite."""

from __future__ import annotations

import sys
from pathlib import Path

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


@click.group()
@click.version_option(version="0.2.0")
def cli():
    """copywrite - Generate music in the style of reference audio using RAVE and MusicGen."""
    pass


# ---------------------------------------------------------------------------
# setup
# ---------------------------------------------------------------------------

@cli.command()
def setup():
    """Check environment, create directories, verify dependencies."""
    from copywrite.config import load_config

    console.print(Panel(
        "[bold]copywrite setup[/bold]\n"
        "Verify GPU, CUDA, dependencies, and create project directories.",
        title="copywrite", border_style="cyan",
    ))
    config = load_config()

    # GPU check
    console.rule("[cyan]GPU[/cyan]")
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory / 1e9
            console.print(f"  [green]GPU: {name} ({vram:.1f} GB VRAM)[/green]")
        else:
            console.print("  [yellow]No CUDA GPU detected. Training requires EC2.[/yellow]")
    except ImportError:
        console.print("  [red]PyTorch not installed. Run: pip install torch torchaudio[/red]")

    # Dependency checks
    console.rule("[cyan]Dependencies[/cyan]")
    for pkg, label in [
        ("librosa", "librosa (audio analysis)"),
        ("soundfile", "soundfile (audio I/O)"),
    ]:
        try:
            __import__(pkg)
            console.print(f"  [green]{label}[/green]")
        except ImportError:
            console.print(f"  [red]{label} — not installed[/red]")

    for pkg, label, group in [
        ("rave", "acids-rave (RAVE)", "rave"),
        ("audiocraft", "audiocraft (MusicGen)", "musicgen"),
        ("peft", "peft (LoRA)", "musicgen"),
        ("transformers", "transformers", "musicgen"),
    ]:
        try:
            __import__(pkg)
            console.print(f"  [green]{label}[/green]")
        except ImportError:
            console.print(f"  [yellow]{label} — install with: pip install -e \".[{group}]\"[/yellow]")

    # AWS CLI
    console.rule("[cyan]AWS CLI[/cyan]")
    import subprocess
    try:
        r = subprocess.run(["aws", "--version"], capture_output=True, text=True, timeout=10)
        console.print(f"  [green]{r.stdout.strip()}[/green]")
    except (FileNotFoundError, subprocess.TimeoutExpired):
        console.print("  [red]AWS CLI not found — needed for EC2 training[/red]")

    # Directories
    console.rule("[cyan]Directories[/cyan]")
    config.ensure_dirs()
    console.print(f"  [green]Created data directories under {config.data_dir}[/green]")

    # Config
    config.save()
    console.print(f"  [green]Config saved to ~/.copywrite/config.yaml[/green]")

    # Reference audio check
    ref_count = sum(1 for _ in config.reference_dir.rglob("*") if _.suffix.lower() in (".mp3", ".wav", ".flac"))
    if ref_count > 0:
        console.print(f"\n  [green]Found {ref_count} reference audio files[/green]")
    else:
        console.print(f"\n  [yellow]No audio files in {config.reference_dir}/ — place your tracks there[/yellow]")

    console.print(Panel(
        "Next steps:\n"
        "  1. [bold]copywrite prepare[/bold] — preprocess reference audio\n"
        "  2. [bold]copywrite train rave[/bold] — train RAVE on EC2\n"
        "  3. [bold]copywrite train musicgen[/bold] — train MusicGen LoRA on EC2\n"
        "  4. [bold]copywrite generate musicgen -p 'your prompt'[/bold] — generate!",
        title="Done", border_style="green",
    ))


# ---------------------------------------------------------------------------
# prepare
# ---------------------------------------------------------------------------

@cli.command()
@click.option("--input", "-i", "input_dir", type=click.Path(exists=True),
              help="Directory containing reference audio (default: data/reference/)")
@click.option("--augment/--no-augment", default=True, help="Apply data augmentation")
@click.option("--caption/--no-caption", default=True, help="Generate MusicGen captions")
def prepare(input_dir, augment, caption):
    """Preprocess reference audio for RAVE and MusicGen training."""
    from copywrite.config import load_config
    from copywrite.data import prepare_all, augment_directory

    config = load_config()
    config.ensure_dirs()
    inp = Path(input_dir) if input_dir else config.reference_dir

    console.print(Panel(f"Preparing training data from [bold]{inp}[/bold]", title="copywrite prepare"))

    # Step 1: Segment audio
    stats = prepare_all(config, inp)
    console.print(f"  RAVE segments: [bold]{stats.get('rave_count', 0)}[/bold]")
    console.print(f"  MusicGen segments: [bold]{stats.get('musicgen_count', 0)}[/bold]")

    # Step 2: Augment
    if augment:
        console.rule("Augmentation")
        aug_files = augment_directory(
            config.rave_preprocessed_dir,
            config.rave_preprocessed_dir,
            sr=config.rave_sample_rate,
        )
        console.print(f"  Created [bold]{len(aug_files)}[/bold] augmented RAVE segments")

    # Step 3: Caption
    if caption:
        console.rule("Captioning")
        from copywrite.musicgen.captions import caption_all
        captions = caption_all(config)
        console.print(f"  Captioned [bold]{len(captions)}[/bold] MusicGen segments")

        # Step 4: Build dataset
        from copywrite.musicgen.dataset import prepare_dataset
        ds_dir = prepare_dataset(config)
        console.print(f"  Dataset ready at [bold]{ds_dir}[/bold]")

    console.print(Panel("Data preparation complete!", border_style="green"))


# ---------------------------------------------------------------------------
# train
# ---------------------------------------------------------------------------

@cli.group()
def train():
    """Train RAVE or MusicGen models."""
    pass

cli.add_command(train)


@train.command("rave")
@click.option("--resume", is_flag=True, help="Resume from last checkpoint")
@click.option("--export", "do_export", is_flag=True, help="Export model after training")
def train_rave(resume, do_export):
    """Train RAVE on EC2 g5.xlarge."""
    from copywrite.config import load_config
    from copywrite.rave.train import preprocess, train_ec2, export_model

    config = load_config()
    console.print(Panel("Training RAVE on EC2", title="copywrite train rave"))

    # Preprocess locally first (fast)
    console.rule("Preprocessing")
    db_path = preprocess(config)
    console.print(f"  Database at [bold]{db_path}[/bold]")

    # Train on EC2
    console.rule("EC2 Training")
    ckpt_path = train_ec2(config, resume=resume)
    console.print(f"  Checkpoints at [bold]{ckpt_path}[/bold]")

    # Export
    if do_export:
        console.rule("Export")
        model_path = export_model(config, ckpt_path)
        console.print(f"  Model exported to [bold]{model_path}[/bold]")

    console.print(Panel("RAVE training complete!", border_style="green"))


@train.command("musicgen")
@click.option("--local", is_flag=True, help="Train locally instead of EC2")
def train_musicgen_cmd(local):
    """Train MusicGen LoRA on EC2 g5.xlarge."""
    from copywrite.config import load_config
    from copywrite.musicgen.train import train_ec2, train_local
    from copywrite.musicgen.dataset import prepare_dataset

    config = load_config()
    console.print(Panel("Training MusicGen LoRA" + (" (local)" if local else " on EC2"),
                        title="copywrite train musicgen"))

    ds_dir = config.musicgen_dataset_dir
    if not (ds_dir / "metadata.jsonl").exists():
        console.print("[yellow]Dataset not found. Run 'copywrite prepare' first.[/yellow]")
        raise SystemExit(1)

    if local:
        adapter_path = train_local(config, ds_dir)
    else:
        adapter_path = train_ec2(config, ds_dir)

    console.print(f"  Adapter saved to [bold]{adapter_path}[/bold]")
    console.print(Panel("MusicGen training complete!", border_style="green"))


# ---------------------------------------------------------------------------
# generate
# ---------------------------------------------------------------------------

@cli.group()
def generate():
    """Generate audio from trained models."""
    pass

cli.add_command(generate)


@generate.command("musicgen")
@click.option("--prompt", "-p", required=True, help="Text description of desired output")
@click.option("--duration", "-d", type=float, default=None, help="Duration in seconds")
@click.option("--count", "-c", type=int, default=1, help="Number of variations")
@click.option("--temperature", "-t", type=float, default=1.0, help="Sampling temperature")
@click.option("--seed", "-s", type=int, default=None, help="Random seed")
def generate_musicgen(prompt, duration, count, temperature, seed):
    """Generate audio with fine-tuned MusicGen."""
    from copywrite.config import load_config
    from copywrite.musicgen.generate import load_model, generate as mg_generate, batch_generate
    from copywrite.utils.audio import save_audio
    import numpy as np

    config = load_config()
    dur = duration or config.generation_duration

    console.print(Panel(f'Prompt: "{prompt}"\nDuration: {dur}s, Count: {count}',
                        title="copywrite generate musicgen"))

    if seed is not None:
        import torch
        torch.manual_seed(seed)
        np.random.seed(seed)

    model, processor = load_model(config)

    if count == 1:
        audio = mg_generate(model, processor, prompt, duration=dur, temperature=temperature)
        out_path = config.generated_dir / "musicgen_output.wav"
        save_audio(out_path, audio, sr=config.musicgen_sample_rate)
        console.print(f"  [green]Saved: {out_path}[/green]")
    else:
        audios = batch_generate(model, processor, prompt, count=count,
                                duration=dur, temperature=temperature)
        for i, audio in enumerate(audios):
            out_path = config.generated_dir / f"musicgen_output_{i:02d}.wav"
            save_audio(out_path, audio, sr=config.musicgen_sample_rate)
            console.print(f"  [green]Saved: {out_path}[/green]")


@generate.command("rave")
@click.option("--mode", type=click.Choice(["random", "walk", "interpolate"]),
              default="random", help="Generation mode")
@click.option("--duration", "-d", type=float, default=30.0, help="Duration in seconds")
@click.option("--count", "-c", type=int, default=1, help="Number of samples (random mode)")
@click.option("--temperature", "-t", type=float, default=1.0, help="Sampling temperature")
def generate_rave(mode, duration, count, temperature):
    """Generate audio from trained RAVE model."""
    from copywrite.config import load_config
    from copywrite.rave.generate import load_model, generate_random
    from copywrite.rave.explore import random_walk, batch_generate as rave_batch
    from copywrite.utils.audio import save_audio

    config = load_config()
    if not config.rave_model_path or not Path(config.rave_model_path).exists():
        console.print("[red]No RAVE model found. Train first with 'copywrite train rave'[/red]")
        raise SystemExit(1)

    model = load_model(config.rave_model_path)

    if mode == "random":
        audios = rave_batch(model, count, duration, temperature=temperature)
        for i, audio in enumerate(audios):
            out = config.generated_dir / f"rave_random_{i:02d}.wav"
            save_audio(out, audio, sr=config.rave_sample_rate)
            console.print(f"  [green]Saved: {out}[/green]")
    elif mode == "walk":
        audio = random_walk(model, duration)
        out = config.generated_dir / "rave_walk.wav"
        save_audio(out, audio, sr=config.rave_sample_rate)
        console.print(f"  [green]Saved: {out}[/green]")


@generate.command("combined")
@click.option("--prompt", "-p", required=True, help="Text description")
@click.option("--duration", "-d", type=float, default=30.0)
@click.option("--strategy", type=click.Choice(["texture", "seed", "resynth"]),
              default="texture")
def generate_combined(prompt, duration, strategy):
    """Generate using both RAVE + MusicGen."""
    from copywrite.config import load_config
    from copywrite.pipeline.combined import generate_combined as gen_combined
    from copywrite.utils.audio import save_audio

    config = load_config()
    audio = gen_combined(config, prompt, duration=duration, strategy=strategy)
    out = config.generated_dir / "combined_output.wav"
    save_audio(out, audio, sr=config.musicgen_sample_rate)
    console.print(f"  [green]Saved: {out}[/green]")


# ---------------------------------------------------------------------------
# explore
# ---------------------------------------------------------------------------

@cli.command()
@click.option("--dimension", "-d", type=int, default=0, help="Latent dimension to sweep")
@click.option("--steps", type=int, default=10, help="Number of interpolation steps")
@click.option("--duration", type=float, default=4.0, help="Duration per step")
def explore(dimension, steps, duration):
    """Explore RAVE latent space."""
    from copywrite.config import load_config
    from copywrite.rave.generate import load_model
    from copywrite.rave.explore import latent_sweep
    from copywrite.utils.audio import save_audio

    config = load_config()
    if not config.rave_model_path or not Path(config.rave_model_path).exists():
        console.print("[red]No RAVE model found.[/red]")
        raise SystemExit(1)

    model = load_model(config.rave_model_path)
    audios = latent_sweep(model, dimension, steps=steps)
    for i, audio in enumerate(audios):
        out = config.generated_dir / f"explore_dim{dimension}_{i:02d}.wav"
        save_audio(out, audio, sr=config.rave_sample_rate)
    console.print(f"  [green]Saved {len(audios)} samples to {config.generated_dir}[/green]")


# ---------------------------------------------------------------------------
# caption
# ---------------------------------------------------------------------------

@cli.command()
def caption():
    """Generate text captions for MusicGen training segments."""
    from copywrite.config import load_config
    from copywrite.musicgen.captions import caption_all

    config = load_config()
    captions = caption_all(config)
    console.print(f"Captioned [bold]{len(captions)}[/bold] segments → {config.captions_dir}")


# ---------------------------------------------------------------------------
# status
# ---------------------------------------------------------------------------

@cli.command()
def status():
    """Show training status, data stats, and model availability."""
    from copywrite.config import load_config

    config = load_config()
    console.print(Panel("copywrite status", border_style="cyan"))

    table = Table(title="Data")
    table.add_column("Item", style="bold")
    table.add_column("Count / Path")

    # Reference audio
    ref_count = sum(1 for _ in config.reference_dir.rglob("*")
                    if _.suffix.lower() in (".mp3", ".wav", ".flac"))
    table.add_row("Reference tracks", str(ref_count))

    # RAVE segments
    rave_count = sum(1 for _ in config.rave_preprocessed_dir.glob("*.wav")) if config.rave_preprocessed_dir.exists() else 0
    table.add_row("RAVE segments", str(rave_count))

    # MusicGen segments
    mg_count = sum(1 for _ in config.musicgen_preprocessed_dir.glob("*.wav")) if config.musicgen_preprocessed_dir.exists() else 0
    table.add_row("MusicGen segments", str(mg_count))

    # Captions
    cap_count = sum(1 for _ in config.captions_dir.glob("*.txt")) if config.captions_dir.exists() else 0
    table.add_row("Captions", str(cap_count))

    console.print(table)

    # Models
    table2 = Table(title="Models")
    table2.add_column("Model", style="bold")
    table2.add_column("Status")

    # RAVE
    if config.rave_model_path and Path(config.rave_model_path).exists():
        table2.add_row("RAVE", f"[green]Ready[/green] ({config.rave_model_path})")
    elif config.rave_checkpoint_dir.exists() and any(config.rave_checkpoint_dir.rglob("*.ckpt")):
        table2.add_row("RAVE", "[yellow]Checkpoints found (not exported)[/yellow]")
    else:
        table2.add_row("RAVE", "[red]Not trained[/red]")

    # MusicGen
    if config.musicgen_adapter_path and Path(config.musicgen_adapter_path).exists():
        table2.add_row("MusicGen LoRA", f"[green]Ready[/green] ({config.musicgen_adapter_path})")
    elif config.musicgen_checkpoint_dir.exists() and any(config.musicgen_checkpoint_dir.rglob("adapter_*")):
        table2.add_row("MusicGen LoRA", "[yellow]Checkpoint found[/yellow]")
    else:
        table2.add_row("MusicGen LoRA", "[red]Not trained[/red]")

    console.print(table2)

    # Generated
    gen_count = sum(1 for _ in config.generated_dir.glob("*.wav")) if config.generated_dir.exists() else 0
    console.print(f"\nGenerated tracks: [bold]{gen_count}[/bold]")


if __name__ == "__main__":
    cli()
