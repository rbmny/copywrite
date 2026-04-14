"""CLI entry point for copywrite."""

from __future__ import annotations

import platform
import shutil
import subprocess
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

console = Console()


@click.group()
@click.version_option(version="0.1.0")
def cli():
    """copywrite - Reverse-engineer production styles and generate new tracks."""
    pass


# ---------------------------------------------------------------------------
# setup
# ---------------------------------------------------------------------------

@cli.command()
def setup():
    """Set up SuperCollider and project environment."""
    from copywrite.config import load_config

    console.print(Panel(
        "[bold]copywrite setup[/bold]\n"
        "Verify SuperCollider, compile SynthDefs, create project directories.",
        title="copywrite",
        border_style="cyan",
    ))

    config = load_config()

    # -- 1. Locate SuperCollider --
    console.rule("[cyan]SuperCollider[/cyan]")
    sc_path = config.supercollider_path
    sclang = config.sclang_path
    scsynth = config.scsynth_path

    found = sclang.exists() and scsynth.exists()

    if not found and sc_path:
        console.print(f"[yellow]Configured path not valid:[/yellow] {sc_path}")

    if not found:
        # Try common locations / PATH
        for name in ("sclang", "sclang.exe"):
            which = shutil.which(name)
            if which:
                sc_path = str(Path(which).parent)
                config.supercollider_path = sc_path
                sclang = config.sclang_path
                scsynth = config.scsynth_path
                found = sclang.exists() and scsynth.exists()
                break

    if not found:
        console.print("[red]SuperCollider not found.[/red]")
        system = platform.system()
        if system == "Windows":
            console.print(
                "  Install from https://supercollider.github.io/downloads\n"
                "  Then re-run [bold]copywrite setup[/bold]."
            )
        elif system == "Darwin":
            console.print(
                "  Install via Homebrew:  [bold]brew install supercollider[/bold]\n"
                "  Or download from https://supercollider.github.io/downloads\n"
                "  Then re-run [bold]copywrite setup[/bold]."
            )
        else:
            console.print(
                "  Install via your package manager, e.g.:\n"
                "    [bold]sudo apt install supercollider[/bold]\n"
                "    [bold]sudo dnf install supercollider[/bold]\n"
                "  Then re-run [bold]copywrite setup[/bold]."
            )
        console.print()
        console.print(
            "[dim]After installing, set the path in "
            "~/.copywrite/config.yaml (supercollider_path) "
            "if it is not on your PATH.[/dim]"
        )
        # Continue with remaining steps even without SC
    else:
        console.print(f"[green]sclang:[/green]  {sclang}")
        console.print(f"[green]scsynth:[/green] {scsynth}")

        # Quick sanity check
        try:
            subprocess.run(
                [str(sclang), "-h"],
                capture_output=True,
                timeout=10,
            )
            console.print("[green]sclang executes successfully.[/green]")
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as exc:
            console.print(f"[yellow]sclang test returned an error (may still work): {exc}[/yellow]")

    # -- 2. Compile SynthDefs --
    console.rule("[cyan]SynthDefs[/cyan]")
    if found:
        try:
            from copywrite.engine import SynthDefManager

            mgr = SynthDefManager(config)
            mgr.compile()
            names = mgr.list_synthdefs()
            console.print(f"[green]{len(names)} SynthDefs compiled:[/green] {', '.join(names)}")
        except Exception as exc:
            console.print(f"[red]SynthDef compilation failed:[/red] {exc}")
    else:
        console.print("[yellow]Skipping SynthDef compilation (SuperCollider not found).[/yellow]")

    # -- 3. Create config file --
    console.rule("[cyan]Configuration[/cyan]")
    config.save()
    console.print(f"[green]Config saved to[/green] {Path.home() / '.copywrite' / 'config.yaml'}")

    # -- 4. Create data directories --
    console.rule("[cyan]Directories[/cyan]")
    config.ensure_dirs()
    for d in [config.reference_dir, config.transcriptions_dir,
              config.style_model_dir, config.generated_dir]:
        console.print(f"  [green]+[/green] {d}")

    # -- Summary --
    console.print()
    console.print(Panel(
        "[green]Setup complete.[/green]\n\n"
        "Next steps:\n"
        "  1. Place reference audio in [bold]data/reference/[/bold]\n"
        "  2. Run [bold]copywrite transcribe[/bold]",
        title="Done",
        border_style="green",
    ))


# ---------------------------------------------------------------------------
# transcribe
# ---------------------------------------------------------------------------

@cli.command()
@click.option('--input', '-i', 'input_dir', type=click.Path(exists=True),
              help='Directory containing reference audio files')
@click.option('--max-iterations', '-n', type=int, default=None,
              help='Max transcription iterations per track')
@click.option('--threshold', '-t', type=float, default=None,
              help='Score threshold to accept transcription')
def transcribe(input_dir, max_iterations, threshold):
    """Transcribe reference tracks to SuperCollider code."""
    from copywrite.config import load_config

    config = load_config()
    config.ensure_dirs()

    if max_iterations is not None:
        config.transcribe_max_iterations = max_iterations
    if threshold is not None:
        config.transcribe_score_threshold = threshold

    ref_dir = Path(input_dir) if input_dir else config.reference_dir

    audio_exts = {".wav", ".mp3", ".flac", ".aiff", ".aif", ".ogg"}
    tracks = sorted(
        p for p in ref_dir.iterdir()
        if p.is_file() and p.suffix.lower() in audio_exts
    )

    if not tracks:
        console.print(f"[red]No audio files found in {ref_dir}[/red]")
        console.print("Supported formats: " + ", ".join(sorted(audio_exts)))
        raise SystemExit(1)

    console.print(Panel(
        f"[bold]Transcribing {len(tracks)} track(s)[/bold]\n"
        f"Source: {ref_dir}\n"
        f"Max iterations: {config.transcribe_max_iterations}\n"
        f"Score threshold: {config.transcribe_score_threshold}",
        title="copywrite transcribe",
        border_style="cyan",
    ))

    from copywrite.transcriber import transcription_loop

    results = []
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        console=console,
    ) as progress:
        task = progress.add_task("Transcribing", total=len(tracks))

        for track_path in tracks:
            progress.update(task, description=f"Transcribing {track_path.name}")
            try:
                result = transcription_loop(track_path, config)
                results.append(result)
                score = result.get("best_score", 0)
                iters = result.get("iterations", 0)
                console.print(
                    f"  [green]{track_path.name}[/green] "
                    f"score={score:.3f} iterations={iters}"
                )
            except Exception as exc:
                console.print(f"  [red]{track_path.name}: {exc}[/red]")
            progress.advance(task)

    # Summary
    console.print()
    table = Table(title="Transcription Results")
    table.add_column("Track", style="bold")
    table.add_column("Score", justify="right")
    table.add_column("Iterations", justify="right")
    table.add_column("Status")

    for r in results:
        score = r.get("best_score", 0)
        iters = r.get("iterations", 0)
        analysis = r.get("analysis", {})
        name = Path(analysis.get("file_path", "unknown")).name if isinstance(analysis, dict) else "unknown"
        status = "[green]passed[/green]" if score >= config.transcribe_score_threshold else "[yellow]below threshold[/yellow]"
        table.add_row(name, f"{score:.3f}", str(iters), status)

    console.print(table)
    console.print(f"\nResults saved to {config.transcriptions_dir}")


# ---------------------------------------------------------------------------
# extract
# ---------------------------------------------------------------------------

@cli.command()
@click.option('--name', '-n', default='style_model',
              help='Name for the style model')
def extract(name):
    """Extract style model from transcriptions."""
    from copywrite.config import load_config

    config = load_config()

    # Validate: transcriptions must exist
    if not config.transcriptions_dir.exists() or not any(config.transcriptions_dir.iterdir()):
        console.print("[red]No transcriptions found.[/red]")
        console.print("Run [bold]copywrite transcribe[/bold] first.")
        raise SystemExit(1)

    console.print(Panel(
        f"[bold]Extracting style model[/bold]\n"
        f"Source: {config.transcriptions_dir}\n"
        f"Name: {name}",
        title="copywrite extract",
        border_style="cyan",
    ))

    from copywrite.extractor import extract_style_model

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        progress.add_task("Extracting style model...", total=None)
        model = extract_style_model(config.transcriptions_dir)

    # Save model
    config.style_model_dir.mkdir(parents=True, exist_ok=True)
    model_path = config.style_model_dir / f"{name}.json"
    model.save(model_path)
    console.print(f"[green]Style model saved to {model_path}[/green]")

    # Save report
    report_path = config.style_model_dir / f"{name}_report.md"
    report_path.write_text(model.report(), encoding="utf-8")
    console.print(f"[green]Report saved to {report_path}[/green]")

    # Display summary
    console.print()
    console.print(Panel(model.report(), title="Style Model Summary", border_style="green"))


# ---------------------------------------------------------------------------
# generate
# ---------------------------------------------------------------------------

@cli.command()
@click.option('--count', '-c', type=int, default=None,
              help='Number of tracks to generate')
@click.option('--duration', '-d', type=float, default=None,
              help='Duration in seconds')
@click.option('--bpm', type=int, default=None,
              help='Override BPM (otherwise sampled from style model)')
@click.option('--key', '-k', type=str, default=None,
              help='Override key (e.g., Cm, F#m)')
@click.option('--seed', '-s', type=int, default=None,
              help='Random seed for reproducibility')
def generate(count, duration, bpm, key, seed):
    """Generate new tracks from the style model."""
    from copywrite.config import load_config

    config = load_config()
    config.ensure_dirs()

    count = count or config.default_count
    duration = duration or config.default_duration

    # Load style model
    model_path = config.style_model_dir / "style_model.json"
    if not model_path.exists():
        # Try any .json in the dir
        json_files = list(config.style_model_dir.glob("*.json"))
        if not json_files:
            console.print("[red]No style model found.[/red]")
            console.print("Run [bold]copywrite extract[/bold] first.")
            raise SystemExit(1)
        model_path = json_files[0]

    console.print(Panel(
        f"[bold]Generating {count} track(s)[/bold]\n"
        f"Style model: {model_path.name}\n"
        f"Duration: {duration}s\n"
        f"BPM: {bpm or 'from model'}\n"
        f"Key: {key or 'from model'}\n"
        f"Seed: {seed or 'random'}",
        title="copywrite generate",
        border_style="cyan",
    ))

    from copywrite.extractor import StyleModel
    from copywrite.generator import generate_and_render

    model = StyleModel.load(model_path) if hasattr(StyleModel, 'load') else None
    if model is None:
        import json
        with open(model_path) as f:
            model_data = json.load(f)
        # Try constructing from data
        from copywrite.extractor import extract_style_model
        console.print("[yellow]Falling back to re-extraction.[/yellow]")
        model = extract_style_model(config.transcriptions_dir)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        console=console,
    ) as progress:
        task = progress.add_task("Generating tracks", total=count)

        tracks = generate_and_render(
            style_model=model,
            config=config,
            count=count,
            duration=duration,
            bpm=bpm,
            key=key,
        )

        progress.update(task, completed=count)

    # Summary
    console.print()
    table = Table(title="Generated Tracks")
    table.add_column("Track", style="bold")
    table.add_column("File")
    table.add_column("Duration", justify="right")

    for t in tracks:
        title = getattr(t, 'title', None) or "untitled"
        output = getattr(t, 'output_path', None) or getattr(t, 'wav_path', None) or ""
        dur = getattr(t, 'duration', duration)
        table.add_row(title, str(output), f"{dur}s")

    console.print(table)
    console.print(f"\nTracks saved to {config.generated_dir}")


# ---------------------------------------------------------------------------
# run (full pipeline)
# ---------------------------------------------------------------------------

@cli.command()
@click.option('--input', '-i', 'input_dir', type=click.Path(exists=True),
              help='Directory containing reference audio files')
@click.option('--count', '-c', type=int, default=None,
              help='Number of tracks to generate')
@click.option('--duration', '-d', type=float, default=None,
              help='Duration in seconds')
def run(input_dir, count, duration):
    """Run the full pipeline: transcribe -> extract -> generate."""
    from copywrite.config import load_config

    config = load_config()
    config.ensure_dirs()

    count = count or config.default_count
    duration = duration or config.default_duration

    ref_dir = Path(input_dir) if input_dir else config.reference_dir
    audio_exts = {".wav", ".mp3", ".flac", ".aiff", ".aif", ".ogg"}
    tracks = sorted(
        p for p in ref_dir.iterdir()
        if p.is_file() and p.suffix.lower() in audio_exts
    )

    if not tracks:
        console.print(f"[red]No audio files found in {ref_dir}[/red]")
        raise SystemExit(1)

    console.print(Panel(
        f"[bold]Full pipeline[/bold]\n"
        f"Reference tracks: {len(tracks)} in {ref_dir}\n"
        f"Generate: {count} track(s) x {duration}s",
        title="copywrite run",
        border_style="cyan",
    ))

    # -- Step 1: Transcribe --
    console.rule("[cyan]Step 1/3: Transcribe[/cyan]")
    from copywrite.transcriber import transcription_loop

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        console=console,
    ) as progress:
        task = progress.add_task("Transcribing", total=len(tracks))
        for track_path in tracks:
            progress.update(task, description=f"Transcribing {track_path.name}")
            try:
                result = transcription_loop(track_path, config)
                score = result.get("best_score", 0)
                console.print(
                    f"  [green]{track_path.name}[/green] score={score:.3f}"
                )
            except Exception as exc:
                console.print(f"  [red]{track_path.name}: {exc}[/red]")
            progress.advance(task)

    # -- Step 2: Extract --
    console.rule("[cyan]Step 2/3: Extract style model[/cyan]")
    from copywrite.extractor import extract_style_model

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        progress.add_task("Extracting...", total=None)
        model = extract_style_model(config.transcriptions_dir)

    model_path = config.style_model_dir / "style_model.json"
    config.style_model_dir.mkdir(parents=True, exist_ok=True)
    model.save(model_path)
    report_path = config.style_model_dir / "style_model_report.md"
    report_path.write_text(model.report(), encoding="utf-8")
    console.print(f"[green]Style model saved to {model_path}[/green]")

    # -- Step 3: Generate --
    console.rule("[cyan]Step 3/3: Generate[/cyan]")
    from copywrite.generator import generate_and_render

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        console=console,
    ) as progress:
        task = progress.add_task("Generating", total=count)
        generated = generate_and_render(
            style_model=model,
            config=config,
            count=count,
            duration=duration,
        )
        progress.update(task, completed=count)

    # Summary
    console.print()
    console.print(Panel(
        f"[green]Pipeline complete.[/green]\n\n"
        f"  Transcribed: {len(tracks)} track(s)\n"
        f"  Style model: {model_path}\n"
        f"  Generated:   {len(generated)} track(s) in {config.generated_dir}",
        title="Done",
        border_style="green",
    ))


# ---------------------------------------------------------------------------
# inspect
# ---------------------------------------------------------------------------

@cli.command()
def inspect():
    """Display the current style model."""
    from copywrite.config import load_config

    config = load_config()

    model_path = config.style_model_dir / "style_model.json"
    if not model_path.exists():
        json_files = list(config.style_model_dir.glob("*.json"))
        if not json_files:
            console.print("[red]No style model found.[/red]")
            console.print("Run [bold]copywrite extract[/bold] first.")
            raise SystemExit(1)
        model_path = json_files[0]

    console.print(Panel(
        f"[bold]Style Model[/bold]\n{model_path}",
        title="copywrite inspect",
        border_style="cyan",
    ))

    import json
    with open(model_path) as f:
        data = json.load(f)

    # Display key parameters in a table
    table = Table(title="Style Parameters", show_lines=True)
    table.add_column("Parameter", style="bold cyan")
    table.add_column("Value")

    def _add_flat(prefix: str, obj: dict | list, table: Table) -> None:
        if isinstance(obj, dict):
            for k, v in obj.items():
                full_key = f"{prefix}.{k}" if prefix else k
                if isinstance(v, dict):
                    _add_flat(full_key, v, table)
                elif isinstance(v, list) and len(v) > 8:
                    table.add_row(full_key, f"[{len(v)} items]")
                else:
                    table.add_row(full_key, str(v))

    _add_flat("", data, table)
    console.print(table)

    # Also show the markdown report if it exists
    report_path = model_path.with_name(model_path.stem + "_report.md")
    if report_path.exists():
        console.print()
        console.print(Panel(
            report_path.read_text(encoding="utf-8"),
            title="Style Model Report",
            border_style="green",
        ))


if __name__ == '__main__':
    cli()
