"""RAVE preprocessing, EC2 training, and model export."""

from __future__ import annotations

import subprocess
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


def preprocess(config: CopywriteConfig) -> Path:
    """Run ``rave preprocess`` on local WAV files.

    Reads audio from *config.rave_preprocessed_dir* and writes a Lightning
    database to ``config.rave_checkpoint_dir / "rave_db"``.

    Returns the path to the preprocessed database directory.
    """
    input_path = config.rave_preprocessed_dir
    output_path = config.rave_checkpoint_dir / "rave_db"
    output_path.mkdir(parents=True, exist_ok=True)

    console.print(
        f"[bold]Preprocessing[/bold] RAVE data\n"
        f"  input:  {input_path}\n"
        f"  output: {output_path}"
    )

    cmd = [
        "rave", "preprocess",
        "--input_path", str(input_path),
        "--output_path", str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"rave preprocess failed (exit {result.returncode}):\n"
            f"{result.stderr.strip()}"
        )

    console.print(f"[green]Preprocessing complete:[/green] {output_path}")
    return output_path


def train_ec2(config: CopywriteConfig, resume: bool = False) -> Path:
    """Train a RAVE model on an EC2 g5.xlarge instance.

    Steps:
        1. Launch EC2 instance
        2. Upload preprocessed data
        3. Install dependencies (acids-rave, ffmpeg)
        4. Run ``rave train``
        5. Download checkpoints
        6. Terminate instance

    Returns the local checkpoint directory.
    """
    db_path = config.rave_checkpoint_dir / "rave_db"
    if not db_path.exists():
        raise RuntimeError(
            f"Preprocessed database not found at {db_path}. "
            "Run preprocess() first."
        )

    instance = None
    try:
        # 1. Launch
        console.rule("[cyan]Launching EC2 instance[/cyan]")
        instance = launch_instance(
            instance_type=config.ec2_instance_type,
            ami_id=config.ec2_ami_id,
            key_name=config.ec2_key_name,
            security_group=config.ec2_security_group,
            region=config.ec2_region,
        )
        instance = wait_for_instance(instance.instance_id, config.ec2_region)

        ip = instance.public_ip
        key = config.ec2_key_path

        # 2. Upload preprocessed data
        console.rule("[cyan]Uploading preprocessed data[/cyan]")
        db_files = list(db_path.iterdir())
        run_remote_command(ip, key, "mkdir -p ~/rave_db")
        upload_files(ip, key, db_files, "~/rave_db")

        # 3. Install dependencies
        console.rule("[cyan]Installing dependencies[/cyan]")
        run_remote_command(
            ip, key,
            "sudo apt-get update -qq && sudo apt-get install -y -qq ffmpeg "
            "&& pip install acids-rave",
            timeout=600,
        )

        # 4. Run training
        console.rule("[cyan]Starting RAVE training[/cyan]")
        train_cmd = (
            f"rave train"
            f" --config {config.rave_config}"
            f" --db_path ~/rave_db"
            f" --out_path ~/rave_output"
            f" --name copywrite"
            f" --batch {config.rave_batch_size}"
            f" --gpu 0"
        )
        if resume:
            train_cmd += " --ckpt ~/rave_output/copywrite"

        console.print(f"[bold]Training command:[/bold] {train_cmd}")
        console.print("[dim]Training will take several hours...[/dim]")
        run_remote_command(ip, key, train_cmd, timeout=86400)

        # 5. Download checkpoints
        console.rule("[cyan]Downloading checkpoints[/cyan]")
        config.rave_checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # List remote checkpoint files
        ls_output = run_remote_command(
            ip, key, "find ~/rave_output -type f -name '*.ckpt' -o -name '*.yaml'"
        )
        remote_files = [
            line.strip() for line in ls_output.splitlines() if line.strip()
        ]

        if remote_files:
            download_files(ip, key, remote_files, config.rave_checkpoint_dir)
            console.print(
                f"[green]Downloaded {len(remote_files)} checkpoint file(s)[/green]"
            )
        else:
            console.print("[yellow]No checkpoint files found on remote.[/yellow]")

    finally:
        # 6. Terminate
        if instance is not None:
            console.rule("[cyan]Terminating EC2 instance[/cyan]")
            terminate_instance(instance.instance_id, config.ec2_region)

    console.print(f"[green]Training complete.[/green] Checkpoints: {config.rave_checkpoint_dir}")
    return config.rave_checkpoint_dir


def export_model(
    config: CopywriteConfig,
    run_path: Path | None = None,
    streaming: bool = True,
) -> Path:
    """Export a trained RAVE checkpoint to a TorchScript ``.ts`` file.

    If *run_path* is not provided, the latest run directory inside
    ``config.rave_checkpoint_dir`` is used.

    Returns the path to the exported ``.ts`` model.
    """
    if run_path is None:
        # Find the latest run directory (rave train creates timestamped dirs)
        candidates = sorted(
            (p for p in config.rave_checkpoint_dir.iterdir() if p.is_dir()),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if not candidates:
            raise RuntimeError(
                f"No run directories found in {config.rave_checkpoint_dir}. "
                "Train a model first."
            )
        run_path = candidates[0]

    console.print(
        f"[bold]Exporting[/bold] RAVE model\n"
        f"  run:       {run_path}\n"
        f"  streaming: {streaming}"
    )

    cmd = ["rave", "export", "--run", str(run_path)]
    if streaming:
        cmd.append("--streaming")

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"rave export failed (exit {result.returncode}):\n"
            f"{result.stderr.strip()}"
        )

    # Find the exported .ts file
    ts_files = list(run_path.glob("*.ts"))
    if not ts_files:
        # rave export may place the file next to the run path
        ts_files = list(run_path.parent.glob("*.ts"))
    if not ts_files:
        raise RuntimeError(
            f"Export succeeded but no .ts file found in {run_path}"
        )

    model_path = max(ts_files, key=lambda p: p.stat().st_mtime)
    config.rave_model_path = model_path
    console.print(f"[green]Exported model:[/green] {model_path}")
    return model_path


def get_training_status(config: CopywriteConfig) -> dict:
    """Check the state of RAVE training artifacts.

    Returns a dict with:
        - ``checkpoint_dir``: path to the checkpoint directory
        - ``checkpoints``: list of ``.ckpt`` file paths found
        - ``epoch_count``: number of checkpoint files (proxy for epochs)
        - ``latest_checkpoint``: path to the most recent checkpoint, or None
        - ``exported_model``: path to the ``.ts`` file if it exists, or None
    """
    ckpt_dir = config.rave_checkpoint_dir
    checkpoints = sorted(ckpt_dir.rglob("*.ckpt")) if ckpt_dir.exists() else []
    latest = max(checkpoints, key=lambda p: p.stat().st_mtime) if checkpoints else None

    ts_files = sorted(ckpt_dir.rglob("*.ts")) if ckpt_dir.exists() else []
    exported = max(ts_files, key=lambda p: p.stat().st_mtime) if ts_files else None
    if exported is None and config.rave_model_path and config.rave_model_path.exists():
        exported = config.rave_model_path

    return {
        "checkpoint_dir": ckpt_dir,
        "checkpoints": checkpoints,
        "epoch_count": len(checkpoints),
        "latest_checkpoint": latest,
        "exported_model": exported,
    }
