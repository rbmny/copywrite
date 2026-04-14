"""EC2 instance management via the AWS CLI (no boto3)."""

from __future__ import annotations

import json
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from rich.console import Console

console = Console()


@dataclass
class EC2Instance:
    instance_id: str
    public_ip: str | None
    state: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_aws(args: list[str], region: str) -> dict:
    """Run an AWS CLI command and return parsed JSON output."""
    cmd = ["aws", "--region", region, "--output", "json"] + args
    console.print(f"[dim]$ {' '.join(cmd)}[/dim]")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"AWS CLI failed (exit {result.returncode}):\n{result.stderr.strip()}"
        )
    if not result.stdout.strip():
        return {}
    return json.loads(result.stdout)


def _parse_instance(inst: dict) -> EC2Instance:
    """Extract an EC2Instance from an AWS describe-instances entry."""
    return EC2Instance(
        instance_id=inst["InstanceId"],
        public_ip=inst.get("PublicIpAddress"),
        state=inst["State"]["Name"],
    )


# ---------------------------------------------------------------------------
# Instance lifecycle
# ---------------------------------------------------------------------------

def launch_instance(
    instance_type: str,
    ami_id: str,
    key_name: str,
    security_group: str,
    region: str,
) -> EC2Instance:
    """Launch a new EC2 instance and return its initial state."""
    console.print(
        f"[bold]Launching[/bold] {instance_type} in {region} "
        f"(AMI {ami_id}) ..."
    )
    data = _run_aws(
        [
            "ec2", "run-instances",
            "--instance-type", instance_type,
            "--image-id", ami_id,
            "--key-name", key_name,
            "--security-group-ids", security_group,
            "--count", "1",
        ],
        region=region,
    )
    inst = data["Instances"][0]
    ec2 = _parse_instance(inst)
    console.print(f"[green]Launched:[/green] {ec2.instance_id} ({ec2.state})")
    return ec2


def wait_for_instance(
    instance_id: str, region: str, timeout: int = 300
) -> EC2Instance:
    """Poll until the instance is running and has a public IP."""
    console.print(
        f"[bold]Waiting[/bold] for {instance_id} to reach 'running' "
        f"(timeout {timeout}s) ..."
    )
    deadline = time.monotonic() + timeout
    delay = 5.0

    while True:
        ec2 = get_instance_status(instance_id, region)
        if ec2.state == "running" and ec2.public_ip:
            console.print(
                f"[green]Running:[/green] {ec2.instance_id} @ {ec2.public_ip}"
            )
            return ec2

        if time.monotonic() >= deadline:
            raise RuntimeError(
                f"Timed out waiting for {instance_id} "
                f"(state={ec2.state}, ip={ec2.public_ip})"
            )

        console.print(
            f"  [dim]state={ec2.state}, ip={ec2.public_ip} — "
            f"retrying in {delay:.0f}s[/dim]"
        )
        time.sleep(delay)
        delay = min(delay * 1.5, 30.0)


def get_instance_status(instance_id: str, region: str) -> EC2Instance:
    """Return the current state of an instance."""
    data = _run_aws(
        ["ec2", "describe-instances", "--instance-ids", instance_id],
        region=region,
    )
    inst = data["Reservations"][0]["Instances"][0]
    return _parse_instance(inst)


def terminate_instance(instance_id: str, region: str) -> None:
    """Terminate an EC2 instance."""
    console.print(f"[bold]Terminating[/bold] {instance_id} ...")
    _run_aws(
        ["ec2", "terminate-instances", "--instance-ids", instance_id],
        region=region,
    )
    console.print(f"[green]Terminated:[/green] {instance_id}")


# ---------------------------------------------------------------------------
# File transfer
# ---------------------------------------------------------------------------

def upload_files(
    instance_ip: str,
    key_path: str,
    local_paths: list[Path],
    remote_dir: str,
) -> None:
    """Upload local files to the remote instance via scp."""
    for local in local_paths:
        local = Path(local)
        if not local.exists():
            raise RuntimeError(f"Local file not found: {local}")
        dest = f"ubuntu@{instance_ip}:{remote_dir}/{local.name}"
        console.print(f"[bold]Uploading[/bold] {local.name} -> {dest}")
        _scp(key_path, str(local), dest)
    console.print(f"[green]Uploaded {len(local_paths)} file(s).[/green]")


def download_files(
    instance_ip: str,
    key_path: str,
    remote_paths: list[str],
    local_dir: Path,
) -> None:
    """Download files from the remote instance via scp."""
    local_dir = Path(local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)
    for remote in remote_paths:
        src = f"ubuntu@{instance_ip}:{remote}"
        filename = Path(remote).name
        dest = str(local_dir / filename)
        console.print(f"[bold]Downloading[/bold] {src} -> {dest}")
        _scp(key_path, src, dest)
    console.print(f"[green]Downloaded {len(remote_paths)} file(s).[/green]")


def _scp(key_path: str, src: str, dest: str) -> None:
    """Run a single scp transfer."""
    cmd = [
        "scp",
        "-i", key_path,
        "-o", "StrictHostKeyChecking=no",
        src,
        dest,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"scp failed:\n{result.stderr.strip()}")


# ---------------------------------------------------------------------------
# Remote execution
# ---------------------------------------------------------------------------

def run_remote_command(
    instance_ip: str,
    key_path: str,
    command: str,
    timeout: int = 7200,
) -> str:
    """Run a command on the remote instance via SSH and return stdout."""
    cmd = [
        "ssh",
        "-i", key_path,
        "-o", "StrictHostKeyChecking=no",
        "-o", "ServerAliveInterval=30",
        f"ubuntu@{instance_ip}",
        command,
    ]
    console.print(f"[bold]SSH[/bold] {instance_ip}: {command[:80]}{'...' if len(command) > 80 else ''}")
    result = subprocess.run(
        cmd, capture_output=True, text=True, timeout=timeout
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Remote command failed (exit {result.returncode}):\n"
            f"{result.stderr.strip()}"
        )
    return result.stdout
