"""Manage a real-time scsynth process for testing SynthDefs."""

from __future__ import annotations

import socket
import struct
import subprocess
import time
from typing import Any

from rich.console import Console

from copywrite.config import CopywriteConfig

console = Console()


def _encode_osc_string(s: str) -> bytes:
    """Encode a string as an OSC string (null-terminated, padded to 4-byte boundary)."""
    encoded = s.encode("ascii") + b"\x00"
    padding = (4 - len(encoded) % 4) % 4
    return encoded + b"\x00" * padding


def _encode_osc_message(addr: str, *args: Any) -> bytes:
    """Build an OSC message from an address pattern and arguments.

    Supported argument types: int, float, str, bytes.
    """
    type_tag = ","
    arg_data = b""
    for a in args:
        if isinstance(a, int):
            type_tag += "i"
            arg_data += struct.pack(">i", a)
        elif isinstance(a, float):
            type_tag += "f"
            arg_data += struct.pack(">f", a)
        elif isinstance(a, str):
            type_tag += "s"
            arg_data += _encode_osc_string(a)
        elif isinstance(a, bytes):
            type_tag += "b"
            arg_data += struct.pack(">i", len(a)) + a
            padding = (4 - len(a) % 4) % 4
            arg_data += b"\x00" * padding
        else:
            raise TypeError(f"Unsupported OSC argument type: {type(a)}")
    return _encode_osc_string(addr) + _encode_osc_string(type_tag) + arg_data


class SCServer:
    """Manages a real-time scsynth process for live playback and SynthDef testing."""

    DEFAULT_PORT = 57110

    def __init__(self, config: CopywriteConfig, port: int | None = None) -> None:
        self._config = config
        self._port = port or self.DEFAULT_PORT
        self._process: subprocess.Popen | None = None
        self._sock: socket.socket | None = None

    # -- lifecycle --

    def boot(self) -> None:
        """Start the scsynth process."""
        if self.is_running():
            console.print("[yellow]scsynth is already running.[/yellow]")
            return

        scsynth = str(self._config.scsynth_path)
        synthdef_dir = str(self._config.synthdef_dir)

        cmd = [
            scsynth,
            "-u", str(self._port),
            "-a", "1024",
            "-m", str(self._config.sc_memory),
            "-D", "0",
            "-R", "0",
            "-l", "1",
            "-z", str(self._config.sc_block_size),
            "-S", str(self._config.sc_sample_rate),
            "-U", synthdef_dir,
        ]

        console.print(f"[cyan]Booting scsynth on port {self._port}...[/cyan]")
        try:
            self._process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        except FileNotFoundError:
            raise RuntimeError(
                f"scsynth not found at {scsynth}. "
                "Set supercollider_path in ~/.copywrite/config.yaml"
            )

        # Give the server a moment to initialise
        time.sleep(1.5)

        if self._process.poll() is not None:
            stderr = self._process.stderr.read().decode(errors="replace") if self._process.stderr else ""
            raise RuntimeError(f"scsynth failed to start:\n{stderr}")

        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # Send /notify to register for replies
        self.send_osc("/notify", 1)
        console.print("[green]scsynth is running.[/green]")

    def quit(self) -> None:
        """Stop the scsynth process."""
        if self._sock is not None:
            try:
                self.send_osc("/quit")
            except OSError:
                pass
            self._sock.close()
            self._sock = None

        if self._process is not None:
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait()
            self._process = None
            console.print("[cyan]scsynth stopped.[/cyan]")

    def is_running(self) -> bool:
        """Return True if the scsynth process is alive."""
        return self._process is not None and self._process.poll() is None

    # -- OSC --

    def send_osc(self, addr: str, *args: Any) -> None:
        """Send a single OSC message to the running server."""
        if self._sock is None:
            raise RuntimeError("Server is not running; call boot() first.")
        msg = _encode_osc_message(addr, *args)
        self._sock.sendto(msg, ("127.0.0.1", self._port))
