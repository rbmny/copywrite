"""Manage, compile, and introspect the copywrite SynthDef library."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

from rich.console import Console

from copywrite.config import CopywriteConfig

console = Console()

# Regex to extract SynthDef name and argument list from the .scd source.
_SYNTHDEF_RE = re.compile(
    r"SynthDef\s*\(\s*\\(\w+)\s*,\s*\{\s*\|([^|]*)\|",
    re.MULTILINE,
)


class SynthDefManager:
    """Load, compile, and query the copywrite SynthDef library."""

    def __init__(self, config: CopywriteConfig) -> None:
        self._config = config
        self._lib_path = config.synthdef_dir / "copywrite_lib.scd"

    @property
    def library_path(self) -> Path:
        return self._lib_path

    def compile(self) -> bool:
        """Compile the SynthDef library ``.scd`` via sclang.

        sclang evaluates the file, which calls ``writeDefFile`` for every
        SynthDef, producing ``.scsyndef`` files in ``~/.copywrite/synthdefs/``.

        Returns ``True`` on success.
        """
        if not self._lib_path.exists():
            raise FileNotFoundError(
                f"SynthDef library not found at {self._lib_path}"
            )

        sclang = str(self._config.sclang_path)
        console.print("[cyan]Compiling SynthDef library...[/cyan]")

        try:
            result = subprocess.run(
                [sclang, str(self._lib_path)],
                capture_output=True,
                text=True,
                timeout=120,
            )
        except FileNotFoundError:
            raise RuntimeError(
                f"sclang not found at {sclang}. "
                "Set supercollider_path in ~/.copywrite/config.yaml"
            )
        except subprocess.TimeoutExpired:
            raise RuntimeError("sclang timed out while compiling SynthDefs.")

        output = result.stdout + "\n" + result.stderr

        if result.returncode != 0:
            raise RuntimeError(
                f"SynthDef compilation failed (exit {result.returncode}):\n{output}"
            )

        console.print("[green]SynthDef library compiled successfully.[/green]")
        return True

    def list_synthdefs(self) -> list[str]:
        """Return the names of all SynthDefs defined in the library file."""
        source = self._read_lib()
        return [m.group(1) for m in _SYNTHDEF_RE.finditer(source)]

    def get_synthdef_params(self, name: str) -> dict[str, float]:
        """Parse the library source and return ``{param: default}`` for *name*.

        Raises ``KeyError`` if the SynthDef is not found.
        """
        source = self._read_lib()
        for m in _SYNTHDEF_RE.finditer(source):
            if m.group(1) == name:
                return self._parse_params(m.group(2))
        raise KeyError(f"SynthDef '{name}' not found in {self._lib_path}")

    def get_synthdef_docs(self) -> str:
        """Return a formatted reference string documenting all SynthDefs.

        Intended for use as context provided to sub-agents that generate
        SuperCollider code.
        """
        names = self.list_synthdefs()
        lines: list[str] = ["Available SynthDefs", "=" * 40]
        for name in names:
            params = self.get_synthdef_params(name)
            param_str = ", ".join(
                f"{k}={v}" for k, v in params.items()
            )
            lines.append(f"  \\{name}({param_str})")
        lines.append("")
        return "\n".join(lines)

    # -- internal helpers --

    def _read_lib(self) -> str:
        if not self._lib_path.exists():
            raise FileNotFoundError(
                f"SynthDef library not found at {self._lib_path}"
            )
        return self._lib_path.read_text(encoding="utf-8")

    @staticmethod
    def _parse_params(param_string: str) -> dict[str, float]:
        """Parse a SuperCollider argument string like ``out=0, freq=440`` into
        a dict mapping parameter names to their default values.
        """
        params: dict[str, float] = {}
        for token in param_string.split(","):
            token = token.strip()
            if not token:
                continue
            if "=" in token:
                key, val = token.split("=", 1)
                key = key.strip()
                try:
                    params[key] = float(val.strip())
                except ValueError:
                    params[key] = 0.0
            else:
                params[token.strip()] = 0.0
        return params
