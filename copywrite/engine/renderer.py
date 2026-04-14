"""Non-realtime rendering via sclang (score generation) + scsynth (NRT render)."""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

from rich.console import Console

from copywrite.config import CopywriteConfig

console = Console()


class NRTRenderer:
    """Render SuperCollider code to WAV files using NRT (non-realtime) mode.

    Two-step pipeline:
    1. sclang compiles the SC code and writes a binary .osc score file
    2. scsynth runs in NRT mode on the .osc score to produce a .wav
    """

    def __init__(self, config: CopywriteConfig) -> None:
        self._config = config

    def render(
        self,
        scd_code: str,
        output_path: Path,
        duration: float = 60.0,
    ) -> Path:
        """Render SuperCollider code to a WAV file via NRT.

        The scd_code must define ~score as a Score object.
        Returns output_path on success; raises RuntimeError on failure.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        osc_path = output_path.with_suffix(".osc")

        # Step 1: use sclang to compile the Score into a binary .osc file
        sclang_script = self._build_score_writer(scd_code, osc_path, duration)
        self._run_sclang(sclang_script, tag="compile score")

        if not osc_path.exists():
            raise RuntimeError(
                f"sclang did not produce .osc score file: {osc_path}"
            )

        # Step 2: run scsynth in NRT mode
        self._run_scsynth_nrt(osc_path, output_path, duration)

        if not output_path.exists():
            raise RuntimeError(
                f"scsynth NRT did not produce output: {output_path}"
            )
        return output_path

    def render_synthdef_test(
        self,
        synthdef_name: str,
        params: dict,
        duration: float = 2.0,
    ) -> Path:
        """Render a single SynthDef with given parameters for quick auditioning."""
        output_path = self._config.data_dir / "test_renders" / f"{synthdef_name}_test.wav"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        param_pairs = []
        for k, v in params.items():
            param_pairs.append(f"\\{k}")
            param_pairs.append(str(v))
        param_string = ", ".join(param_pairs)

        s_new_args = f"\\{synthdef_name}, 1000, 0, 0"
        if param_string:
            s_new_args += f", {param_string}"

        scd_code = (
            f"~score = Score.new;\n"
            f"~score.add([0.0, [\\s_new, {s_new_args}]]);\n"
            f"~score.add([{duration}, [\\c_set, 0, 0]]);\n"
        )
        return self.render(scd_code, output_path, duration)

    # -- internal helpers --

    def _build_score_writer(
        self, scd_code: str, osc_path: Path, duration: float
    ) -> str:
        """Build a sclang script that writes a Score to a binary .osc file."""
        osc_str = str(osc_path).replace("\\", "/")

        return (
            f"// User code — must set ~score\n"
            f"{scd_code}\n"
            f"\n"
            f"if(~score.isNil, {{\n"
            f'    "Warning: ~score not defined; rendering silence.".postln;\n'
            f"    ~score = Score.new;\n"
            f"    ~score.add([0.0, [\\c_set, 0, 0]]);\n"
            f"    ~score.add([{duration}, [\\c_set, 0, 0]]);\n"
            f"}});\n"
            f"\n"
            f"~score.sort;\n"
            f'~score.writeOSCFile("{osc_str}");\n'
            f'"Score written to {osc_str}".postln;\n'
            f"0.exit;\n"
        )

    def _run_scsynth_nrt(
        self, osc_path: Path, output_path: Path, duration: float
    ) -> None:
        """Run scsynth in NRT mode on a binary .osc score file."""
        scsynth = str(self._config.scsynth_path)
        sr = self._config.sc_sample_rate

        cmd = [
            scsynth,
            "-N",
            str(osc_path),     # input score
            "_",                # no input audio file
            str(output_path),  # output wav
            str(sr),
            "WAV",
            "int16",
            "-o", "2",         # 2 output channels
        ]

        console.print(f"[cyan]Running scsynth NRT...[/cyan]")
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600,
            )
        except FileNotFoundError:
            raise RuntimeError(
                f"scsynth not found at {scsynth}. "
                "Set supercollider_path in ~/.copywrite/config.yaml"
            )
        except subprocess.TimeoutExpired:
            raise RuntimeError("scsynth NRT timed out after 600s")

        output = result.stdout + "\n" + result.stderr

        if result.returncode != 0:
            raise RuntimeError(
                f"scsynth NRT failed (exit {result.returncode}):\n{output}"
            )

    def _run_sclang(self, script: str, tag: str = "") -> str:
        """Write script to a temp file and execute it with sclang.

        Returns combined stdout+stderr. Raises RuntimeError on real errors.
        """
        sclang = str(self._config.sclang_path)
        label = f" [{tag}]" if tag else ""

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".scd", delete=False, dir=str(self._config.data_dir)
        ) as tmp:
            tmp.write(script)
            tmp_path = tmp.name

        console.print(f"[cyan]Running sclang{label}...[/cyan]")
        try:
            result = subprocess.run(
                [sclang, tmp_path],
                capture_output=True,
                text=True,
                timeout=300,
            )
        except FileNotFoundError:
            raise RuntimeError(
                f"sclang not found at {sclang}. "
                "Set supercollider_path in ~/.copywrite/config.yaml"
            )
        except subprocess.TimeoutExpired:
            raise RuntimeError(f"sclang timed out after 300s{label}")

        output = result.stdout + "\n" + result.stderr

        # sclang exit code 1 with "cleaning up OSC" is normal shutdown via 0.exit
        if result.returncode != 0:
            has_real_error = any(
                "ERROR:" in line or "FAILURE" in line
                for line in output.splitlines()
                if "MethodOverride" not in line
            )
            if has_real_error:
                raise RuntimeError(
                    f"sclang failed{label} (exit {result.returncode}):\n{output}"
                )

        return output
