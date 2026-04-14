"""Configuration management for copywrite."""

from __future__ import annotations

import platform
import shutil
from pathlib import Path
from typing import Optional

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings


def _default_sc_path() -> str:
    """Best-guess path to scsynth/sclang based on OS."""
    system = platform.system()
    if system == "Windows":
        candidates = [
            Path(r"C:\Program Files\SuperCollider\scsynth.exe"),
            Path(r"C:\Program Files (x86)\SuperCollider\scsynth.exe"),
        ]
        for c in candidates:
            if c.exists():
                return str(c.parent)
        found = shutil.which("scsynth") or shutil.which("scsynth.exe")
        if found:
            return str(Path(found).parent)
    elif system == "Darwin":
        app_path = Path("/Applications/SuperCollider.app/Contents/Resources")
        if app_path.exists():
            return str(app_path)
    else:
        found = shutil.which("scsynth")
        if found:
            return str(Path(found).parent)
    return ""


class CopywriteConfig(BaseSettings):
    """Global configuration loaded from ~/.copywrite/config.yaml."""

    # Paths
    project_dir: Path = Field(default_factory=lambda: Path.cwd())
    data_dir: Path = Field(default_factory=lambda: Path.cwd() / "data")
    supercollider_path: str = Field(default_factory=_default_sc_path)

    # SuperCollider
    sc_sample_rate: int = 44100
    sc_block_size: int = 64
    sc_memory: int = 65536  # server memory in KB

    # Transcription
    transcribe_max_iterations: int = 15
    transcribe_score_threshold: float = 0.75

    # Generation
    default_duration: int = 90  # seconds
    default_bpm: int = 120
    default_count: int = 3

    # Scoring
    scoring_weights: dict = Field(default_factory=lambda: {
        "rhythm": 0.25,
        "harmony": 0.20,
        "spectral": 0.25,
        "structure": 0.15,
        "dynamics": 0.15,
    })

    model_config = {"env_prefix": "COPYWRITE_"}

    @property
    def reference_dir(self) -> Path:
        return self.data_dir / "reference"

    @property
    def transcriptions_dir(self) -> Path:
        return self.data_dir / "transcriptions"

    @property
    def style_model_dir(self) -> Path:
        return self.data_dir / "style_model"

    @property
    def generated_dir(self) -> Path:
        return self.data_dir / "generated"

    @property
    def synthdef_dir(self) -> Path:
        return self.project_dir / "synthdefs"

    @property
    def sclang_path(self) -> Path:
        base = Path(self.supercollider_path)
        name = "sclang.exe" if platform.system() == "Windows" else "sclang"
        return base / name

    @property
    def scsynth_path(self) -> Path:
        base = Path(self.supercollider_path)
        name = "scsynth.exe" if platform.system() == "Windows" else "scsynth"
        return base / name

    def ensure_dirs(self) -> None:
        """Create all data directories if they don't exist."""
        for d in [self.reference_dir, self.transcriptions_dir,
                  self.style_model_dir, self.generated_dir]:
            d.mkdir(parents=True, exist_ok=True)

    def save(self, path: Optional[Path] = None) -> None:
        """Write config to YAML."""
        path = path or _config_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        data = self.model_dump(mode="json")
        # Convert Path objects to strings for YAML
        for k, v in data.items():
            if isinstance(v, Path):
                data[k] = str(v)
        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False)


def _config_path() -> Path:
    return Path.home() / ".copywrite" / "config.yaml"


def load_config() -> CopywriteConfig:
    """Load config from ~/.copywrite/config.yaml, falling back to defaults."""
    path = _config_path()
    if path.exists():
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        return CopywriteConfig(**data)
    return CopywriteConfig()
