"""Configuration management for copywrite."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings


class CopywriteConfig(BaseSettings):
    """Global configuration loaded from ~/.copywrite/config.yaml."""

    # Paths
    project_dir: Path = Field(default_factory=lambda: Path.cwd())
    data_dir: Path = Field(default_factory=lambda: Path.cwd() / "data")

    # Data Preparation
    rave_sample_rate: int = 48000
    musicgen_sample_rate: int = 32000
    rave_segment_duration: float = 4.0
    musicgen_segment_duration: float = 20.0

    # RAVE
    rave_config: str = "v2"
    rave_batch_size: int = 8
    rave_checkpoint_dir: Path = Field(
        default_factory=lambda: Path.cwd() / "data" / "checkpoints" / "rave"
    )
    rave_model_path: Optional[Path] = None

    # MusicGen
    musicgen_base_model: str = "facebook/musicgen-medium"
    musicgen_lora_rank: int = 16
    musicgen_lora_alpha: int = 32
    musicgen_lora_target_modules: list[str] = Field(
        default_factory=lambda: ["q_proj", "v_proj"]
    )
    musicgen_learning_rate: float = 1e-4
    musicgen_train_steps: int = 1000
    musicgen_checkpoint_dir: Path = Field(
        default_factory=lambda: Path.cwd() / "data" / "checkpoints" / "musicgen"
    )
    musicgen_adapter_path: Optional[Path] = None

    # EC2
    ec2_instance_type: str = "g5.xlarge"
    ec2_ami_id: str = ""
    ec2_key_name: str = ""
    ec2_key_path: str = ""  # path to .pem file for SSH
    ec2_region: str = "us-east-1"
    ec2_security_group: str = ""

    # Generation
    generation_duration: float = 15.0

    model_config = {"env_prefix": "COPYWRITE_"}

    # --- Directory properties ---

    @property
    def reference_dir(self) -> Path:
        return self.data_dir / "reference"

    @property
    def rave_preprocessed_dir(self) -> Path:
        return self.data_dir / "preprocessed" / "rave"

    @property
    def musicgen_preprocessed_dir(self) -> Path:
        return self.data_dir / "preprocessed" / "musicgen"

    @property
    def captions_dir(self) -> Path:
        return self.data_dir / "captions"

    @property
    def checkpoints_dir(self) -> Path:
        return self.data_dir / "checkpoints"

    @property
    def generated_dir(self) -> Path:
        return self.data_dir / "generated"

    @property
    def musicgen_dataset_dir(self) -> Path:
        return self.data_dir / "musicgen_dataset"

    def ensure_dirs(self) -> None:
        """Create all data directories if they don't exist."""
        for d in [
            self.reference_dir,
            self.rave_preprocessed_dir,
            self.musicgen_preprocessed_dir,
            self.captions_dir,
            self.rave_checkpoint_dir,
            self.musicgen_checkpoint_dir,
            self.generated_dir,
            self.musicgen_dataset_dir,
        ]:
            d.mkdir(parents=True, exist_ok=True)

    def save(self, path: Optional[Path] = None) -> None:
        """Write config to YAML."""
        path = path or _config_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        data = self.model_dump(mode="json")
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
            raw = yaml.safe_load(f) or {}
        # Drop any keys from old SC-based config that no longer exist
        valid_fields = set(CopywriteConfig.model_fields.keys())
        data = {k: v for k, v in raw.items() if k in valid_fields}
        return CopywriteConfig(**data)
    return CopywriteConfig()
