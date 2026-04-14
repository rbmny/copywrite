"""Tests for the engine module."""

from __future__ import annotations

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from copywrite.config import CopywriteConfig
from copywrite.engine import NRTRenderer, SynthDefManager, SCServer


@pytest.fixture
def config(tmp_path):
    """Create a config pointing at a temp directory."""
    synthdef_dir = tmp_path / "synthdefs"
    synthdef_dir.mkdir()
    # Write a minimal SynthDef library file for parsing tests
    lib = synthdef_dir / "copywrite_lib.scd"
    lib.write_text(
        "(\n"
        "SynthDef(\\kick, { |out=0, freq=50, decay=0.5, amp=0.8|\n"
        "  Out.ar(out, SinOsc.ar(freq) * amp);\n"
        "}).writeDefFile;\n"
        "\n"
        "SynthDef(\\snare, { |out=0, freq=200, noiseAmt=0.5, decay=0.2|\n"
        "  Out.ar(out, WhiteNoise.ar * 0.3);\n"
        "}).writeDefFile;\n"
        ")\n",
        encoding="utf-8",
    )
    return CopywriteConfig(
        project_dir=tmp_path,
        data_dir=tmp_path / "data",
        supercollider_path=str(tmp_path / "sc_bin"),
    )


class TestSCServer:
    def test_init(self, config):
        server = SCServer(config)
        assert server._config is config
        assert server._port == SCServer.DEFAULT_PORT

    def test_init_custom_port(self, config):
        server = SCServer(config, port=57200)
        assert server._port == 57200

    def test_is_running_default_false(self, config):
        server = SCServer(config)
        assert server.is_running() is False

    def test_send_osc_raises_when_not_running(self, config):
        server = SCServer(config)
        with pytest.raises(RuntimeError, match="not running"):
            server.send_osc("/test")


class TestNRTRenderer:
    def test_init(self, config):
        renderer = NRTRenderer(config)
        assert renderer._config is config

    def test_build_score_writer(self, config):
        renderer = NRTRenderer(config)
        output = Path("/tmp/test_output.osc")
        script = renderer._build_score_writer(
            "// user code here", output, 10.0
        )
        assert "user code" in script
        assert "writeOSCFile" in script
        assert "10.0" in script

    def test_build_score_writer_contains_sort(self, config):
        renderer = NRTRenderer(config)
        script = renderer._build_score_writer("// code", Path("/out.osc"), 5.0)
        assert "score.sort" in script
        assert "0.exit" in script

    def test_render_raises_on_missing_scsynth(self, config):
        renderer = NRTRenderer(config)
        with pytest.raises(RuntimeError, match="sclang not found"):
            renderer.render(
                "// test code",
                config.data_dir / "test.wav",
                duration=5.0,
            )


class TestSynthDefManager:
    def test_list_synthdefs(self, config):
        mgr = SynthDefManager(config)
        names = mgr.list_synthdefs()
        assert "kick" in names
        assert "snare" in names
        assert len(names) == 2

    def test_get_synthdef_params(self, config):
        mgr = SynthDefManager(config)
        params = mgr.get_synthdef_params("kick")
        assert "out" in params
        assert "freq" in params
        assert params["freq"] == 50.0
        assert params["amp"] == 0.8

    def test_get_synthdef_params_missing_raises(self, config):
        mgr = SynthDefManager(config)
        with pytest.raises(KeyError, match="nonexistent"):
            mgr.get_synthdef_params("nonexistent")

    def test_get_synthdef_docs_returns_string(self, config):
        mgr = SynthDefManager(config)
        docs = mgr.get_synthdef_docs()
        assert isinstance(docs, str)
        assert "kick" in docs
        assert "snare" in docs
        assert "freq=" in docs

    def test_compile_raises_when_sclang_missing(self, config):
        mgr = SynthDefManager(config)
        with pytest.raises(RuntimeError, match="sclang not found"):
            mgr.compile()

    def test_compile_raises_when_lib_missing(self, tmp_path):
        cfg = CopywriteConfig(
            project_dir=tmp_path,
            data_dir=tmp_path / "data",
            supercollider_path=str(tmp_path / "sc"),
        )
        mgr = SynthDefManager(cfg)
        with pytest.raises(FileNotFoundError, match="SynthDef library not found"):
            mgr.compile()
