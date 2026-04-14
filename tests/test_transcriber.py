"""Tests for the transcriber module."""

from __future__ import annotations

import pytest
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock


def _can_import_analyzer() -> bool:
    """Check if the analyzer module is fully importable."""
    try:
        from copywrite.transcriber.analyzer import analyze_track, TrackAnalysis
        return True
    except (ImportError, ModuleNotFoundError, AttributeError):
        return False


def _can_import_codegen() -> bool:
    """Check if the codegen module is fully importable."""
    try:
        from copywrite.transcriber.codegen import generate_sc_code
        return True
    except (ImportError, ModuleNotFoundError, AttributeError):
        return False


_analyzer_available = _can_import_analyzer()
_codegen_available = _can_import_codegen()


@pytest.fixture
def mock_audio(tmp_path):
    """Create a short mono WAV file for testing."""
    sr = 22050
    duration = 2.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    # 440 Hz sine with some noise for onset detection
    y = 0.5 * np.sin(2 * np.pi * 440 * t)
    y += 0.05 * np.random.randn(len(y))
    y = y.astype(np.float32)

    import soundfile as sf
    path = tmp_path / "test_track.wav"
    sf.write(str(path), y, sr)
    return path


@pytest.fixture
def sample_analysis():
    """Build a realistic TrackAnalysis-like dict for codegen tests."""
    return {
        "bpm": 120.0,
        "key": "Am",
        "duration": 30.0,
        "sections": [
            {
                "label": "intro",
                "start_time": 0.0,
                "end_time": 8.0,
                "bpm": 120.0,
                "key": "Am",
                "drum_pattern": {
                    "bars": 1,
                    "bpm": 120.0,
                    "kick": [[1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0]],
                    "snare": [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
                    "hihat": [[1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]],
                    "clap": [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
                },
                "bass_present": False,
                "bass_notes": [],
                "lead_present": False,
                "lead_notes": [],
                "pad_present": False,
                "vocoder_present": False,
                "chord_sequence": [{"chord": "Am", "start": 0.0, "end": 8.0}],
                "filter_automation": None,
                "energy": 0.4,
                "sidechain_active": False,
                "sidechain_depth_db": 0.0,
            },
            {
                "label": "drop",
                "start_time": 8.0,
                "end_time": 24.0,
                "bpm": 120.0,
                "key": "Am",
                "drum_pattern": {
                    "bars": 1,
                    "bpm": 120.0,
                    "kick": [[1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0]],
                    "snare": [[0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]],
                    "hihat": [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]],
                    "clap": [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
                },
                "bass_present": True,
                "bass_notes": [
                    {"pitch_midi": 45, "start": 8.0, "duration": 1.0},
                ],
                "lead_present": False,
                "lead_notes": [],
                "pad_present": False,
                "vocoder_present": False,
                "chord_sequence": [{"chord": "Am", "start": 8.0, "end": 16.0}],
                "filter_automation": None,
                "energy": 0.8,
                "sidechain_active": True,
                "sidechain_depth_db": -6.0,
            },
            {
                "label": "outro",
                "start_time": 24.0,
                "end_time": 30.0,
                "bpm": 120.0,
                "key": "Am",
                "drum_pattern": None,
                "bass_present": False,
                "bass_notes": [],
                "lead_present": False,
                "lead_notes": [],
                "pad_present": True,
                "vocoder_present": False,
                "chord_sequence": [{"chord": "Am", "start": 24.0, "end": 30.0}],
                "filter_automation": None,
                "energy": 0.3,
                "sidechain_active": False,
                "sidechain_depth_db": 0.0,
            },
        ],
        "effects_estimate": {
            "sidechain_depth": -6.0,
            "bitcrushing_detected": False,
        },
        "compression_estimate": {
            "crest_factor": 3.0,
            "dynamic_range": 12.0,
        },
        "spectral_character": {
            "centroid_mean": 2500.0,
        },
    }


class TestTrackAnalysis:
    @pytest.mark.skipif(not _analyzer_available,
                        reason="analyzer module not yet available")
    def test_to_dict_roundtrip(self, mock_audio):
        from copywrite.transcriber.analyzer import analyze_track
        analysis = analyze_track(mock_audio)
        d = analysis.to_dict() if hasattr(analysis, "to_dict") else vars(analysis)
        assert isinstance(d, dict)
        assert "bpm" in d or "tempo" in d

    @pytest.mark.skipif(not _analyzer_available,
                        reason="analyzer module not yet available")
    def test_sections_have_required_fields(self, mock_audio):
        from copywrite.transcriber.analyzer import analyze_track
        analysis = analyze_track(mock_audio)
        d = analysis.to_dict() if hasattr(analysis, "to_dict") else vars(analysis)
        sections = d.get("sections", [])
        for sec in sections:
            assert "label" in sec or "name" in sec
            assert "start_time" in sec or "start" in sec


class TestCodegen:
    @pytest.mark.skipif(not _codegen_available,
                        reason="codegen module not yet available")
    def test_generate_sc_code_returns_string(self, sample_analysis):
        from copywrite.transcriber.codegen import generate_sc_code
        synthdef_docs = (
            "Available SynthDefs\n"
            "========================================\n"
            "  \\kick(out=0, freq=50, decay=0.5, amp=0.8)\n"
            "  \\snare(out=0, freq=200, noiseAmt=0.5, decay=0.2)\n"
            "  \\hihat(out=0, decay=0.05, amp=0.3)\n"
            "  \\bassline(out=0, freq=110, gate=1, filterCutoff=800)\n"
            "  \\padSynth(out=0, freq=440, gate=1, amp=0.4)\n"
        )
        code = generate_sc_code(sample_analysis, synthdef_docs)
        assert isinstance(code, str)
        assert len(code) > 0

    @pytest.mark.skipif(not _codegen_available,
                        reason="codegen module not yet available")
    def test_generated_code_contains_score(self, sample_analysis):
        from copywrite.transcriber.codegen import generate_sc_code
        synthdef_docs = "Available SynthDefs\n\\kick(out=0)\n"
        code = generate_sc_code(sample_analysis, synthdef_docs)
        assert "Score" in code or "s_new" in code or "score" in code.lower()

    @pytest.mark.skipif(not _codegen_available,
                        reason="codegen module not yet available")
    def test_generated_code_uses_synthdef_names(self, sample_analysis):
        from copywrite.transcriber.codegen import generate_sc_code
        synthdef_docs = "Available SynthDefs\n\\kick(out=0)\n\\bassline(out=0)\n"
        code = generate_sc_code(sample_analysis, synthdef_docs)
        assert "kick" in code or "bassline" in code
