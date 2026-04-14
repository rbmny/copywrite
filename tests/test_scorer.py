"""Tests for the scoring module."""

from __future__ import annotations

import json
import pytest
import numpy as np

try:
    from copywrite.scoring.features import (
        AudioFeatures,
        RhythmFeatures,
        HarmonyFeatures,
        SpectralFeatures,
        DynamicsFeatures,
        StructureFeatures,
    )
    from copywrite.scoring.comparator import compare_features, TranscriptionScore
    _scoring_available = True
except ImportError:
    _scoring_available = False

pytestmark = pytest.mark.skipif(
    not _scoring_available,
    reason="scoring module dependencies (librosa/soundfile) not installed",
)


def _make_rhythm(**overrides) -> "RhythmFeatures":
    defaults = dict(
        tempo=120.0,
        beat_positions=[0.0, 0.5, 1.0, 1.5, 2.0],
        onset_density=8.0,
        kick_pattern=[1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
        snare_pattern=[0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        hihat_pattern=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        swing_amount=0.1,
    )
    defaults.update(overrides)
    return RhythmFeatures(**defaults)


def _make_harmony(**overrides) -> "HarmonyFeatures":
    defaults = dict(
        key="Am",
        key_confidence=0.85,
        chroma_mean=[0.5, 0.1, 0.2, 0.1, 0.3, 0.4, 0.1, 0.6, 0.2, 0.8, 0.1, 0.15],
        chord_sequence=[
            {"chord": "Am", "start": 0.0, "end": 2.0},
            {"chord": "F", "start": 2.0, "end": 4.0},
            {"chord": "C", "start": 4.0, "end": 6.0},
            {"chord": "G", "start": 6.0, "end": 8.0},
        ],
        bass_pitches=[45.0, 41.0, 48.0, 43.0],
    )
    defaults.update(overrides)
    return HarmonyFeatures(**defaults)


def _make_spectral(**overrides) -> "SpectralFeatures":
    defaults = dict(
        spectral_centroid_mean=2500.0,
        spectral_centroid_std=800.0,
        spectral_centroid_contour=[2000.0, 2500.0, 3000.0, 2800.0],
        spectral_flatness_mean=0.05,
        spectral_bandwidth_mean=1800.0,
        mfcc_mean=[
            -200.0, 50.0, -10.0, 20.0, -5.0,
            10.0, -3.0, 8.0, -2.0, 5.0,
            -1.0, 3.0, -0.5,
        ],
        mfcc_std=[
            30.0, 15.0, 12.0, 10.0, 8.0,
            7.0, 6.0, 5.0, 4.0, 3.0,
            2.5, 2.0, 1.5,
        ],
        filter_cutoff_estimate=[4000.0, 5000.0, 6000.0, 5500.0],
    )
    defaults.update(overrides)
    return SpectralFeatures(**defaults)


def _make_dynamics(**overrides) -> "DynamicsFeatures":
    defaults = dict(
        rms_mean=0.15,
        rms_std=0.04,
        rms_contour=[0.1, 0.15, 0.18, 0.16, 0.12],
        crest_factor=4.5,
        dynamic_range=20.0,
        sidechain_depth=3.0,
        sidechain_rate=2.0,
    )
    defaults.update(overrides)
    return DynamicsFeatures(**defaults)


def _make_structure(**overrides) -> "StructureFeatures":
    defaults = dict(
        duration=30.0,
        section_boundaries=[0.0, 8.0, 24.0],
        section_labels=["intro", "drop", "outro"],
        energy_contour=[0.3, 0.9, 0.2],
    )
    defaults.update(overrides)
    return StructureFeatures(**defaults)


def _make_features(**overrides) -> "AudioFeatures":
    """Build a complete AudioFeatures with realistic defaults."""
    return AudioFeatures(
        file_path=overrides.pop("file_path", "/tmp/test.wav"),
        rhythm=overrides.pop("rhythm", _make_rhythm()),
        harmony=overrides.pop("harmony", _make_harmony()),
        spectral=overrides.pop("spectral", _make_spectral()),
        dynamics=overrides.pop("dynamics", _make_dynamics()),
        structure=overrides.pop("structure", _make_structure()),
    )


class TestAudioFeatures:
    def test_to_dict(self):
        features = _make_features()
        d = features.to_dict()
        assert isinstance(d, dict)
        assert d["file_path"] == "/tmp/test.wav"
        assert "rhythm" in d
        assert "harmony" in d
        assert "spectral" in d
        assert "dynamics" in d
        assert "structure" in d
        assert d["rhythm"]["tempo"] == 120.0
        assert d["harmony"]["key"] == "Am"

    def test_save_load_roundtrip(self, tmp_path):
        original = _make_features()
        path = tmp_path / "features.json"
        original.save(path)

        loaded = AudioFeatures.load(path)

        assert loaded.file_path == original.file_path
        assert loaded.rhythm.tempo == original.rhythm.tempo
        assert loaded.harmony.key == original.harmony.key
        assert loaded.spectral.spectral_centroid_mean == original.spectral.spectral_centroid_mean
        assert loaded.dynamics.rms_mean == original.dynamics.rms_mean
        assert loaded.structure.duration == original.structure.duration
        assert loaded.rhythm.kick_pattern == original.rhythm.kick_pattern
        assert loaded.harmony.chord_sequence == original.harmony.chord_sequence

    def test_save_creates_valid_json(self, tmp_path):
        features = _make_features()
        path = tmp_path / "features.json"
        features.save(path)

        with open(path) as f:
            data = json.load(f)
        assert isinstance(data, dict)
        assert data["rhythm"]["tempo"] == 120.0


class TestComparator:
    def test_identical_features_score_one(self):
        features = _make_features()
        score = compare_features(features, features)
        assert isinstance(score, TranscriptionScore)
        # Identical features should produce a perfect or near-perfect score
        assert score.overall >= 0.99
        assert score.rhythm_score >= 0.99
        assert score.harmony_score >= 0.99
        assert score.spectral_score >= 0.99
        assert score.dynamics_score >= 0.99
        assert score.structure_score >= 0.99

    def test_different_features_score_low(self):
        ref = _make_features()
        rendered = _make_features(
            rhythm=_make_rhythm(
                tempo=70.0,
                kick_pattern=[0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
                snare_pattern=[1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
                hihat_pattern=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                onset_density=2.0,
            ),
            harmony=_make_harmony(
                key="F#",
                chroma_mean=[0.1, 0.8, 0.1, 0.1, 0.1, 0.1, 0.8, 0.1, 0.1, 0.1, 0.1, 0.1],
                chord_sequence=[
                    {"chord": "F#", "start": 0.0, "end": 4.0},
                    {"chord": "B", "start": 4.0, "end": 8.0},
                ],
            ),
            spectral=_make_spectral(
                spectral_centroid_mean=500.0,
                spectral_flatness_mean=0.8,
                spectral_bandwidth_mean=500.0,
                mfcc_mean=[
                    -100.0, 10.0, 5.0, -5.0, 2.0,
                    -1.0, 0.5, -0.2, 0.1, -0.05,
                    0.02, -0.01, 0.005,
                ],
            ),
            dynamics=_make_dynamics(
                rms_mean=0.01,
                crest_factor=15.0,
                dynamic_range=50.0,
                sidechain_depth=0.0,
            ),
            structure=_make_structure(
                section_boundaries=[0.0],
                section_labels=["intro"],
                energy_contour=[0.1],
            ),
        )
        score = compare_features(ref, rendered)
        assert score.overall < 0.6

    def test_weights_affect_score(self):
        ref = _make_features()
        # Same everything except rhythm
        rendered = _make_features(
            rhythm=_make_rhythm(
                tempo=60.0,
                kick_pattern=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                onset_density=1.0,
            ),
        )

        # High rhythm weight -> lower overall
        high_rhythm = {
            "rhythm": 0.80, "harmony": 0.05, "spectral": 0.05,
            "structure": 0.05, "dynamics": 0.05,
        }
        # Low rhythm weight -> higher overall
        low_rhythm = {
            "rhythm": 0.05, "harmony": 0.30, "spectral": 0.30,
            "structure": 0.20, "dynamics": 0.15,
        }

        score_high = compare_features(ref, rendered, weights=high_rhythm)
        score_low = compare_features(ref, rendered, weights=low_rhythm)

        assert score_high.overall < score_low.overall

    def test_score_summary_is_string(self):
        features = _make_features()
        score = compare_features(features, features)
        summary = score.summary()
        assert isinstance(summary, str)
        assert "Overall" in summary
        assert "Rhythm" in summary
        assert "Harmony" in summary

    def test_score_diagnostics_present(self):
        features = _make_features()
        score = compare_features(features, features)
        assert isinstance(score.diagnostics, dict)
        assert "rhythm" in score.diagnostics
        assert "harmony" in score.diagnostics
        assert "spectral" in score.diagnostics

    def test_score_fields_in_range(self):
        ref = _make_features()
        rendered = _make_features(
            rhythm=_make_rhythm(tempo=100.0),
        )
        score = compare_features(ref, rendered)
        for field in [score.overall, score.rhythm_score, score.harmony_score,
                      score.spectral_score, score.dynamics_score, score.structure_score]:
            assert 0.0 <= field <= 1.0
