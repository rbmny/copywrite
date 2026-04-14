"""Audio feature extraction for copywrite scoring."""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np
import librosa
import soundfile as sf


# ---------------------------------------------------------------------------
# Krumhansl-Schmuckler key profiles
# ---------------------------------------------------------------------------
_MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09,
                           2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
_MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53,
                           2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

_NOTE_NAMES = ["C", "C#", "D", "Eb", "E", "F",
               "F#", "G", "Ab", "A", "Bb", "B"]

# Chord templates: major, minor, dominant-7 for each root
_CHORD_TEMPLATES: dict[str, np.ndarray] = {}
for _i, _name in enumerate(_NOTE_NAMES):
    maj = np.zeros(12); maj[_i] = 1; maj[(_i + 4) % 12] = 1; maj[(_i + 7) % 12] = 1
    _CHORD_TEMPLATES[_name] = maj / np.linalg.norm(maj)
    mi = np.zeros(12); mi[_i] = 1; mi[(_i + 3) % 12] = 1; mi[(_i + 7) % 12] = 1
    _CHORD_TEMPLATES[f"{_name}m"] = mi / np.linalg.norm(mi)
    d7 = np.zeros(12); d7[_i] = 1; d7[(_i + 4) % 12] = 1; d7[(_i + 7) % 12] = 1; d7[(_i + 10) % 12] = 1
    _CHORD_TEMPLATES[f"{_name}7"] = d7 / np.linalg.norm(d7)

# Relative-key mapping (major <-> relative minor)
_RELATIVE_KEYS: dict[str, str] = {}
for _i, _name in enumerate(_NOTE_NAMES):
    _rel_min_idx = (_i + 9) % 12
    _RELATIVE_KEYS[_name] = f"{_NOTE_NAMES[_rel_min_idx]}m"
    _RELATIVE_KEYS[f"{_NOTE_NAMES[_rel_min_idx]}m"] = _name


# ---------------------------------------------------------------------------
# Feature dataclasses
# ---------------------------------------------------------------------------

@dataclass
class RhythmFeatures:
    tempo: float
    beat_positions: list[float]
    onset_density: float
    kick_pattern: list[int]
    snare_pattern: list[int]
    hihat_pattern: list[int]
    swing_amount: float


@dataclass
class HarmonyFeatures:
    key: str
    key_confidence: float
    chroma_mean: list[float]
    chord_sequence: list[dict]
    bass_pitches: list[float]


@dataclass
class SpectralFeatures:
    spectral_centroid_mean: float
    spectral_centroid_std: float
    spectral_centroid_contour: list[float]
    spectral_flatness_mean: float
    spectral_bandwidth_mean: float
    mfcc_mean: list[float]
    mfcc_std: list[float]
    filter_cutoff_estimate: list[float]


@dataclass
class DynamicsFeatures:
    rms_mean: float
    rms_std: float
    rms_contour: list[float]
    crest_factor: float
    dynamic_range: float
    sidechain_depth: float
    sidechain_rate: float


@dataclass
class StructureFeatures:
    duration: float
    section_boundaries: list[float]
    section_labels: list[str]
    energy_contour: list[float]


@dataclass
class AudioFeatures:
    """Complete feature set for one audio file."""
    file_path: str
    rhythm: RhythmFeatures
    harmony: HarmonyFeatures
    spectral: SpectralFeatures
    dynamics: DynamicsFeatures
    structure: StructureFeatures

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: Path) -> None:
        """Save features to JSON."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=_json_default)

    @classmethod
    def load(cls, path: Path) -> AudioFeatures:
        """Load features from JSON."""
        with open(path) as f:
            data = json.load(f)
        return cls(
            file_path=data["file_path"],
            rhythm=RhythmFeatures(**data["rhythm"]),
            harmony=HarmonyFeatures(**data["harmony"]),
            spectral=SpectralFeatures(**data["spectral"]),
            dynamics=DynamicsFeatures(**data["dynamics"]),
            structure=StructureFeatures(**data["structure"]),
        )


# ---------------------------------------------------------------------------
# JSON serialisation helper
# ---------------------------------------------------------------------------

def _json_default(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    raise TypeError(f"Not serializable: {type(obj)}")


# ---------------------------------------------------------------------------
# Sub-extractors
# ---------------------------------------------------------------------------

def _extract_rhythm(y: np.ndarray, sr: int) -> RhythmFeatures:
    # Beat tracking
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    tempo = float(np.atleast_1d(tempo)[0])
    beat_times = librosa.frames_to_time(beat_frames, sr=sr).tolist()

    # Onset density
    onsets = librosa.onset.onset_detect(y=y, sr=sr)
    onset_times = librosa.frames_to_time(onsets, sr=sr)
    duration = len(y) / sr
    onset_density = float(len(onset_times) / max(duration, 0.01))

    # HPSS for percussive component
    y_harm, y_perc = librosa.effects.hpss(y)

    # Frequency-band energy at beat positions for drum patterns
    hop = 512
    n_fft = 2048
    S = np.abs(librosa.stft(y_perc, n_fft=n_fft, hop_length=hop))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

    kick_band = (freqs >= 20) & (freqs <= 150)
    snare_band = (freqs >= 150) & (freqs <= 1000)
    hihat_band = freqs >= 5000

    kick_energy = S[kick_band, :].sum(axis=0) if kick_band.any() else np.zeros(S.shape[1])
    snare_energy = S[snare_band, :].sum(axis=0) if snare_band.any() else np.zeros(S.shape[1])
    hihat_energy = S[hihat_band, :].sum(axis=0) if hihat_band.any() else np.zeros(S.shape[1])

    def _quantise_pattern(energy: np.ndarray, beat_frames_arr: np.ndarray) -> list[int]:
        """Quantise band energy at beat positions to a 16-step binary pattern."""
        if len(beat_frames_arr) < 2:
            return [0] * 16
        # subdivide beats into 16th notes (4 per beat, 4 beats = 16 steps)
        n_beats = min(len(beat_frames_arr) - 1, 4)
        steps: list[float] = []
        for b in range(n_beats):
            start = beat_frames_arr[b]
            end = beat_frames_arr[b + 1]
            for s in range(4):
                frac = start + (end - start) * s / 4
                steps.append(frac)
        # pad to 16 if needed
        while len(steps) < 16:
            steps.append(steps[-1] if steps else 0)
        steps = steps[:16]

        values = []
        for s in steps:
            idx = int(round(s))
            idx = min(idx, len(energy) - 1)
            values.append(float(energy[idx]))

        if max(values) == 0:
            return [0] * 16
        threshold = np.mean(values) + 0.5 * np.std(values)
        return [1 if v > threshold else 0 for v in values]

    bf = librosa.time_to_frames(beat_times, sr=sr, hop_length=hop) if beat_times else np.array([])
    kick_pattern = _quantise_pattern(kick_energy, bf)
    snare_pattern = _quantise_pattern(snare_energy, bf)
    hihat_pattern = _quantise_pattern(hihat_energy, bf)

    # Swing: deviation of even-numbered onsets from perfect grid
    swing = 0.0
    if len(onset_times) >= 4 and len(beat_times) >= 2:
        beat_dur = np.median(np.diff(beat_times)) if len(beat_times) > 1 else 0.5
        grid_16th = beat_dur / 4.0
        if grid_16th > 0:
            deviations = []
            for ot in onset_times:
                # nearest grid position
                grid_pos = round(ot / grid_16th)
                if grid_pos % 2 == 1:  # off-beat 16ths
                    expected = grid_pos * grid_16th
                    deviations.append(abs(ot - expected) / grid_16th)
            if deviations:
                swing = float(min(np.mean(deviations), 1.0))

    return RhythmFeatures(
        tempo=tempo,
        beat_positions=beat_times,
        onset_density=onset_density,
        kick_pattern=kick_pattern,
        snare_pattern=snare_pattern,
        hihat_pattern=hihat_pattern,
        swing_amount=swing,
    )


def _detect_key(chroma: np.ndarray) -> tuple[str, float]:
    """Detect key using Krumhansl-Schmuckler algorithm.
    Returns (key_name, confidence).
    """
    chroma_sum = chroma.mean(axis=1)
    chroma_sum = chroma_sum / (np.linalg.norm(chroma_sum) + 1e-10)

    best_corr = -2.0
    best_key = "C"
    for shift in range(12):
        major_shifted = np.roll(_MAJOR_PROFILE, shift)
        major_shifted = major_shifted / (np.linalg.norm(major_shifted) + 1e-10)
        corr_maj = float(np.corrcoef(chroma_sum, major_shifted)[0, 1])

        minor_shifted = np.roll(_MINOR_PROFILE, shift)
        minor_shifted = minor_shifted / (np.linalg.norm(minor_shifted) + 1e-10)
        corr_min = float(np.corrcoef(chroma_sum, minor_shifted)[0, 1])

        if corr_maj > best_corr:
            best_corr = corr_maj
            best_key = _NOTE_NAMES[shift]
        if corr_min > best_corr:
            best_corr = corr_min
            best_key = f"{_NOTE_NAMES[shift]}m"

    confidence = float(max(0.0, min(1.0, (best_corr + 1.0) / 2.0)))
    return best_key, confidence


def _extract_harmony(y: np.ndarray, sr: int, beat_times: list[float]) -> HarmonyFeatures:
    # Chroma
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    chroma_mean = chroma.mean(axis=1).tolist()

    # Key detection
    key, key_confidence = _detect_key(chroma)

    # Chord sequence: one chord per beat window
    hop = 512
    chord_sequence: list[dict] = []
    if len(beat_times) >= 2:
        for i in range(len(beat_times) - 1):
            start_t = beat_times[i]
            end_t = beat_times[i + 1]
            start_frame = librosa.time_to_frames(start_t, sr=sr, hop_length=hop)
            end_frame = librosa.time_to_frames(end_t, sr=sr, hop_length=hop)
            start_frame = max(0, min(start_frame, chroma.shape[1] - 1))
            end_frame = max(start_frame + 1, min(end_frame, chroma.shape[1]))
            window_chroma = chroma[:, start_frame:end_frame].mean(axis=1)
            norm = np.linalg.norm(window_chroma)
            if norm > 1e-10:
                window_chroma = window_chroma / norm

            best_chord = "C"
            best_sim = -1.0
            for chord_name, tmpl in _CHORD_TEMPLATES.items():
                sim = float(np.dot(window_chroma, tmpl))
                if sim > best_sim:
                    best_sim = sim
                    best_chord = chord_name

            chord_sequence.append({
                "chord": best_chord,
                "start": round(start_t, 3),
                "end": round(end_t, 3),
            })

    # Bass pitch tracking on harmonic component
    y_harm, _ = librosa.effects.hpss(y)
    # Filter to low frequencies for bass
    y_bass = librosa.effects.preemphasis(y_harm, coef=-0.95)  # boost lows
    pitches, magnitudes = librosa.piptrack(y=y_bass, sr=sr, fmin=30, fmax=300)
    bass_pitches: list[float] = []
    # Downsample: pick one pitch per beat
    for bt in beat_times:
        frame = librosa.time_to_frames(bt, sr=sr)
        frame = min(frame, pitches.shape[1] - 1)
        mag_col = magnitudes[:, frame]
        if mag_col.max() > 0:
            idx = int(mag_col.argmax())
            freq = float(pitches[idx, frame])
            if freq > 0:
                midi = float(librosa.hz_to_midi(freq))
                bass_pitches.append(round(midi, 1))
            else:
                bass_pitches.append(0.0)
        else:
            bass_pitches.append(0.0)

    return HarmonyFeatures(
        key=key,
        key_confidence=key_confidence,
        chroma_mean=chroma_mean,
        chord_sequence=chord_sequence,
        bass_pitches=bass_pitches,
    )


def _extract_spectral(y: np.ndarray, sr: int, beat_times: list[float]) -> SpectralFeatures:
    hop = 512

    centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop)[0]
    flatness = librosa.feature.spectral_flatness(y=y, hop_length=hop)[0]
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr, hop_length=hop)[0]
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=hop)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, hop_length=hop, roll_percent=0.85)[0]

    # Downsample centroid contour to ~1 per beat
    centroid_contour: list[float] = []
    if beat_times:
        for bt in beat_times:
            frame = librosa.time_to_frames(bt, sr=sr, hop_length=hop)
            frame = min(frame, len(centroid) - 1)
            centroid_contour.append(float(centroid[frame]))
    else:
        # Downsample to ~50 points
        step = max(1, len(centroid) // 50)
        centroid_contour = [float(centroid[i]) for i in range(0, len(centroid), step)]

    # Filter cutoff estimate downsampled similarly
    cutoff_contour: list[float] = []
    if beat_times:
        for bt in beat_times:
            frame = librosa.time_to_frames(bt, sr=sr, hop_length=hop)
            frame = min(frame, len(rolloff) - 1)
            cutoff_contour.append(float(rolloff[frame]))
    else:
        step = max(1, len(rolloff) // 50)
        cutoff_contour = [float(rolloff[i]) for i in range(0, len(rolloff), step)]

    return SpectralFeatures(
        spectral_centroid_mean=float(centroid.mean()),
        spectral_centroid_std=float(centroid.std()),
        spectral_centroid_contour=centroid_contour,
        spectral_flatness_mean=float(flatness.mean()),
        spectral_bandwidth_mean=float(bandwidth.mean()),
        mfcc_mean=[float(m) for m in mfcc.mean(axis=1)],
        mfcc_std=[float(m) for m in mfcc.std(axis=1)],
        filter_cutoff_estimate=cutoff_contour,
    )


def _extract_dynamics(y: np.ndarray, sr: int, beat_times: list[float]) -> DynamicsFeatures:
    hop = 512
    rms = librosa.feature.rms(y=y, hop_length=hop)[0]

    rms_mean = float(rms.mean())
    rms_std = float(rms.std())

    # Downsample RMS contour
    if beat_times:
        rms_contour: list[float] = []
        for bt in beat_times:
            frame = librosa.time_to_frames(bt, sr=sr, hop_length=hop)
            frame = min(frame, len(rms) - 1)
            rms_contour.append(float(rms[frame]))
    else:
        step = max(1, len(rms) // 50)
        rms_contour = [float(rms[i]) for i in range(0, len(rms), step)]

    # Crest factor
    peak = float(np.abs(y).max())
    crest_factor = float(peak / max(rms_mean, 1e-10))

    # Dynamic range: 95th - 5th percentile of RMS in dB
    rms_db = librosa.amplitude_to_db(rms + 1e-10)
    p95 = float(np.percentile(rms_db, 95))
    p5 = float(np.percentile(rms_db, 5))
    dynamic_range = p95 - p5

    # Sidechain detection: look for periodic dips at beat rate
    sidechain_depth = 0.0
    sidechain_rate = 0.0
    if len(beat_times) >= 2 and len(rms) > 10:
        beat_dur = float(np.median(np.diff(beat_times)))
        beat_frames = int(round(beat_dur * sr / hop))
        if beat_frames > 2:
            # Autocorrelation of RMS at beat period
            rms_centered = rms - rms.mean()
            acf = np.correlate(rms_centered, rms_centered, mode="full")
            acf = acf[len(acf) // 2:]
            if len(acf) > beat_frames:
                acf_norm = acf / (acf[0] + 1e-10)
                beat_acf = float(acf_norm[min(beat_frames, len(acf_norm) - 1)])
                if beat_acf > 0.3:
                    # Measure dip depth: look at RMS near beat onsets vs between
                    dips: list[float] = []
                    for bt in beat_times:
                        frame = librosa.time_to_frames(bt, sr=sr, hop_length=hop)
                        frame = min(frame, len(rms) - 1)
                        # window around beat onset
                        w = max(1, beat_frames // 8)
                        start_f = max(0, frame - w)
                        end_f = min(len(rms), frame + w)
                        mid_f = min(len(rms) - 1, frame + beat_frames // 2)
                        mid_start = max(0, mid_f - w)
                        mid_end = min(len(rms), mid_f + w)
                        on_beat_rms = float(rms[start_f:end_f].mean())
                        off_beat_rms = float(rms[mid_start:mid_end].mean())
                        if on_beat_rms > 1e-10:
                            ratio = off_beat_rms / on_beat_rms
                            if ratio < 1.0:
                                dips.append(float(librosa.amplitude_to_db(np.array([ratio]))[0]))
                    if dips:
                        sidechain_depth = float(abs(np.median(dips)))
                        sidechain_rate = 1.0 / max(beat_dur, 0.01)

    return DynamicsFeatures(
        rms_mean=rms_mean,
        rms_std=rms_std,
        rms_contour=rms_contour,
        crest_factor=crest_factor,
        dynamic_range=dynamic_range,
        sidechain_depth=sidechain_depth,
        sidechain_rate=sidechain_rate,
    )


def _extract_structure(y: np.ndarray, sr: int) -> StructureFeatures:
    duration = float(len(y) / sr)
    hop = 512

    # Recurrence matrix + clustering for section boundaries
    try:
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=hop)
        # Use a beat-synchronous representation for stability
        tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, hop_length=hop)
        if len(beat_frames) > 4:
            beat_mfcc = librosa.util.sync(mfcc, beat_frames, aggregate=np.mean)
            R = librosa.segment.recurrence_matrix(
                beat_mfcc, k=int(max(2, beat_mfcc.shape[1] // 10)),
                width=3, mode="affinity", sym=True,
            )
            # Laplacian segmentation via eigendecomposition
            from scipy.ndimage import median_filter
            R_filtered = median_filter(R, size=(3, 3))
            # Cluster into sections using agglomerative clustering
            from sklearn.cluster import AgglomerativeClustering  # type: ignore
            n_sections = min(max(2, len(beat_frames) // 16), 8)
            clustering = AgglomerativeClustering(n_clusters=n_sections)
            labels = clustering.fit_predict(beat_mfcc.T)

            # Find boundaries where label changes
            beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=hop)
            boundaries: list[float] = [0.0]
            section_labels_raw: list[int] = [int(labels[0])]
            for i in range(1, len(labels)):
                if labels[i] != labels[i - 1]:
                    if i < len(beat_times):
                        boundaries.append(float(beat_times[i]))
                    section_labels_raw.append(int(labels[i]))
        else:
            boundaries = [0.0]
            section_labels_raw = [0]
    except Exception:
        boundaries = [0.0]
        section_labels_raw = [0]

    # RMS per section for energy contour and labelling
    rms = librosa.feature.rms(y=y, hop_length=hop)[0]
    rms_times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop)
    section_ends = boundaries[1:] + [duration]
    energy_contour: list[float] = []
    for i, start in enumerate(boundaries):
        end = section_ends[i]
        mask = (rms_times >= start) & (rms_times < end)
        if mask.any():
            energy_contour.append(float(rms[mask].mean()))
        else:
            energy_contour.append(0.0)

    # Heuristic section labels based on energy
    if energy_contour:
        max_e = max(energy_contour) if max(energy_contour) > 0 else 1.0
        normalised = [e / max_e for e in energy_contour]
        section_labels: list[str] = []
        n = len(normalised)
        for i, e in enumerate(normalised):
            if i == 0 and e < 0.5:
                section_labels.append("intro")
            elif i == n - 1 and e < 0.5:
                section_labels.append("outro")
            elif e >= 0.75:
                section_labels.append("drop")
            elif e < 0.4:
                section_labels.append("breakdown")
            elif i > 0 and normalised[i] > normalised[i - 1]:
                section_labels.append("build")
            else:
                section_labels.append("drop")
    else:
        section_labels = ["intro"]

    return StructureFeatures(
        duration=duration,
        section_boundaries=boundaries,
        section_labels=section_labels,
        energy_contour=energy_contour,
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def extract_features(audio_path: Path, sr: int = 44100) -> AudioFeatures:
    """Extract all features from an audio file."""
    y, file_sr = sf.read(str(audio_path), dtype="float32", always_2d=True)
    # Convert to mono
    if y.ndim == 2:
        y = y.mean(axis=1)
    # Resample if needed
    if file_sr != sr:
        y = librosa.resample(y, orig_sr=file_sr, target_sr=sr)

    rhythm = _extract_rhythm(y, sr)
    harmony = _extract_harmony(y, sr, rhythm.beat_positions)
    spectral = _extract_spectral(y, sr, rhythm.beat_positions)
    dynamics = _extract_dynamics(y, sr, rhythm.beat_positions)
    structure = _extract_structure(y, sr)

    return AudioFeatures(
        file_path=str(audio_path),
        rhythm=rhythm,
        harmony=harmony,
        spectral=spectral,
        dynamics=dynamics,
        structure=structure,
    )
