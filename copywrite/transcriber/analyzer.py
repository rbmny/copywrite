"""Extended audio analysis for transcription — builds a complete TrackAnalysis."""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path

import librosa
import numpy as np
from scipy import signal as scipy_signal

from copywrite.scoring import extract_features


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class DrumPattern:
    """A single drum pattern extracted from the track."""
    bars: int
    bpm: float
    kick: list[list[int]]
    snare: list[list[int]]
    hihat: list[list[int]]
    clap: list[list[int]]


@dataclass
class FilterAutomation:
    """Filter cutoff movement over time."""
    timestamps: list[float]
    cutoff_values: list[float]
    resonance_estimate: float
    sweep_rate_hz_per_sec: float


@dataclass
class SectionAnalysis:
    """Detailed analysis of one section of a track."""
    label: str
    start_time: float
    end_time: float
    bpm: float
    key: str
    chord_sequence: list[dict]
    drum_pattern: DrumPattern | None
    bass_present: bool
    bass_notes: list[dict]
    lead_present: bool
    lead_notes: list[dict]
    pad_present: bool
    vocoder_present: bool
    filter_automation: FilterAutomation | None
    energy: float
    sidechain_active: bool
    sidechain_depth_db: float


@dataclass
class TrackAnalysis:
    """Complete analysis of one reference track."""
    file_path: str
    title: str
    duration: float
    bpm: float
    key: str
    sections: list[SectionAnalysis]
    global_filter_automation: FilterAutomation
    spectral_character: dict
    compression_estimate: dict
    effects_estimate: dict

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: Path) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=_json_default)

    @classmethod
    def load(cls, path: Path) -> TrackAnalysis:
        with open(path) as f:
            data = json.load(f)
        sections = [
            SectionAnalysis(
                label=s["label"],
                start_time=s["start_time"],
                end_time=s["end_time"],
                bpm=s["bpm"],
                key=s["key"],
                chord_sequence=s["chord_sequence"],
                drum_pattern=DrumPattern(**s["drum_pattern"]) if s.get("drum_pattern") else None,
                bass_present=s["bass_present"],
                bass_notes=s["bass_notes"],
                lead_present=s["lead_present"],
                lead_notes=s["lead_notes"],
                pad_present=s["pad_present"],
                vocoder_present=s["vocoder_present"],
                filter_automation=FilterAutomation(**s["filter_automation"]) if s.get("filter_automation") else None,
                energy=s["energy"],
                sidechain_active=s["sidechain_active"],
                sidechain_depth_db=s["sidechain_depth_db"],
            )
            for s in data["sections"]
        ]
        gfa = data["global_filter_automation"]
        return cls(
            file_path=data["file_path"],
            title=data["title"],
            duration=data["duration"],
            bpm=data["bpm"],
            key=data["key"],
            sections=sections,
            global_filter_automation=FilterAutomation(**gfa),
            spectral_character=data["spectral_character"],
            compression_estimate=data["compression_estimate"],
            effects_estimate=data["effects_estimate"],
        )


def _json_default(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    raise TypeError(f"Not serializable: {type(obj)}")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _estimate_key(y: np.ndarray, sr: int) -> tuple[str, float]:
    """Estimate musical key using chroma features."""
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    chroma_mean = chroma.mean(axis=1)

    key_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    major_template = np.array([1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1], dtype=float)
    minor_template = np.array([1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0], dtype=float)

    best_corr = -1.0
    best_key = "C"
    for shift in range(12):
        maj = np.corrcoef(chroma_mean, np.roll(major_template, shift))[0, 1]
        if maj > best_corr:
            best_corr = maj
            best_key = f"{key_names[shift]} major"
        mnr = np.corrcoef(chroma_mean, np.roll(minor_template, shift))[0, 1]
        if mnr > best_corr:
            best_corr = mnr
            best_key = f"{key_names[shift]} minor"
    return best_key, float(max(best_corr, 0.0))


def _detect_sections(
    y: np.ndarray, sr: int, boundaries: np.ndarray | None, labels: list[str] | None
) -> list[tuple[str, float, float]]:
    """Return list of (label, start_time, end_time) tuples.

    Uses RMS-based energy segmentation with a minimum section duration
    of 8 seconds to produce musically meaningful sections.
    """
    duration = float(len(y) / sr)
    min_section_dur = max(8.0, duration / 20.0)  # at least 8s, or 5% of track

    # Compute RMS energy contour at ~1 Hz resolution
    hop = sr  # 1 frame per second
    rms = librosa.feature.rms(y=y, frame_length=2 * hop, hop_length=hop)[0]

    if len(rms) < 4:
        # Too short — just one section
        return [("drop", 0.0, duration)]

    # Smooth the energy contour
    from scipy.ndimage import uniform_filter1d
    smoothed = uniform_filter1d(rms, size=max(3, int(min_section_dur)))

    # Find change points using derivative threshold
    diff = np.abs(np.diff(smoothed))
    threshold = np.percentile(diff, 85)
    change_points = np.where(diff > threshold)[0]

    # Convert to timestamps and enforce minimum gap
    times = [0.0]
    for cp in change_points:
        t = float(cp)
        if t - times[-1] >= min_section_dur:
            times.append(t)
    times.append(duration)

    # Cap at ~10 sections max
    while len(times) > 12:
        # Merge the shortest section
        durs = [times[i + 1] - times[i] for i in range(len(times) - 1)]
        shortest = np.argmin(durs)
        times.pop(shortest + 1 if shortest < len(times) - 2 else shortest)

    # Assign labels based on energy profile
    sections = []
    global_rms = float(np.mean(rms)) + 1e-10
    for i in range(len(times) - 1):
        s, e = times[i], times[i + 1]
        s_idx = max(0, int(s))
        e_idx = min(len(rms), int(e))
        if e_idx > s_idx:
            seg_energy = float(np.mean(rms[s_idx:e_idx])) / global_rms
        else:
            seg_energy = 0.5

        if i == 0 and seg_energy < 0.8:
            label = "intro"
        elif i == len(times) - 2 and seg_energy < 0.8:
            label = "outro"
        elif seg_energy > 1.2:
            label = "drop"
        elif seg_energy > 0.8:
            label = "build"
        else:
            label = "breakdown"

        sections.append((label, float(s), float(e)))
    return sections


def _extract_drum_pattern(
    y_perc: np.ndarray, sr: int, start: float, end: float, bpm: float
) -> DrumPattern | None:
    """Extract a 16-step drum pattern from the percussive component of a section."""
    start_sample = int(start * sr)
    end_sample = min(int(end * sr), len(y_perc))
    if end_sample - start_sample < sr // 2:
        return None

    segment = y_perc[start_sample:end_sample]
    beat_dur = 60.0 / bpm
    step_dur = beat_dur / 4.0  # 16th note
    bars = max(1, int(round((end - start) / (beat_dur * 4))))
    bars = min(bars, 8)  # cap at 8 bars

    # Frequency bands for drum components
    bands = {
        "kick": (30, 150),
        "snare": (200, 1000),
        "hihat": (6000, 16000),
        "clap": (1000, 4000),
    }

    result = {"kick": [], "snare": [], "hihat": [], "clap": []}
    for bar_idx in range(bars):
        for name, (lo, hi) in bands.items():
            steps = []
            # Bandpass filter
            nyq = sr / 2.0
            lo_norm = max(lo / nyq, 0.001)
            hi_norm = min(hi / nyq, 0.999)
            if lo_norm >= hi_norm:
                steps = [0] * 16
                result[name].append(steps)
                continue
            try:
                sos = scipy_signal.butter(4, [lo_norm, hi_norm], btype="band", output="sos")
                filtered = scipy_signal.sosfilt(sos, segment)
            except ValueError:
                steps = [0] * 16
                result[name].append(steps)
                continue

            env = np.abs(filtered)
            # Smooth envelope
            win_size = max(1, int(0.005 * sr))
            if win_size < len(env):
                kernel = np.ones(win_size) / win_size
                env = np.convolve(env, kernel, mode="same")

            threshold = np.percentile(env, 75) if len(env) > 0 else 0
            for step in range(16):
                t = bar_idx * beat_dur * 4 + step * step_dur
                sample_idx = int(t * sr)
                if sample_idx < len(env):
                    steps.append(1 if env[sample_idx] > threshold else 0)
                else:
                    steps.append(0)
            result[name].append(steps)

    return DrumPattern(
        bars=bars,
        bpm=bpm,
        kick=result["kick"],
        snare=result["snare"],
        hihat=result["hihat"],
        clap=result["clap"],
    )


def _track_bass_notes(
    y_harm: np.ndarray, sr: int, start: float, end: float
) -> list[dict]:
    """Detect bass notes from the harmonic component of a section."""
    start_sample = int(start * sr)
    end_sample = min(int(end * sr), len(y_harm))
    if end_sample - start_sample < sr // 4:
        return []

    segment = y_harm[start_sample:end_sample]

    # Low-pass filter at 300 Hz to isolate bass
    nyq = sr / 2.0
    cutoff = min(300.0 / nyq, 0.999)
    try:
        sos = scipy_signal.butter(4, cutoff, btype="low", output="sos")
        bass_signal = scipy_signal.sosfilt(sos, segment)
    except ValueError:
        return []

    # Use pyin for pitch detection
    f0, voiced, _ = librosa.pyin(
        bass_signal, fmin=30, fmax=300, sr=sr, frame_length=2048
    )
    if f0 is None or len(f0) == 0:
        return []

    times = librosa.times_like(f0, sr=sr)

    # Median-filter the f0 to remove pitch wobble (window = 5 frames)
    from scipy.ndimage import median_filter
    f0_clean = f0.copy()
    valid = voiced & np.isfinite(f0) & (f0 > 0)
    if np.sum(valid) > 5:
        midi_raw = np.where(valid, librosa.hz_to_midi(np.where(f0 > 0, f0, 1)), 0)
        midi_filt = median_filter(midi_raw, size=5)
        # Round to nearest semitone
        midi_filt = np.round(midi_filt)
    else:
        return []

    # Group consecutive same-pitch frames into notes
    beat_dur = 60.0 / 120.0  # fallback; real BPM is used in codegen
    min_note_dur = 0.06  # 60ms minimum note
    notes = []
    current_midi = None
    note_start = 0.0

    for i in range(len(midi_filt)):
        t = float(times[i]) + start
        midi = int(midi_filt[i])
        is_voiced = bool(valid[i]) and midi > 0

        if is_voiced:
            if midi != current_midi:
                # Close previous note
                if current_midi is not None:
                    dur = t - note_start
                    if dur >= min_note_dur:
                        notes.append({
                            "pitch_midi": current_midi,
                            "start": round(note_start, 4),
                            "duration": round(dur, 4),
                        })
                current_midi = midi
                note_start = t
        else:
            if current_midi is not None:
                dur = t - note_start
                if dur >= min_note_dur:
                    notes.append({
                        "pitch_midi": current_midi,
                        "start": round(note_start, 4),
                        "duration": round(dur, 4),
                    })
                current_midi = None

    # Close any open note
    if current_midi is not None:
        dur = end - note_start
        if dur >= min_note_dur:
            notes.append({
                "pitch_midi": current_midi,
                "start": round(note_start, 4),
                "duration": round(dur, 4),
            })

    return notes


def _detect_element_presence(
    y: np.ndarray, sr: int, start: float, end: float
) -> dict[str, bool]:
    """Detect whether bass, lead, pad, and vocoder elements are present."""
    start_sample = int(start * sr)
    end_sample = min(int(end * sr), len(y))
    if end_sample - start_sample < sr // 4:
        return {"bass": False, "lead": False, "pad": False, "vocoder": False}

    segment = y[start_sample:end_sample]
    S = np.abs(librosa.stft(segment))
    freqs = librosa.fft_frequencies(sr=sr)

    def band_energy(lo: float, hi: float) -> float:
        mask = (freqs >= lo) & (freqs <= hi)
        if not mask.any():
            return 0.0
        return float(np.mean(S[mask, :]))

    total_energy = float(np.mean(S)) + 1e-10
    bass_energy = band_energy(30, 300) / total_energy
    lead_energy = band_energy(300, 4000) / total_energy
    pad_energy = band_energy(200, 2000) / total_energy

    # Pad detection: look for sustained tones (low spectral flux in mid range)
    mid_mask = (freqs >= 200) & (freqs <= 2000)
    if mid_mask.any() and S.shape[1] > 1:
        mid_S = S[mid_mask, :]
        flux = np.mean(np.diff(mid_S, axis=1) ** 2)
        pad_sustained = flux < np.percentile(np.diff(S, axis=1) ** 2, 30)
    else:
        pad_sustained = False

    # Vocoder detection: look for harmonic comb pattern with modulation
    vocoder = False
    if S.shape[1] > 4:
        spectral_flatness = librosa.feature.spectral_flatness(S=S)
        flatness_std = float(np.std(spectral_flatness))
        if flatness_std > 0.1 and lead_energy > 0.3:
            vocoder = True

    return {
        "bass": bass_energy > 0.15,
        "lead": lead_energy > 0.25,
        "pad": pad_sustained and pad_energy > 0.2,
        "vocoder": vocoder,
    }


def _estimate_filter_automation(
    y: np.ndarray, sr: int, start: float, end: float
) -> FilterAutomation | None:
    """Estimate filter cutoff movement from spectral rolloff."""
    start_sample = int(start * sr)
    end_sample = min(int(end * sr), len(y))
    if end_sample - start_sample < sr // 4:
        return None

    segment = y[start_sample:end_sample]
    rolloff = librosa.feature.spectral_rolloff(y=segment, sr=sr, roll_percent=0.85)[0]
    times = librosa.times_like(rolloff, sr=sr)

    # Subsample for storage
    step = max(1, len(rolloff) // 100)
    ts = [float(t + start) for t in times[::step]]
    vals = [float(v) for v in rolloff[::step]]

    if len(vals) < 2:
        return None

    sweep_rate = float(np.mean(np.abs(np.diff(vals)) / (np.diff(ts) + 1e-10)))

    # Resonance estimate: ratio of spectral peak near rolloff to broadband
    S = np.abs(librosa.stft(segment))
    freqs = librosa.fft_frequencies(sr=sr)
    mean_rolloff = float(np.mean(rolloff))
    rolloff_bin = np.argmin(np.abs(freqs - mean_rolloff))
    nearby = S[max(0, rolloff_bin - 3) : rolloff_bin + 4, :]
    resonance = float(np.clip(np.mean(nearby) / (np.mean(S) + 1e-10) - 1.0, 0.0, 1.0))

    return FilterAutomation(
        timestamps=ts,
        cutoff_values=vals,
        resonance_estimate=resonance,
        sweep_rate_hz_per_sec=sweep_rate,
    )


def _detect_sidechain(
    y: np.ndarray, sr: int, start: float, end: float, bpm: float
) -> tuple[bool, float]:
    """Detect sidechain compression from amplitude modulation at beat rate."""
    start_sample = int(start * sr)
    end_sample = min(int(end * sr), len(y))
    if end_sample - start_sample < sr:
        return False, 0.0

    segment = y[start_sample:end_sample]
    # RMS envelope
    hop = 512
    rms = librosa.feature.rms(y=segment, hop_length=hop)[0]
    if len(rms) < 8:
        return False, 0.0

    # Expected beat period in RMS frames
    beat_period_sec = 60.0 / bpm
    beat_period_frames = beat_period_sec * sr / hop

    # Autocorrelation of RMS
    rms_centered = rms - np.mean(rms)
    autocorr = np.correlate(rms_centered, rms_centered, mode="full")
    autocorr = autocorr[len(autocorr) // 2 :]
    if len(autocorr) < 2:
        return False, 0.0
    autocorr = autocorr / (autocorr[0] + 1e-10)

    # Check for peak at beat period
    target_lag = int(round(beat_period_frames))
    search_lo = max(1, target_lag - max(2, int(target_lag * 0.15)))
    search_hi = min(len(autocorr), target_lag + max(2, int(target_lag * 0.15)) + 1)
    if search_hi <= search_lo:
        return False, 0.0

    peak_val = float(np.max(autocorr[search_lo:search_hi]))
    sidechain_active = peak_val > 0.3

    # Depth in dB
    if sidechain_active:
        rms_max = float(np.percentile(rms, 95))
        rms_min = float(np.percentile(rms, 5))
        if rms_min > 0:
            depth_db = float(20.0 * np.log10(rms_max / rms_min))
        else:
            depth_db = 20.0
    else:
        depth_db = 0.0

    return sidechain_active, depth_db


def _extract_chords(
    y_harm: np.ndarray, sr: int, start: float, end: float
) -> list[dict]:
    """Extract a rough chord sequence from the harmonic component."""
    start_sample = int(start * sr)
    end_sample = min(int(end * sr), len(y_harm))
    if end_sample - start_sample < sr // 2:
        return []

    segment = y_harm[start_sample:end_sample]
    chroma = librosa.feature.chroma_cqt(y=segment, sr=sr)
    times = librosa.times_like(chroma, sr=sr)

    note_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

    # Aggregate chroma into ~1 second windows
    hop_sec = 1.0
    hop_frames = max(1, int(hop_sec * sr / 512))
    chords = []
    for i in range(0, chroma.shape[1], hop_frames):
        window = chroma[:, i : i + hop_frames]
        profile = window.mean(axis=1)
        root = int(np.argmax(profile))
        # Simple major/minor detection
        third_major = profile[(root + 4) % 12]
        third_minor = profile[(root + 3) % 12]
        quality = "maj" if third_major >= third_minor else "min"
        chord_name = f"{note_names[root]}{quality}"
        t_start = float(times[min(i, len(times) - 1)]) + start
        t_end = float(times[min(i + hop_frames, len(times) - 1)]) + start
        chords.append({"chord": chord_name, "start": t_start, "end": t_end})

    return chords


def _estimate_effects(
    y: np.ndarray, sr: int, features
) -> dict:
    """Estimate effects chain parameters: bitcrushing, sidechain, compression."""
    # Bitcrushing detection: quantization noise shows up as high-frequency flatness
    spectral_flatness = librosa.feature.spectral_flatness(y=y)
    mean_flatness = float(np.mean(spectral_flatness))
    bitcrushing_detected = mean_flatness > 0.3

    sidechain_depth = 0.0
    if hasattr(features, "dynamics"):
        sidechain_depth = float(getattr(features.dynamics, "sidechain_depth", 0.0))

    return {
        "bitcrushing_detected": bitcrushing_detected,
        "bitcrushing_amount": min(1.0, mean_flatness / 0.5) if bitcrushing_detected else 0.0,
        "sidechain_depth": sidechain_depth,
        "sidechain_rate": float(getattr(features.dynamics, "sidechain_rate", 0.0)) if hasattr(features, "dynamics") else 0.0,
    }


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def analyze_track(audio_path: Path, sr: int = 44100) -> TrackAnalysis:
    """Perform deep analysis of a reference track.

    Steps:
    1. Load audio, compute basic features via scoring module
    2. Segment into sections using structural analysis
    3. For each section, extract detailed element-level information
    4. Estimate effects chain parameters
    5. Build the complete TrackAnalysis
    """
    audio_path = Path(audio_path)

    # 1. Load audio
    y, sr = librosa.load(str(audio_path), sr=sr, mono=True)
    duration = float(len(y) / sr)

    # Basic features from scoring module
    features = extract_features(audio_path, sr=sr)

    # BPM and key
    bpm = float(features.rhythm.tempo)
    key, _ = _estimate_key(y, sr)
    # Use scoring module's key if confidence is high
    if features.harmony.key_confidence > 0.5 and features.harmony.key:
        key = features.harmony.key

    # 2. HPSS: separate harmonic and percussive
    D = librosa.stft(y)
    H, P = librosa.decompose.hpss(D)
    y_harm = librosa.istft(H, length=len(y))
    y_perc = librosa.istft(P, length=len(y))

    # 3. Section detection
    section_boundaries = getattr(features.structure, "section_boundaries", None)
    section_labels = getattr(features.structure, "section_labels", None)
    if section_boundaries is not None:
        section_boundaries = np.array(section_boundaries)
    raw_sections = _detect_sections(y, sr, section_boundaries, section_labels)

    # 4. Global filter automation
    global_fa = _estimate_filter_automation(y, sr, 0.0, duration)
    if global_fa is None:
        global_fa = FilterAutomation(
            timestamps=[0.0, duration],
            cutoff_values=[1000.0, 1000.0],
            resonance_estimate=0.0,
            sweep_rate_hz_per_sec=0.0,
        )

    # 5. Per-section analysis
    sections: list[SectionAnalysis] = []
    for label, sec_start, sec_end in raw_sections:
        # Chords
        chords = _extract_chords(y_harm, sr, sec_start, sec_end)

        # Drum pattern
        drum_pattern = _extract_drum_pattern(y_perc, sr, sec_start, sec_end, bpm)

        # Bass notes
        bass_notes = _track_bass_notes(y_harm, sr, sec_start, sec_end)

        # Element presence
        elements = _detect_element_presence(y, sr, sec_start, sec_end)

        # Filter automation
        section_fa = _estimate_filter_automation(y, sr, sec_start, sec_end)

        # Sidechain
        sc_active, sc_depth = _detect_sidechain(y, sr, sec_start, sec_end, bpm)

        # Energy (RMS of section relative to global)
        sec_start_sample = int(sec_start * sr)
        sec_end_sample = min(int(sec_end * sr), len(y))
        if sec_end_sample > sec_start_sample:
            sec_rms = float(np.sqrt(np.mean(y[sec_start_sample:sec_end_sample] ** 2)))
            global_rms = float(np.sqrt(np.mean(y ** 2))) + 1e-10
            energy = float(np.clip(sec_rms / global_rms, 0.0, 2.0) / 2.0)
        else:
            energy = 0.0

        sections.append(SectionAnalysis(
            label=label,
            start_time=sec_start,
            end_time=sec_end,
            bpm=bpm,
            key=key,
            chord_sequence=chords,
            drum_pattern=drum_pattern,
            bass_present=elements["bass"],
            bass_notes=bass_notes,
            lead_present=elements["lead"],
            lead_notes=[],  # Lead note tracking is similar to bass but harder; use presence flag
            pad_present=elements["pad"],
            vocoder_present=elements["vocoder"],
            filter_automation=section_fa,
            energy=energy,
            sidechain_active=sc_active,
            sidechain_depth_db=sc_depth,
        ))

    # 6. Spectral character
    S = np.abs(librosa.stft(y))
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    spectral_character = {
        "mfcc_mean": np.mean(mfcc, axis=1).tolist(),
        "mfcc_std": np.std(mfcc, axis=1).tolist(),
        "centroid_mean": float(np.mean(spectral_centroid)),
        "centroid_std": float(np.std(spectral_centroid)),
    }

    # 7. Compression estimate
    rms = librosa.feature.rms(y=y)[0]
    peak = float(np.max(np.abs(y))) + 1e-10
    rms_mean = float(np.mean(rms))
    crest_factor = float(peak / (rms_mean + 1e-10))
    dynamic_range = float(20.0 * np.log10(
        (np.percentile(rms, 95) + 1e-10) / (np.percentile(rms, 5) + 1e-10)
    ))
    compression_estimate = {
        "crest_factor": crest_factor,
        "dynamic_range": dynamic_range,
    }

    # 8. Effects estimate
    effects_estimate = _estimate_effects(y, sr, features)

    return TrackAnalysis(
        file_path=str(audio_path),
        title=audio_path.stem,
        duration=duration,
        bpm=bpm,
        key=key,
        sections=sections,
        global_filter_automation=global_fa,
        spectral_character=spectral_character,
        compression_estimate=compression_estimate,
        effects_estimate=effects_estimate,
    )
