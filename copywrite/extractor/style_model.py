"""Extract parametric style models from multiple track analyses."""

from __future__ import annotations

import json
import statistics
from collections import Counter
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# Distribution dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ParameterDistribution:
    """Statistical distribution of a parameter across tracks."""
    min: float
    max: float
    mean: float
    median: float
    stddev: float
    values: list[float]

    @classmethod
    def from_values(cls, values: list[float]) -> ParameterDistribution:
        if not values:
            return cls(min=0.0, max=0.0, mean=0.0, median=0.0, stddev=0.0, values=[])
        return cls(
            min=float(np.min(values)),
            max=float(np.max(values)),
            mean=float(np.mean(values)),
            median=float(np.median(values)),
            stddev=float(np.std(values)) if len(values) > 1 else 0.0,
            values=[float(v) for v in values],
        )

    def sample(self, rng: np.random.Generator | None = None) -> float:
        """Sample a value from this distribution (clamped gaussian)."""
        if not self.values:
            return self.mean
        rng = rng or np.random.default_rng()
        value = rng.normal(self.mean, max(self.stddev, 1e-6))
        return float(np.clip(value, self.min, self.max))


@dataclass
class PatternDistribution:
    """Distribution of rhythmic patterns across tracks."""
    patterns: list[list[int]]
    most_common: list[int]
    density_range: tuple[float, float]

    def sample(self, rng: np.random.Generator | None = None) -> list[int]:
        """Sample a pattern from the observed set."""
        if not self.patterns:
            return [0] * 16
        rng = rng or np.random.default_rng()
        idx = rng.integers(0, len(self.patterns))
        return list(self.patterns[idx])


@dataclass
class ChordVocabulary:
    """Harmonic language extracted from tracks."""
    common_keys: list[str]
    key_weights: dict[str, float]
    chord_types: list[str]
    common_progressions: list[list[str]]
    chords_per_section: ParameterDistribution

    def sample_key(self, rng: np.random.Generator | None = None) -> str:
        if not self.common_keys:
            return "Cm"
        rng = rng or np.random.default_rng()
        keys = list(self.key_weights.keys())
        weights = np.array(list(self.key_weights.values()))
        weights = weights / weights.sum()
        return str(rng.choice(keys, p=weights))

    def sample_progression(self, key: str, length: int = 4,
                           rng: np.random.Generator | None = None) -> list[str]:
        rng = rng or np.random.default_rng()
        # Try to find a progression that starts with a chord in the requested key
        matching = [p for p in self.common_progressions if len(p) >= length]
        if matching:
            prog = list(matching[rng.integers(0, len(matching))])
            return prog[:length]
        # Fall back: pick from all progressions or build from chord_types
        if self.common_progressions:
            prog = list(self.common_progressions[rng.integers(0, len(self.common_progressions))])
            # Pad or trim
            while len(prog) < length:
                prog.append(prog[-1] if prog else key)
            return prog[:length]
        # Last resort: repeat the key chord
        return [key] * length


@dataclass
class ArrangementModel:
    """How tracks are structured."""
    section_order_patterns: list[list[str]]
    section_durations: dict[str, ParameterDistribution]
    element_entry_patterns: list[dict]
    transition_types: list[str]


@dataclass
class EffectsProfile:
    """Effects chain characteristics."""
    sidechain_depth: ParameterDistribution
    sidechain_active_ratio: float
    filter_cutoff_range: ParameterDistribution
    filter_resonance: ParameterDistribution
    filter_sweep_rate: ParameterDistribution
    filter_active_ratio: float
    bitcrushing_detected_ratio: float
    compression_threshold: ParameterDistribution
    compression_ratio: ParameterDistribution


@dataclass
class StyleModel:
    """Complete parametric style model extracted from reference tracks."""
    name: str
    track_count: int

    # Tempo and groove
    bpm: ParameterDistribution
    swing: ParameterDistribution

    # Drums
    kick_patterns: PatternDistribution
    snare_patterns: PatternDistribution
    hihat_patterns: PatternDistribution

    # Harmony
    harmony: ChordVocabulary

    # Arrangement
    arrangement: ArrangementModel

    # Effects
    effects: EffectsProfile

    # Spectral character
    spectral_centroid: ParameterDistribution
    spectral_flatness: ParameterDistribution
    mfcc_profile: list[ParameterDistribution]

    # Dynamics
    crest_factor: ParameterDistribution
    dynamic_range: ParameterDistribution
    rms_mean: ParameterDistribution

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=_json_default)

    @classmethod
    def load(cls, path: Path) -> StyleModel:
        with open(path) as f:
            data = json.load(f)
        return _reconstruct_style_model(data)

    def report(self) -> str:
        """Generate a human-readable summary report."""
        lines: list[str] = []
        lines.append(f"# Style Model: {self.name}")
        lines.append(f"## Extracted from {self.track_count} tracks")
        lines.append("")

        # Tempo & Groove
        lines.append("### Tempo & Groove")
        lines.append(f"- BPM: {self.bpm.min:.0f}-{self.bpm.max:.0f} (mean: {self.bpm.mean:.1f})")
        swing_desc = "minimal" if self.swing.mean < 0.1 else "moderate" if self.swing.mean < 0.3 else "heavy"
        lines.append(f"- Swing: {swing_desc} ({self.swing.min:.2f}-{self.swing.max:.2f})")
        lines.append("")

        # Drums
        lines.append("### Drums")
        kick_density = sum(self.kick_patterns.most_common) / max(len(self.kick_patterns.most_common), 1)
        four_on_floor = (self.kick_patterns.most_common[:4] == [1, 0, 0, 0] * 1
                         if len(self.kick_patterns.most_common) >= 4 else False)
        lines.append(f"- Kick density: {kick_density:.0%} ({len(self.kick_patterns.patterns)} patterns observed)")
        snare_density = sum(self.snare_patterns.most_common) / max(len(self.snare_patterns.most_common), 1)
        lines.append(f"- Snare density: {snare_density:.0%}")
        hihat_density = sum(self.hihat_patterns.most_common) / max(len(self.hihat_patterns.most_common), 1)
        lines.append(f"- Hi-hat density: {hihat_density:.0%}")
        lines.append("")

        # Harmony
        lines.append("### Harmony")
        top_keys = self.harmony.common_keys[:5]
        lines.append(f"- Common keys: {', '.join(top_keys)}")
        lines.append(f"- Chord types: {', '.join(self.harmony.chord_types[:8])}")
        lines.append(f"- Chords/section: {self.harmony.chords_per_section.mean:.1f} (avg)")
        if self.harmony.common_progressions:
            lines.append(f"- Example progression: {' -> '.join(self.harmony.common_progressions[0])}")
        lines.append("")

        # Arrangement
        lines.append("### Arrangement")
        section_types = set()
        for pattern in self.arrangement.section_order_patterns:
            section_types.update(pattern)
        lines.append(f"- Section types: {', '.join(sorted(section_types))}")
        for sec_type, dur in self.arrangement.section_durations.items():
            lines.append(f"- {sec_type} duration: {dur.mean:.1f}s ({dur.min:.1f}-{dur.max:.1f}s)")
        lines.append("")

        # Effects
        lines.append("### Effects")
        lines.append(f"- Sidechain: {self.effects.sidechain_active_ratio:.0%} of sections "
                      f"(depth: {self.effects.sidechain_depth.mean:.1f} dB)")
        lines.append(f"- Filter sweeps: {self.effects.filter_active_ratio:.0%} of sections")
        lines.append(f"- Filter cutoff: {self.effects.filter_cutoff_range.min:.0f}-"
                      f"{self.effects.filter_cutoff_range.max:.0f} Hz")
        lines.append(f"- Bitcrushing: {self.effects.bitcrushing_detected_ratio:.0%} presence")
        lines.append("")

        # Spectral
        lines.append("### Spectral Character")
        lines.append(f"- Centroid: {self.spectral_centroid.mean:.0f} Hz "
                      f"({self.spectral_centroid.min:.0f}-{self.spectral_centroid.max:.0f})")
        lines.append(f"- Flatness: {self.spectral_flatness.mean:.4f}")
        lines.append("")

        # Dynamics
        lines.append("### Dynamics")
        lines.append(f"- Crest factor: {self.crest_factor.mean:.2f}")
        lines.append(f"- Dynamic range: {self.dynamic_range.mean:.1f} dB")
        lines.append(f"- RMS mean: {self.rms_mean.mean:.4f}")

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# JSON serialisation
# ---------------------------------------------------------------------------

def _json_default(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    if isinstance(obj, tuple):
        return list(obj)
    raise TypeError(f"Not serializable: {type(obj)}")


# ---------------------------------------------------------------------------
# Reconstruction from JSON
# ---------------------------------------------------------------------------

def _reconstruct_param_dist(data: dict) -> ParameterDistribution:
    return ParameterDistribution(
        min=data["min"],
        max=data["max"],
        mean=data["mean"],
        median=data["median"],
        stddev=data["stddev"],
        values=data["values"],
    )


def _reconstruct_pattern_dist(data: dict) -> PatternDistribution:
    density_range = data.get("density_range", [0.0, 1.0])
    return PatternDistribution(
        patterns=data["patterns"],
        most_common=data["most_common"],
        density_range=tuple(density_range),
    )


def _reconstruct_chord_vocab(data: dict) -> ChordVocabulary:
    return ChordVocabulary(
        common_keys=data["common_keys"],
        key_weights=data["key_weights"],
        chord_types=data["chord_types"],
        common_progressions=data["common_progressions"],
        chords_per_section=_reconstruct_param_dist(data["chords_per_section"]),
    )


def _reconstruct_arrangement(data: dict) -> ArrangementModel:
    section_durations = {
        k: _reconstruct_param_dist(v)
        for k, v in data["section_durations"].items()
    }
    return ArrangementModel(
        section_order_patterns=data["section_order_patterns"],
        section_durations=section_durations,
        element_entry_patterns=data["element_entry_patterns"],
        transition_types=data["transition_types"],
    )


def _reconstruct_effects(data: dict) -> EffectsProfile:
    return EffectsProfile(
        sidechain_depth=_reconstruct_param_dist(data["sidechain_depth"]),
        sidechain_active_ratio=data["sidechain_active_ratio"],
        filter_cutoff_range=_reconstruct_param_dist(data["filter_cutoff_range"]),
        filter_resonance=_reconstruct_param_dist(data["filter_resonance"]),
        filter_sweep_rate=_reconstruct_param_dist(data["filter_sweep_rate"]),
        filter_active_ratio=data["filter_active_ratio"],
        bitcrushing_detected_ratio=data["bitcrushing_detected_ratio"],
        compression_threshold=_reconstruct_param_dist(data["compression_threshold"]),
        compression_ratio=_reconstruct_param_dist(data["compression_ratio"]),
    )


def _reconstruct_style_model(data: dict) -> StyleModel:
    """Reconstruct a StyleModel from a loaded JSON dict."""
    return StyleModel(
        name=data["name"],
        track_count=data["track_count"],
        bpm=_reconstruct_param_dist(data["bpm"]),
        swing=_reconstruct_param_dist(data["swing"]),
        kick_patterns=_reconstruct_pattern_dist(data["kick_patterns"]),
        snare_patterns=_reconstruct_pattern_dist(data["snare_patterns"]),
        hihat_patterns=_reconstruct_pattern_dist(data["hihat_patterns"]),
        harmony=_reconstruct_chord_vocab(data["harmony"]),
        arrangement=_reconstruct_arrangement(data["arrangement"]),
        effects=_reconstruct_effects(data["effects"]),
        spectral_centroid=_reconstruct_param_dist(data["spectral_centroid"]),
        spectral_flatness=_reconstruct_param_dist(data["spectral_flatness"]),
        mfcc_profile=[_reconstruct_param_dist(d) for d in data["mfcc_profile"]],
        crest_factor=_reconstruct_param_dist(data["crest_factor"]),
        dynamic_range=_reconstruct_param_dist(data["dynamic_range"]),
        rms_mean=_reconstruct_param_dist(data["rms_mean"]),
    )


# ---------------------------------------------------------------------------
# Extraction helpers
# ---------------------------------------------------------------------------

def _collect_drum_patterns(
    sections: list[dict], instrument: str
) -> PatternDistribution:
    """Collect all observed patterns for a drum instrument across sections."""
    patterns: list[list[int]] = []
    for section in sections:
        drum = section.get("drum_pattern")
        if not drum:
            continue
        bars = drum.get(instrument, [])
        for bar in bars:
            if isinstance(bar, list) and any(v != 0 for v in bar):
                patterns.append(bar)

    if not patterns:
        return PatternDistribution(
            patterns=[[0] * 16],
            most_common=[0] * 16,
            density_range=(0.0, 0.0),
        )

    # Find most common pattern
    pattern_strs = [str(p) for p in patterns]
    counter = Counter(pattern_strs)
    most_common_str = counter.most_common(1)[0][0]
    # Find the actual pattern list that matches
    most_common = next(p for p in patterns if str(p) == most_common_str)

    densities = [sum(p) / len(p) for p in patterns]
    return PatternDistribution(
        patterns=patterns,
        most_common=most_common,
        density_range=(min(densities), max(densities)),
    )


def _collect_chord_vocabulary(analyses: list[dict]) -> ChordVocabulary:
    """Build a ChordVocabulary from all track analyses."""
    key_counter: Counter[str] = Counter()
    chord_type_counter: Counter[str] = Counter()
    all_progressions: list[list[str]] = []
    chords_per_section_vals: list[float] = []

    for analysis in analyses:
        key = analysis.get("key", "")
        if key:
            key_counter[key] += 1

        for section in analysis.get("sections", []):
            seq = section.get("chord_sequence", [])
            chords = [c["chord"] for c in seq if isinstance(c, dict) and "chord" in c]
            if chords:
                chords_per_section_vals.append(float(len(chords)))
                # Extract chord types
                for chord_name in chords:
                    ctype = _classify_chord_type(chord_name)
                    chord_type_counter[ctype] += 1
                # Collect progressions in groups of 4
                for i in range(0, len(chords) - 3, 4):
                    all_progressions.append(chords[i:i + 4])

            section_key = section.get("key", "")
            if section_key:
                key_counter[section_key] += 1

    common_keys = [k for k, _ in key_counter.most_common()]
    total_key_count = sum(key_counter.values()) or 1
    key_weights = {k: v / total_key_count for k, v in key_counter.items()}

    chord_types = [ct for ct, _ in chord_type_counter.most_common()]

    # Deduplicate progressions, keep most common
    prog_counter: Counter[str] = Counter()
    for prog in all_progressions:
        prog_counter[str(prog)] += 1
    common_progressions: list[list[str]] = []
    for prog_str, _ in prog_counter.most_common(20):
        # Safely reconstruct the list
        prog = next(p for p in all_progressions if str(p) == prog_str)
        common_progressions.append(prog)

    return ChordVocabulary(
        common_keys=common_keys or ["Cm"],
        key_weights=key_weights or {"Cm": 1.0},
        chord_types=chord_types or ["minor"],
        common_progressions=common_progressions,
        chords_per_section=ParameterDistribution.from_values(
            chords_per_section_vals or [4.0]
        ),
    )


def _classify_chord_type(chord_name: str) -> str:
    """Classify a chord name string into its type."""
    if not chord_name:
        return "major"
    # Strip root note
    root_end = 1
    if len(chord_name) > 1 and chord_name[1] in ("#", "b"):
        root_end = 2
    suffix = chord_name[root_end:]
    if suffix.startswith("dim"):
        return "diminished"
    if suffix.startswith("aug"):
        return "augmented"
    if suffix.startswith("m7") or suffix.startswith("min7"):
        return "minor7"
    if suffix.startswith("maj7"):
        return "major7"
    if suffix == "7" or suffix.startswith("dom"):
        return "dominant7"
    if suffix.startswith("m") or suffix.startswith("min"):
        return "minor"
    if suffix.startswith("sus"):
        return "sus"
    return "major"


def _collect_arrangement(analyses: list[dict]) -> ArrangementModel:
    """Build arrangement model from all analyses."""
    section_orders: list[list[str]] = []
    section_dur_vals: dict[str, list[float]] = {}
    element_entries: list[dict] = []
    transition_types_set: set[str] = set()

    for analysis in analyses:
        sections = analysis.get("sections", [])
        if not sections:
            continue

        order = [s["label"] for s in sections if "label" in s]
        if order:
            section_orders.append(order)

        for section in sections:
            label = section.get("label", "unknown")
            start = section.get("start_time", 0.0)
            end = section.get("end_time", 0.0)
            dur = end - start
            if dur > 0:
                section_dur_vals.setdefault(label, []).append(dur)

            # Track which elements are present in each section type
            entry = {
                "section_type": label,
                "bass": section.get("bass_present", False),
                "lead": section.get("lead_present", False),
                "pad": section.get("pad_present", False),
                "vocoder": section.get("vocoder_present", False),
                "drums": section.get("drum_pattern") is not None,
                "energy": section.get("energy", 0.5),
            }
            element_entries.append(entry)

            # Infer transition types from filter automation
            fa = section.get("filter_automation")
            if fa and fa.get("cutoff_values"):
                cutoffs = fa["cutoff_values"]
                if len(cutoffs) >= 2:
                    if cutoffs[-1] > cutoffs[0] * 3:
                        transition_types_set.add("filter_sweep_up")
                    elif cutoffs[-1] < cutoffs[0] * 0.3:
                        transition_types_set.add("filter_sweep_down")
            if section.get("energy", 0.5) < 0.3:
                transition_types_set.add("energy_drop")

    section_durations = {
        k: ParameterDistribution.from_values(v)
        for k, v in section_dur_vals.items()
    }

    # Default transition types if none detected
    if not transition_types_set:
        transition_types_set = {"filter_sweep_up", "energy_drop", "cut"}

    return ArrangementModel(
        section_order_patterns=section_orders or [["intro", "build", "drop", "breakdown", "drop", "outro"]],
        section_durations=section_durations or {
            "intro": ParameterDistribution.from_values([16.0]),
            "build": ParameterDistribution.from_values([16.0]),
            "drop": ParameterDistribution.from_values([32.0]),
            "breakdown": ParameterDistribution.from_values([16.0]),
            "outro": ParameterDistribution.from_values([16.0]),
        },
        element_entry_patterns=element_entries,
        transition_types=sorted(transition_types_set),
    )


def _collect_effects(analyses: list[dict]) -> EffectsProfile:
    """Build effects profile from all analyses."""
    sidechain_depths: list[float] = []
    sidechain_active_count = 0
    total_sections = 0
    cutoff_values: list[float] = []
    resonance_values: list[float] = []
    sweep_rates: list[float] = []
    filter_active_count = 0
    bitcrushing_count = 0
    crest_factors: list[float] = []
    dynamic_ranges: list[float] = []

    for analysis in analyses:
        # Global effects
        effects_est = analysis.get("effects_estimate", {})
        if effects_est.get("sidechain_depth"):
            sidechain_depths.append(abs(float(effects_est["sidechain_depth"])))
        if effects_est.get("bitcrushing_detected"):
            bitcrushing_count += 1

        comp_est = analysis.get("compression_estimate", {})
        if comp_est.get("crest_factor"):
            crest_factors.append(float(comp_est["crest_factor"]))
        if comp_est.get("dynamic_range"):
            dynamic_ranges.append(float(comp_est["dynamic_range"]))

        # Per-section effects
        for section in analysis.get("sections", []):
            total_sections += 1
            if section.get("sidechain_active"):
                sidechain_active_count += 1
                depth = abs(section.get("sidechain_depth_db", 0.0))
                if depth > 0:
                    sidechain_depths.append(depth)

            fa = section.get("filter_automation")
            if fa:
                for cv in fa.get("cutoff_values", []):
                    cutoff_values.append(float(cv))
                res = fa.get("resonance_estimate")
                if res is not None:
                    resonance_values.append(float(res))
                sr = fa.get("sweep_rate_hz_per_sec")
                if sr is not None:
                    sweep_rates.append(float(sr))
                if fa.get("cutoff_values"):
                    filter_active_count += 1

        # Global filter automation
        gfa = analysis.get("global_filter_automation", {})
        if gfa:
            for cv in gfa.get("cutoff_values", []):
                cutoff_values.append(float(cv))
            res = gfa.get("resonance_estimate")
            if res is not None:
                resonance_values.append(float(res))

    total_sections = max(total_sections, 1)
    total_tracks = max(len(analyses), 1)

    return EffectsProfile(
        sidechain_depth=ParameterDistribution.from_values(sidechain_depths or [6.0]),
        sidechain_active_ratio=sidechain_active_count / total_sections,
        filter_cutoff_range=ParameterDistribution.from_values(cutoff_values or [500.0, 2000.0]),
        filter_resonance=ParameterDistribution.from_values(resonance_values or [0.3]),
        filter_sweep_rate=ParameterDistribution.from_values(sweep_rates or [300.0]),
        filter_active_ratio=filter_active_count / total_sections,
        bitcrushing_detected_ratio=bitcrushing_count / total_tracks,
        compression_threshold=ParameterDistribution.from_values(
            crest_factors or [3.0]
        ),
        compression_ratio=ParameterDistribution.from_values(
            dynamic_ranges or [12.0]
        ),
    )


# ---------------------------------------------------------------------------
# Main extraction function
# ---------------------------------------------------------------------------

def extract_style_model(
    transcriptions_dir: Path, name: str = "daft_punk"
) -> StyleModel:
    """Load all TrackAnalysis JSONs and extract a parametric style model.

    Scans transcriptions_dir for *_analysis.json files, aggregates their
    parameters into distributions, and builds a StyleModel.
    """
    transcriptions_dir = Path(transcriptions_dir)
    analysis_files = sorted(transcriptions_dir.glob("*_analysis.json"))

    analyses: list[dict] = []
    for f in analysis_files:
        with open(f) as fh:
            data = json.load(fh)
        analyses.append(data)

    if not analyses:
        # Return a default style model with sensible electronic music defaults
        return _build_default_style_model(name)

    # Collect all sections across all tracks
    all_sections: list[dict] = []
    for analysis in analyses:
        all_sections.extend(analysis.get("sections", []))

    # Tempo
    bpm_values = [float(a["bpm"]) for a in analyses if a.get("bpm")]
    for s in all_sections:
        if s.get("bpm"):
            bpm_values.append(float(s["bpm"]))
    bpm_values = bpm_values or [120.0]

    # Swing - estimate from drum pattern regularity
    swing_values: list[float] = []
    for analysis in analyses:
        spectral = analysis.get("spectral_character", {})
        # Use a small default swing for electronic music
        swing_values.append(0.02)
    swing_values = swing_values or [0.0]

    # Drum patterns
    kick_patterns = _collect_drum_patterns(all_sections, "kick")
    snare_patterns = _collect_drum_patterns(all_sections, "snare")
    hihat_patterns = _collect_drum_patterns(all_sections, "hihat")

    # Harmony
    harmony = _collect_chord_vocabulary(analyses)

    # Arrangement
    arrangement = _collect_arrangement(analyses)

    # Effects
    effects = _collect_effects(analyses)

    # Spectral character
    centroid_values: list[float] = []
    flatness_values: list[float] = []
    mfcc_per_coeff: dict[int, list[float]] = {}
    for analysis in analyses:
        spectral = analysis.get("spectral_character", {})
        if spectral.get("centroid_mean"):
            centroid_values.append(float(spectral["centroid_mean"]))
        mfcc_mean = spectral.get("mfcc_mean", [])
        for i, val in enumerate(mfcc_mean):
            mfcc_per_coeff.setdefault(i, []).append(float(val))

    # Dynamics
    crest_values: list[float] = []
    dr_values: list[float] = []
    rms_values: list[float] = []
    for analysis in analyses:
        comp = analysis.get("compression_estimate", {})
        if comp.get("crest_factor"):
            crest_values.append(float(comp["crest_factor"]))
        if comp.get("dynamic_range"):
            dr_values.append(float(comp["dynamic_range"]))

    # Energy values from sections as a proxy for RMS
    for section in all_sections:
        energy = section.get("energy")
        if energy is not None:
            rms_values.append(float(energy))

    # Build MFCC profile
    n_mfcc = max(mfcc_per_coeff.keys(), default=-1) + 1 if mfcc_per_coeff else 13
    mfcc_profile = []
    for i in range(n_mfcc):
        vals = mfcc_per_coeff.get(i, [0.0])
        mfcc_profile.append(ParameterDistribution.from_values(vals))

    return StyleModel(
        name=name,
        track_count=len(analyses),
        bpm=ParameterDistribution.from_values(bpm_values),
        swing=ParameterDistribution.from_values(swing_values),
        kick_patterns=kick_patterns,
        snare_patterns=snare_patterns,
        hihat_patterns=hihat_patterns,
        harmony=harmony,
        arrangement=arrangement,
        effects=effects,
        spectral_centroid=ParameterDistribution.from_values(centroid_values or [2500.0]),
        spectral_flatness=ParameterDistribution.from_values(flatness_values or [0.01]),
        mfcc_profile=mfcc_profile or [ParameterDistribution.from_values([0.0])],
        crest_factor=ParameterDistribution.from_values(crest_values or [3.0]),
        dynamic_range=ParameterDistribution.from_values(dr_values or [12.0]),
        rms_mean=ParameterDistribution.from_values(rms_values or [0.5]),
    )


def _build_default_style_model(name: str) -> StyleModel:
    """Build a default style model when no analyses are available."""
    default_kick = [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0]
    default_snare = [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
    default_hihat = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]

    return StyleModel(
        name=name,
        track_count=0,
        bpm=ParameterDistribution.from_values([120.0]),
        swing=ParameterDistribution.from_values([0.02]),
        kick_patterns=PatternDistribution(
            patterns=[default_kick], most_common=default_kick,
            density_range=(0.25, 0.25),
        ),
        snare_patterns=PatternDistribution(
            patterns=[default_snare], most_common=default_snare,
            density_range=(0.125, 0.125),
        ),
        hihat_patterns=PatternDistribution(
            patterns=[default_hihat], most_common=default_hihat,
            density_range=(0.5, 0.5),
        ),
        harmony=ChordVocabulary(
            common_keys=["Cm"],
            key_weights={"Cm": 1.0},
            chord_types=["minor"],
            common_progressions=[["Cm", "Fm", "Ab", "Gm"]],
            chords_per_section=ParameterDistribution.from_values([4.0]),
        ),
        arrangement=ArrangementModel(
            section_order_patterns=[["intro", "build", "drop", "breakdown", "drop", "outro"]],
            section_durations={
                "intro": ParameterDistribution.from_values([16.0]),
                "build": ParameterDistribution.from_values([16.0]),
                "drop": ParameterDistribution.from_values([32.0]),
                "breakdown": ParameterDistribution.from_values([16.0]),
                "outro": ParameterDistribution.from_values([16.0]),
            },
            element_entry_patterns=[],
            transition_types=["filter_sweep_up", "energy_drop", "cut"],
        ),
        effects=EffectsProfile(
            sidechain_depth=ParameterDistribution.from_values([6.0]),
            sidechain_active_ratio=0.6,
            filter_cutoff_range=ParameterDistribution.from_values([500.0, 2000.0]),
            filter_resonance=ParameterDistribution.from_values([0.3]),
            filter_sweep_rate=ParameterDistribution.from_values([300.0]),
            filter_active_ratio=0.4,
            bitcrushing_detected_ratio=0.0,
            compression_threshold=ParameterDistribution.from_values([3.0]),
            compression_ratio=ParameterDistribution.from_values([12.0]),
        ),
        spectral_centroid=ParameterDistribution.from_values([2500.0]),
        spectral_flatness=ParameterDistribution.from_values([0.01]),
        mfcc_profile=[ParameterDistribution.from_values([0.0])],
        crest_factor=ParameterDistribution.from_values([3.0]),
        dynamic_range=ParameterDistribution.from_values([12.0]),
        rms_mean=ParameterDistribution.from_values([0.5]),
    )
