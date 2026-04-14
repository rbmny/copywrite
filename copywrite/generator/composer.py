"""Generate new tracks constrained by a StyleModel."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from rich.console import Console

from copywrite.config import CopywriteConfig, load_config
from copywrite.engine import NRTRenderer, SynthDefManager
from copywrite.extractor.style_model import StyleModel

console = Console()

# ---------------------------------------------------------------------------
# Music theory constants
# ---------------------------------------------------------------------------

_NOTE_TO_SEMITONE: dict[str, int] = {
    "C": 0, "C#": 1, "Db": 1,
    "D": 2, "D#": 3, "Eb": 3,
    "E": 4, "Fb": 4, "E#": 5,
    "F": 5, "F#": 6, "Gb": 6,
    "G": 7, "G#": 8, "Ab": 8,
    "A": 9, "A#": 10, "Bb": 10,
    "B": 11, "Cb": 11, "B#": 0,
}

_CHORD_INTERVALS: dict[str, list[int]] = {
    "": [0, 4, 7],           # major
    "maj": [0, 4, 7],
    "m": [0, 3, 7],          # minor
    "min": [0, 3, 7],
    "7": [0, 4, 7, 10],      # dominant 7
    "dom7": [0, 4, 7, 10],
    "m7": [0, 3, 7, 10],     # minor 7
    "min7": [0, 3, 7, 10],
    "maj7": [0, 4, 7, 11],   # major 7
    "dim": [0, 3, 6],        # diminished
    "aug": [0, 4, 8],        # augmented
    "sus2": [0, 2, 7],
    "sus4": [0, 5, 7],
    "dim7": [0, 3, 6, 9],
    "m9": [0, 3, 7, 10, 14],
    "9": [0, 4, 7, 10, 14],
    "add9": [0, 4, 7, 14],
}

# Section type -> likely elements present
_SECTION_ELEMENTS: dict[str, dict[str, float]] = {
    "intro": {"drums": 0.4, "bass": 0.3, "pad": 0.7, "lead": 0.0, "vocoder": 0.1},
    "build": {"drums": 0.8, "bass": 0.7, "pad": 0.6, "lead": 0.3, "vocoder": 0.2},
    "drop": {"drums": 1.0, "bass": 1.0, "pad": 0.4, "lead": 0.6, "vocoder": 0.3},
    "breakdown": {"drums": 0.3, "bass": 0.4, "pad": 0.8, "lead": 0.2, "vocoder": 0.2},
    "outro": {"drums": 0.5, "bass": 0.3, "pad": 0.6, "lead": 0.0, "vocoder": 0.0},
    "transition": {"drums": 0.2, "bass": 0.2, "pad": 0.5, "lead": 0.0, "vocoder": 0.1},
}


# ---------------------------------------------------------------------------
# GeneratedTrack dataclass
# ---------------------------------------------------------------------------

@dataclass
class GeneratedTrack:
    """A generated track with its metadata."""
    title: str
    sc_code: str
    duration: float
    bpm: float
    key: str
    parameters: dict
    output_path: Path | None = None
    score: float | None = None


# ---------------------------------------------------------------------------
# Music theory helpers
# ---------------------------------------------------------------------------

def _note_name_to_midi(note: str, octave: int = 4) -> int:
    """Convert note name (e.g., 'C', 'F#', 'Bb') to MIDI number."""
    note = note.strip()
    if not note:
        return 60  # middle C
    semitone = _NOTE_TO_SEMITONE.get(note)
    if semitone is None:
        return 60
    return semitone + (octave + 1) * 12


def _parse_chord_name(chord: str) -> tuple[str, str]:
    """Parse a chord string into (root_note, quality).

    Examples: 'Cm' -> ('C', 'm'), 'F#m7' -> ('F#', 'm7'),
              'Bb' -> ('Bb', ''), 'Abdim' -> ('Ab', 'dim')
    """
    if not chord:
        return ("C", "")
    # Extract root: 1 or 2 chars
    root_end = 1
    if len(chord) > 1 and chord[1] in ("#", "b"):
        root_end = 2
    root = chord[:root_end]
    quality = chord[root_end:]
    return (root, quality)


def _midi_note_for_chord(chord: str, octave: int = 4) -> list[int]:
    """Convert a chord name to MIDI note numbers."""
    root, quality = _parse_chord_name(chord)
    root_midi = _note_name_to_midi(root, octave)
    intervals = _CHORD_INTERVALS.get(quality, _CHORD_INTERVALS[""])
    return [root_midi + iv for iv in intervals]


# ---------------------------------------------------------------------------
# Arrangement generation
# ---------------------------------------------------------------------------

def _generate_arrangement(
    style_model: StyleModel, duration: float, bpm: float,
    rng: np.random.Generator,
) -> list[dict]:
    """Generate section structure."""
    # Pick a section order from the style model
    patterns = style_model.arrangement.section_order_patterns
    if patterns:
        order = list(patterns[rng.integers(0, len(patterns))])
    else:
        order = ["intro", "build", "drop", "breakdown", "drop", "outro"]

    # Calculate target section durations that sum to total duration
    raw_durations: list[float] = []
    for section_type in order:
        dist = style_model.arrangement.section_durations.get(section_type)
        if dist:
            raw_durations.append(max(4.0, dist.sample(rng)))
        else:
            raw_durations.append(16.0)

    # Scale durations to fit target duration
    total_raw = sum(raw_durations)
    scale = duration / max(total_raw, 1.0)
    durations = [d * scale for d in raw_durations]

    # Quantize to nearest bar (4 beats)
    bar_duration = 4 * 60.0 / bpm
    durations = [max(bar_duration, round(d / bar_duration) * bar_duration) for d in durations]

    # Adjust last section to match total duration
    current_total = sum(durations)
    if current_total != duration:
        durations[-1] = max(bar_duration, durations[-1] + (duration - current_total))

    # Build section list
    sections: list[dict] = []
    current_time = 0.0
    for i, section_type in enumerate(order):
        section_dur = durations[i]
        sections.append({
            "type": section_type,
            "start_time": current_time,
            "end_time": current_time + section_dur,
            "duration": section_dur,
        })
        current_time += section_dur

    return sections


def _generate_section_content(
    section_type: str, style_model: StyleModel,
    key: str, bpm: float, section_duration: float,
    rng: np.random.Generator,
) -> dict:
    """Generate musical content for one section."""
    content: dict = {"type": section_type}

    # Determine which elements are present
    probs = _SECTION_ELEMENTS.get(section_type, _SECTION_ELEMENTS["drop"])
    # Modulate by style model element entry patterns
    element_stats = _compute_element_stats(style_model, section_type)
    elements_present = {}
    for element, base_prob in probs.items():
        stat_prob = element_stats.get(element, base_prob)
        combined = (base_prob + stat_prob) / 2.0
        elements_present[element] = rng.random() < combined
    content["elements"] = elements_present

    bar_duration = 4 * 60.0 / bpm
    n_bars = max(1, int(round(section_duration / bar_duration)))

    # Drums
    if elements_present.get("drums"):
        content["kick"] = style_model.kick_patterns.sample(rng)
        content["snare"] = style_model.snare_patterns.sample(rng)
        content["hihat"] = style_model.hihat_patterns.sample(rng)
    else:
        content["kick"] = [0] * 16
        content["snare"] = [0] * 16
        content["hihat"] = [0] * 16
    content["n_bars"] = n_bars

    # Chord progression
    progression_length = max(1, min(n_bars, 8))
    progression = style_model.harmony.sample_progression(key, progression_length, rng)
    content["chords"] = progression

    # Bass line: follow chord roots
    if elements_present.get("bass"):
        bass_notes: list[dict] = []
        beat_dur = 60.0 / bpm
        for bar_idx in range(n_bars):
            chord_idx = bar_idx % len(progression)
            root, _ = _parse_chord_name(progression[chord_idx])
            root_midi = _note_name_to_midi(root, octave=2)
            bar_start = bar_idx * bar_duration
            # Bass plays a simple pattern: root on beats 1 and 3
            for beat_offset in [0.0, 2.0 * beat_dur]:
                bass_notes.append({
                    "pitch_midi": root_midi,
                    "start": bar_start + beat_offset,
                    "duration": beat_dur * 1.5,
                })
        content["bass_notes"] = bass_notes
    else:
        content["bass_notes"] = []

    # Pad chords
    if elements_present.get("pad"):
        pad_notes: list[dict] = []
        for bar_idx in range(n_bars):
            chord_idx = bar_idx % len(progression)
            midi_notes = _midi_note_for_chord(progression[chord_idx], octave=4)
            bar_start = bar_idx * bar_duration
            pad_notes.append({
                "midi_notes": midi_notes,
                "start": bar_start,
                "duration": bar_duration,
            })
        content["pad_notes"] = pad_notes
    else:
        content["pad_notes"] = []

    # Lead melody: simple arpeggiation of chord tones in drop/build sections
    if elements_present.get("lead"):
        lead_notes: list[dict] = []
        beat_dur = 60.0 / bpm
        sixteenth = beat_dur / 4.0
        for bar_idx in range(n_bars):
            chord_idx = bar_idx % len(progression)
            midi_notes = _midi_note_for_chord(progression[chord_idx], octave=5)
            bar_start = bar_idx * bar_duration
            # Arpeggiate chord tones on 8th notes
            for step in range(8):
                note_idx = step % len(midi_notes)
                lead_notes.append({
                    "pitch_midi": midi_notes[note_idx],
                    "start": bar_start + step * beat_dur / 2.0,
                    "duration": beat_dur / 2.0,
                })
        content["lead_notes"] = lead_notes
    else:
        content["lead_notes"] = []

    # Energy level
    energy_map = {
        "intro": 0.3, "build": 0.6, "drop": 0.9,
        "breakdown": 0.4, "outro": 0.3, "transition": 0.2,
    }
    content["energy"] = energy_map.get(section_type, 0.5)

    # Sidechain
    content["sidechain"] = (
        section_type in ("drop", "build")
        and rng.random() < style_model.effects.sidechain_active_ratio
    )

    # Filter automation
    content["filter_sweep"] = section_type in ("build", "transition")

    return content


def _compute_element_stats(style_model: StyleModel, section_type: str) -> dict[str, float]:
    """Compute element presence probabilities from style model for a section type."""
    entries = [e for e in style_model.arrangement.element_entry_patterns
               if e.get("section_type") == section_type]
    if not entries:
        return {}
    stats: dict[str, float] = {}
    for element in ("drums", "bass", "lead", "pad", "vocoder"):
        values = [1.0 if e.get(element) else 0.0 for e in entries]
        stats[element] = float(np.mean(values))
    return stats


# ---------------------------------------------------------------------------
# SuperCollider code generation
# ---------------------------------------------------------------------------

def _build_sc_code(
    arrangement: list[dict], bpm: float, key: str,
    effects_params: dict, duration: float,
    synthdef_docs: str,
) -> str:
    """Build complete SuperCollider NRT code from generated arrangement."""
    lines: list[str] = []
    lines.append("(")
    lines.append(f"// Generated track - BPM: {bpm}, Key: {key}")
    lines.append(f"// Duration: {duration:.1f}s")
    lines.append(f"// SynthDefs available: {synthdef_docs.splitlines()[0] if synthdef_docs else 'standard'}")
    lines.append("")
    lines.append("var score = Score.new;")
    lines.append(f"var bpm = {bpm};")
    lines.append(f"var beatDur = 60.0 / {bpm};")
    lines.append("var sixteenth = beatDur / 4.0;")
    lines.append("")

    # Allocate buses for effects routing
    lines.append("// Effect buses")
    lines.append("var sidechainBus = 16;")
    lines.append("var filterBus = 18;")
    lines.append("")

    # Node ID counter
    node_id = 1000

    # Set up effect synths at time 0
    sidechain_depth = effects_params.get("sidechain_depth", 6.0)
    filter_cutoff = effects_params.get("filter_cutoff", 1000.0)
    filter_res = effects_params.get("filter_resonance", 0.3)
    compression_ratio = effects_params.get("compression_ratio", 4.0)
    bitcrushing = effects_params.get("bitcrushing", False)

    # Sidechain compressor on main bus
    lines.append(f"// Master effects chain")
    lines.append(f"score.add([0.0, [\\s_new, \\alesis3630, {node_id}, 0, 0, "
                 f"\\in, 0, \\threshold, -10, \\ratio, {compression_ratio:.1f}]]);")
    node_id += 1

    if bitcrushing:
        lines.append(f"score.add([0.0, [\\s_new, \\sp1200, {node_id}, 0, 0, "
                     f"\\in, 0, \\bits, 12, \\rate, 22050]]);")
        node_id += 1

    lines.append("")

    # Generate events for each section
    for section in arrangement:
        sec_content = section.get("content", {})
        sec_type = section.get("type", "drop")
        sec_start = section.get("start_time", 0.0)
        sec_dur = section.get("duration", 16.0)
        n_bars = sec_content.get("n_bars", 4)

        bar_dur = 4.0 * 60.0 / bpm
        beat_dur = 60.0 / bpm
        sixteenth_dur = beat_dur / 4.0

        lines.append(f"// --- Section: {sec_type} (t={sec_start:.1f}s, {sec_dur:.1f}s) ---")

        # Sidechain for this section
        if sec_content.get("sidechain"):
            lines.append(f"score.add([{sec_start:.4f}, [\\s_new, \\sidechain, {node_id}, 0, 0, "
                         f"\\in, 0, \\depth, {sidechain_depth:.1f}, "
                         f"\\rate, {bpm / 60.0:.4f}]]);")
            sc_node = node_id
            node_id += 1
            sec_end = sec_start + sec_dur
            lines.append(f"score.add([{sec_end:.4f}, [\\n_free, {sc_node}]]);")

        # Filter sweep for builds/transitions
        if sec_content.get("filter_sweep"):
            lines.append(f"score.add([{sec_start:.4f}, [\\s_new, \\filterSweep, {node_id}, 0, 0, "
                         f"\\in, 0, \\startFreq, 200, \\endFreq, {filter_cutoff:.0f}, "
                         f"\\dur, {sec_dur:.4f}, \\rq, {filter_res:.2f}]]);")
            fs_node = node_id
            node_id += 1
            sec_end = sec_start + sec_dur
            lines.append(f"score.add([{sec_end:.4f}, [\\n_free, {fs_node}]]);")

        # Drums
        kick = sec_content.get("kick", [0] * 16)
        snare = sec_content.get("snare", [0] * 16)
        hihat = sec_content.get("hihat", [0] * 16)

        for bar_idx in range(n_bars):
            bar_start = sec_start + bar_idx * bar_dur
            if bar_start >= sec_start + sec_dur:
                break
            for step in range(16):
                step_time = bar_start + step * sixteenth_dur
                if step_time >= sec_start + sec_dur:
                    break
                if kick[step % len(kick)]:
                    lines.append(f"score.add([{step_time:.4f}, [\\s_new, \\kick, {node_id}, 0, 0]]);")
                    node_id += 1
                if snare[step % len(snare)]:
                    lines.append(f"score.add([{step_time:.4f}, [\\s_new, \\snare, {node_id}, 0, 0]]);")
                    node_id += 1
                if hihat[step % len(hihat)]:
                    lines.append(f"score.add([{step_time:.4f}, [\\s_new, \\hihat, {node_id}, 0, 0, "
                                 f"\\amp, 0.3]]);")
                    node_id += 1

        # Bass
        for note in sec_content.get("bass_notes", []):
            t = sec_start + note["start"]
            if t >= sec_start + sec_dur:
                break
            midi = note["pitch_midi"]
            dur = note["duration"]
            freq = 440.0 * (2.0 ** ((midi - 69) / 12.0))
            lines.append(f"score.add([{t:.4f}, [\\s_new, \\bassline, {node_id}, 0, 0, "
                         f"\\freq, {freq:.2f}, \\dur, {dur:.4f}, \\amp, 0.5]]);")
            node_id += 1

        # Pad chords
        for pad in sec_content.get("pad_notes", []):
            t = sec_start + pad["start"]
            if t >= sec_start + sec_dur:
                break
            dur = pad["duration"]
            for midi in pad["midi_notes"]:
                freq = 440.0 * (2.0 ** ((midi - 69) / 12.0))
                lines.append(f"score.add([{t:.4f}, [\\s_new, \\padSynth, {node_id}, 0, 0, "
                             f"\\freq, {freq:.2f}, \\dur, {dur:.4f}, \\amp, 0.15]]);")
                node_id += 1

        # Lead melody
        for note in sec_content.get("lead_notes", []):
            t = sec_start + note["start"]
            if t >= sec_start + sec_dur:
                break
            midi = note["pitch_midi"]
            dur = note["duration"]
            freq = 440.0 * (2.0 ** ((midi - 69) / 12.0))
            lines.append(f"score.add([{t:.4f}, [\\s_new, \\miniVoyager, {node_id}, 0, 0, "
                         f"\\freq, {freq:.2f}, \\dur, {dur:.4f}, \\amp, 0.3]]);")
            node_id += 1

        lines.append("")

    # End marker
    lines.append(f"// End marker")
    lines.append(f"score.add([{duration:.4f}, [\\c_set, 0, 0]]);")
    lines.append("")
    lines.append("~score = score;")
    lines.append(")")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main composition function
# ---------------------------------------------------------------------------

def compose_track(
    style_model: StyleModel,
    synthdef_docs: str,
    duration: float = 90.0,
    bpm: float | None = None,
    key: str | None = None,
    seed: int | None = None,
    title: str = "generated",
) -> GeneratedTrack:
    """Compose a new track constrained by the style model.

    Steps:
    1. Sample global parameters from style model (BPM, key, or use overrides)
    2. Generate arrangement structure following style model patterns
    3. For each section, generate musical content
    4. Apply effects parameters from style model
    5. Generate complete SuperCollider code
    6. Return GeneratedTrack (not yet rendered)
    """
    rng = np.random.default_rng(seed)

    # 1. Sample global parameters
    track_bpm = bpm if bpm is not None else style_model.bpm.sample(rng)
    track_bpm = max(60.0, min(200.0, track_bpm))
    track_key = key if key is not None else style_model.harmony.sample_key(rng)

    # 2. Generate arrangement
    arrangement = _generate_arrangement(style_model, duration, track_bpm, rng)

    # 3. Generate content for each section
    for section in arrangement:
        content = _generate_section_content(
            section["type"], style_model, track_key, track_bpm,
            section["duration"], rng,
        )
        section["content"] = content

    # 4. Effects parameters
    effects_params = {
        "sidechain_depth": style_model.effects.sidechain_depth.sample(rng),
        "filter_cutoff": style_model.effects.filter_cutoff_range.sample(rng),
        "filter_resonance": style_model.effects.filter_resonance.sample(rng),
        "compression_ratio": style_model.effects.compression_ratio.sample(rng),
        "bitcrushing": rng.random() < style_model.effects.bitcrushing_detected_ratio,
    }

    # 5. Generate SC code
    sc_code = _build_sc_code(
        arrangement, track_bpm, track_key,
        effects_params, duration, synthdef_docs,
    )

    # Collect all sampled parameters for metadata
    parameters = {
        "bpm": track_bpm,
        "key": track_key,
        "seed": seed,
        "arrangement": [
            {"type": s["type"], "start": s["start_time"],
             "end": s["end_time"], "duration": s["duration"]}
            for s in arrangement
        ],
        "effects": effects_params,
    }

    return GeneratedTrack(
        title=title,
        sc_code=sc_code,
        duration=duration,
        bpm=track_bpm,
        key=track_key,
        parameters=parameters,
    )


# ---------------------------------------------------------------------------
# Generate, render, and score
# ---------------------------------------------------------------------------

def generate_and_render(
    style_model: StyleModel,
    config: CopywriteConfig,
    count: int = 3,
    duration: float = 90.0,
    bpm: float | None = None,
    key: str | None = None,
) -> list[GeneratedTrack]:
    """Generate multiple tracks, render them, and optionally score them.

    1. For each track: compose -> render -> score against style model bounds
    2. Save SC code, rendered wav, and metadata for each
    3. Return list of GeneratedTrack with paths and scores filled in
    """
    config.ensure_dirs()
    renderer = NRTRenderer(config)
    synthdef_mgr = SynthDefManager(config)

    try:
        synthdef_docs = synthdef_mgr.get_synthdef_docs()
    except FileNotFoundError:
        synthdef_docs = ""
        console.print("[yellow]SynthDef library not found, using minimal docs[/yellow]")

    tracks: list[GeneratedTrack] = []

    console.print(f"[bold cyan]Generating {count} tracks...[/bold cyan]")
    console.print(f"  Style: {style_model.name} ({style_model.track_count} reference tracks)")
    console.print(f"  Duration: {duration:.0f}s")
    if bpm:
        console.print(f"  BPM: {bpm}")
    if key:
        console.print(f"  Key: {key}")
    console.print()

    for i in range(count):
        seed = np.random.default_rng().integers(0, 2**31)
        title = f"{style_model.name}_gen_{i + 1:03d}"

        console.print(f"[cyan]Track {i + 1}/{count}: {title}[/cyan]")

        # Compose
        console.print("  Composing...")
        track = compose_track(
            style_model, synthdef_docs,
            duration=duration, bpm=bpm, key=key,
            seed=int(seed), title=title,
        )

        # Save SC code
        sc_path = config.generated_dir / f"{title}.scd"
        sc_path.parent.mkdir(parents=True, exist_ok=True)
        with open(sc_path, "w") as f:
            f.write(track.sc_code)
        console.print(f"  SC code saved: {sc_path}")

        # Render
        wav_path = config.generated_dir / f"{title}.wav"
        console.print("  Rendering...")
        try:
            track.output_path = renderer.render(track.sc_code, wav_path, duration)
            console.print(f"  [green]Rendered: {track.output_path}[/green]")
        except RuntimeError as e:
            console.print(f"  [red]Render failed: {e}[/red]")
            track.output_path = None

        # Score against style model bounds
        track.score = _score_against_model(track, style_model)
        if track.score is not None:
            console.print(f"  Score: {track.score:.2f}")

        # Save metadata
        meta_path = config.generated_dir / f"{title}_meta.json"
        meta = {
            "title": track.title,
            "duration": track.duration,
            "bpm": track.bpm,
            "key": track.key,
            "parameters": track.parameters,
            "score": track.score,
            "output_path": str(track.output_path) if track.output_path else None,
            "sc_path": str(sc_path),
        }
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2, default=_json_default)

        tracks.append(track)
        console.print()

    # Summary
    console.print("[bold green]Generation complete![/bold green]")
    scored = [t for t in tracks if t.score is not None]
    if scored:
        best = max(scored, key=lambda t: t.score)
        console.print(f"  Best track: {best.title} (score: {best.score:.2f})")
        avg_score = np.mean([t.score for t in scored])
        console.print(f"  Average score: {avg_score:.2f}")

    return tracks


def _score_against_model(track: GeneratedTrack, style_model: StyleModel) -> float:
    """Score a generated track against the style model parameter bounds.

    Returns a 0-1 score based on how well the track's parameters fall
    within the style model's distributions.
    """
    scores: list[float] = []

    # BPM score
    bpm_score = _param_score(track.bpm, style_model.bpm)
    scores.append(bpm_score)

    # Key match score
    key_score = 1.0 if track.key in style_model.harmony.key_weights else 0.5
    scores.append(key_score)

    # Effects parameter scores
    effects = track.parameters.get("effects", {})
    if "sidechain_depth" in effects:
        scores.append(_param_score(effects["sidechain_depth"],
                                   style_model.effects.sidechain_depth))
    if "filter_cutoff" in effects:
        scores.append(_param_score(effects["filter_cutoff"],
                                   style_model.effects.filter_cutoff_range))
    if "compression_ratio" in effects:
        scores.append(_param_score(effects["compression_ratio"],
                                   style_model.effects.compression_ratio))

    return float(np.mean(scores)) if scores else 0.5


def _param_score(value: float, dist) -> float:
    """Score how well a value fits within a ParameterDistribution."""
    if dist.stddev == 0:
        return 1.0 if abs(value - dist.mean) < 1e-6 else 0.5
    z = abs(value - dist.mean) / max(dist.stddev, 1e-6)
    # Within 1 stddev = great, 2 = okay, beyond = poor
    if z <= 1.0:
        return 1.0
    elif z <= 2.0:
        return 0.7
    elif z <= 3.0:
        return 0.4
    return 0.2


def _json_default(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Not serializable: {type(obj)}")
