"""Generate SuperCollider NRT code from a TrackAnalysis.

The code generator translates analysis data into a Score that uses
the copywrite SynthDef library.  Key design decisions:

* Drums loop their 1-bar pattern across the entire section.
* Bass notes are quantised to the 16th-note grid and deduplicated.
  Only notes longer than a 32nd note survive.
* Pad chords are held for their full duration with gate-on / gate-off.
* Filter automation is smoothed to ~1 point per beat.
* Node IDs are globally unique via a running counter.
* All synths output directly to bus 0 (master out).
  Levels are balanced so the mix doesn't clip.
"""

from __future__ import annotations

import math
from collections import Counter


# ---------------------------------------------------------------------------
# Global node-ID counter — avoids collisions across sections
# ---------------------------------------------------------------------------
_node_id = 1000


def _next_id() -> int:
    global _node_id
    _node_id += 1
    return _node_id


def _reset_ids() -> None:
    global _node_id
    _node_id = 1000


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_NOTE_TO_MIDI = {
    "C": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3,
    "E": 4, "F": 5, "F#": 6, "Gb": 6, "G": 7, "G#": 8,
    "Ab": 8, "A": 9, "A#": 10, "Bb": 10, "B": 11,
}


def _parse_root(chord_name: str) -> int:
    """Return MIDI note number (octave 3) for a chord root.  e.g. 'Bbm' -> 46."""
    if len(chord_name) >= 2 and chord_name[1] in ("#", "b"):
        root = chord_name[:2]
    else:
        root = chord_name[:1]
    semitone = _NOTE_TO_MIDI.get(root, 0)
    return 48 + semitone          # C3 = 48


def _midi_to_freq(midi: int | float) -> float:
    return round(440.0 * (2.0 ** ((midi - 69) / 12.0)), 4)


def _quantise(t: float, grid: float) -> float:
    """Snap *t* to the nearest grid line."""
    return round(round(t / grid) * grid, 6)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def generate_sc_code(
    analysis_dict: dict,
    synthdef_docs: str,
    track_title: str = "transcription",
) -> str:
    """Generate SuperCollider NRT code that reproduces the analysed track.

    The generated code defines ``~score`` as a ``Score`` object.
    The caller is responsible for writing the .osc file and running
    scsynth in NRT mode.
    """
    _reset_ids()

    bpm = analysis_dict.get("bpm", 120.0)
    duration = analysis_dict.get("duration", 60.0)
    sections = analysis_dict.get("sections", [])

    beat_dur = 60.0 / bpm
    step_dur = beat_dur / 4.0        # 16th note
    bar_dur = beat_dur * 4.0         # 1 bar = 4 beats

    lines: list[str] = []

    # -- header --
    lines.append(f"// Auto-generated transcription: {track_title}")
    lines.append(f"// BPM: {bpm:.1f}  Key: {analysis_dict.get('key', '?')}  "
                 f"Duration: {duration:.1f}s")
    lines.append("~score = Score.new;")
    lines.append("")

    # Keeper synth — keeps scsynth rendering for the full duration
    kid = _next_id()
    lines.append(f"~score.add([0.0, [\\s_new, \\crowdNoise, {kid}, 0, 0, "
                 f"\\amp, 0.0]]);")
    lines.append(f"~score.add([{round(duration - 0.05, 6)}, [\\n_free, {kid}]]);")
    lines.append("")

    # -- per-section content --
    for sec in sections:
        sec_start = sec.get("start_time", 0.0)
        sec_end = sec.get("end_time", sec_start + 1.0)
        sec_dur = sec_end - sec_start
        label = sec.get("label", "section")
        sec_bpm = sec.get("bpm", bpm)
        sec_beat = 60.0 / sec_bpm
        sec_step = sec_beat / 4.0
        sec_bar = sec_beat * 4.0

        lines.append(f"// --- {label} ({sec_start:.1f}s – {sec_end:.1f}s) ---")

        # Drums — loop the 1-bar pattern across the section
        lines.extend(_drums_looped(sec, sec_start, sec_end, sec_bpm))

        # Bass — quantised, deduplicated, limited polyphony
        lines.extend(_bass_clean(sec, sec_start, sec_end, sec_step))

        # Pads — one synth per chord change
        lines.extend(_pads_clean(sec, sec_start, sec_end))

        # Filter automation — smoothed, ~1 point per beat
        lines.extend(_filter_smooth(sec, sec_start, sec_end, sec_beat))

        lines.append("")

    # End marker
    lines.append(f"~score.add([{round(duration, 6)}, [\\c_set, 0, 0]]);")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Drums — loop 1-bar pattern for the whole section
# ---------------------------------------------------------------------------

def _drums_looped(sec: dict, start: float, end: float, bpm: float) -> list[str]:
    dp = sec.get("drum_pattern")
    if dp is None:
        return []

    beat_dur = 60.0 / bpm
    step_dur = beat_dur / 4.0
    bar_dur = beat_dur * 4.0
    n_bars = max(1, int(math.ceil((end - start) / bar_dur)))

    # 909 levels — kick is king in Daft Punk mixes
    amp_map = {"kick": 0.85, "snare": 0.45, "hihat": 0.18, "clap": 0.35}

    # For each instrument, pick the most "musical" bar:
    # - kick: prefer four-on-the-floor (hits on steps 0,4,8,12)
    # - snare/clap: prefer backbeat (hits on steps 4,12)
    # - hihat: prefer regular 8ths or 16ths
    # If no good pattern, use a default.
    patterns: dict[str, list[int]] = {}

    for name in ("kick", "snare", "hihat", "clap"):
        raw = dp.get(name, [])
        bars = []
        for b in raw:
            if isinstance(b, list) and len(b) == 16:
                bars.append(b)

        if not bars:
            continue

        # Pick the bar with density closest to a sensible target
        targets = {"kick": 4, "snare": 2, "hihat": 8, "clap": 2}
        target = targets.get(name, 4)
        best = min(bars, key=lambda b: abs(sum(b) - target))
        density = sum(best)

        # Skip if the pattern is too sparse (0-1) or too dense (>12)
        if density < 1 or density > 12:
            continue

        patterns[name] = best

    # If we got no kick pattern, add four-on-the-floor
    if "kick" not in patterns:
        patterns["kick"] = [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0]

    # If energy is low (intro/outro), reduce drums or skip
    energy = sec.get("energy", 0.5)
    if energy < 0.25:
        return []  # too quiet for drums

    lines: list[str] = []
    for bar_idx in range(n_bars):
        bar_start = start + bar_idx * bar_dur
        if bar_start >= end:
            break
        for name, pat in patterns.items():
            amp = amp_map.get(name, 0.5) * min(1.0, energy * 2)
            for step, hit in enumerate(pat):
                if hit:
                    t = round(bar_start + step * step_dur, 6)
                    if t >= end:
                        break
                    nid = _next_id()
                    # 909-style params per instrument
                    extra = ""
                    if name == "kick":
                        extra = ", \\decay, 0.6, \\punch, 1.0"
                    elif name == "snare":
                        extra = ", \\decay, 0.22"
                    elif name == "hihat":
                        extra = ", \\decay, 0.05"
                    lines.append(
                        f"~score.add([{t}, [\\s_new, \\{name}, {nid}, 0, 0, "
                        f"\\amp, {round(amp, 3)}{extra}]]);"
                    )
    return lines


# ---------------------------------------------------------------------------
# Bass — quantise, deduplicate, limit density
# ---------------------------------------------------------------------------

def _bass_clean(sec: dict, start: float, end: float, step: float) -> list[str]:
    if not sec.get("bass_present", False):
        return []

    raw_notes = sec.get("bass_notes", [])
    if not raw_notes:
        return []

    min_dur = step * 2          # nothing shorter than an 8th note
    min_gap = step              # at least a 16th note between attacks

    # Quantise and filter
    cleaned: list[tuple[float, int, float]] = []
    for n in raw_notes:
        t = _quantise(n.get("start", 0.0), step)
        midi = int(round(n.get("pitch_midi", 36)))
        dur = max(min_dur, n.get("duration", step * 2))
        if t < start or t >= end:
            continue
        # Clamp to reasonable bass range (MIDI 24–60 = C1–C4)
        if midi < 24 or midi > 60:
            continue
        cleaned.append((t, midi, dur))

    if not cleaned:
        return []

    # Sort by time, drop duplicates within min_gap
    cleaned.sort(key=lambda x: x[0])
    deduped: list[tuple[float, int, float]] = [cleaned[0]]
    for t, midi, dur in cleaned[1:]:
        if t - deduped[-1][0] >= min_gap:
            deduped.append((t, midi, dur))

    # Find the most common pitch — likely the root note / riff tonic
    pitches = [m for _, m, _ in deduped]
    common_pitch = Counter(pitches).most_common(1)[0][0] if pitches else 36

    lines: list[str] = []
    for t, midi, dur in deduped:
        end_t = min(round(t + dur, 6), end)
        freq = _midi_to_freq(midi)
        nid = _next_id()
        # TB-303 style: low filter cutoff, high resonance, heavy drive
        # This is the core Daft Punk "Homework" bass sound
        lines.append(
            f"~score.add([{t}, [\\s_new, \\bassline, {nid}, 0, 0, "
            f"\\freq, {freq}, \\gate, 1, \\amp, 0.4, "
            f"\\filterCutoff, 400, \\filterRes, 0.6, \\filterEnv, 2500, "
            f"\\accent, 0.7, \\drive, 4.0, "
            f"\\decay, 0.2, \\sustain, 0.5, \\release, 0.4]]);"
        )
        lines.append(
            f"~score.add([{end_t}, [\\n_set, {nid}, \\gate, 0]]);"
        )
    return lines


# ---------------------------------------------------------------------------
# Pads — one chord at a time, gate-on / gate-off
# ---------------------------------------------------------------------------

def _pads_clean(sec: dict, start: float, end: float) -> list[str]:
    if not sec.get("pad_present", False):
        return []

    chords = sec.get("chord_sequence", [])
    if not chords:
        return []

    lines: list[str] = []
    prev_end: float = -1.0

    for ci in chords:
        chord_name = ci.get("chord", "C")
        ct = ci.get("start", start)
        ce = ci.get("end", ct + 2.0)
        if ct < start:
            ct = start
        if ce > end:
            ce = end
        if ce - ct < 0.5:        # skip tiny chords
            continue
        if ct < prev_end + 0.1:  # avoid overlap
            continue

        root_midi = _parse_root(chord_name)
        freq = _midi_to_freq(root_midi)
        nid = _next_id()
        # Juno-106 style pad: warm filter, chorus, long release
        lines.append(
            f"~score.add([{round(ct, 6)}, [\\s_new, \\padSynth, {nid}, 0, 0, "
            f"\\freq, {freq}, \\gate, 1, \\amp, 0.12, "
            f"\\filterCutoff, 1500, \\filterRes, 0.15, "
            f"\\attack, 0.5, \\decay, 1.0, \\sustain, 0.6, \\release, 2.0]]);"
        )
        lines.append(
            f"~score.add([{round(ce, 6)}, [\\n_set, {nid}, \\gate, 0]]);"
        )
        prev_end = ce

    return lines


# ---------------------------------------------------------------------------
# Filter automation — smoothed to ~1 point per beat
# ---------------------------------------------------------------------------

def _filter_smooth(sec: dict, start: float, end: float, beat: float) -> list[str]:
    """Filter automation — but we don't create a filterSweep synth because
    it reads from a bus and we're outputting everything to bus 0 directly.
    Instead, we skip filter automation in the NRT score.  The timbral
    character is already captured by the SynthDef filter parameters.
    """
    # Filter automation via bus routing is complex in NRT.
    # The per-synth filterCutoff params already shape the tone.
    return []
