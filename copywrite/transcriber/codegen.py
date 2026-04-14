"""Generate SuperCollider NRT code from a TrackAnalysis."""

from __future__ import annotations

from pathlib import Path


def generate_sc_code(
    analysis_dict: dict,
    synthdef_docs: str,
    track_title: str = "transcription",
) -> str:
    """Generate SuperCollider NRT code that reproduces the analyzed track.

    Builds SC code that:
    1. Sets up the NRT Score with buses and groups
    2. Creates synth instances for each element in each section
    3. Applies filter automation via n_set messages
    4. Applies effects (sidechain, bitcrushing, compression)
    5. Follows the arrangement structure

    Args:
        analysis_dict: TrackAnalysis as a dict.
        synthdef_docs: Documentation string for available SynthDefs.
        track_title: Name for the output.

    Returns:
        Complete SuperCollider code string ready for NRT rendering.
    """
    bpm = analysis_dict.get("bpm", 120.0)
    duration = analysis_dict.get("duration", 60.0)
    sections = analysis_dict.get("sections", [])
    effects = analysis_dict.get("effects_estimate", {})

    lines: list[str] = []
    lines.append(_build_score_header(bpm, duration))

    # Effects chain setup
    lines.extend(_build_effects_chain(analysis_dict))

    # Track which bar we're on for each section
    beat_dur = 60.0 / bpm
    bar_dur = beat_dur * 4.0

    for section in sections:
        start_bar = max(0, int(round(section["start_time"] / bar_dur)))
        lines.append(f"// --- Section: {section.get('label', 'unknown')} "
                      f"(bars {start_bar}+) ---")

        lines.extend(_build_drum_events(section, start_bar))
        lines.extend(_build_bass_events(section, start_bar))
        lines.extend(_build_chord_events(section, start_bar))
        lines.extend(_build_filter_automation(section, start_bar))

    lines.append("")
    lines.append(f"// End marker")
    lines.append(f"~score.add([{duration}, [\\c_set, 0, 0]]);")
    lines.append("")

    return "\n".join(lines)


def _build_score_header(bpm: float, duration: float) -> str:
    """Build the Score setup boilerplate."""
    return (
        f"// Auto-generated transcription — BPM: {bpm}, duration: {duration:.1f}s\n"
        f"~score = Score.new;\n"
        f"\n"
        f"// Group structure: sources -> effects -> master\n"
        f"~score.add([0.0, [\\g_new, 100, 0, 0]]);  // source group\n"
        f"~score.add([0.0, [\\g_new, 200, 3, 100]]);  // effects group\n"
        f"\n"
        f"// Silent keeper synth to prevent early termination\n"
        f"~score.add([0.0, [\\s_new, \\crowdNoise, 99, 0, 100, \\amp, 0.0]]);\n"
        f"~score.add([{duration - 0.1}, [\\n_free, 99]]);\n"
    )


def _build_drum_events(section: dict, start_bar: int) -> list[str]:
    """Generate Score entries for drum patterns."""
    dp = section.get("drum_pattern")
    if dp is None:
        return []

    bpm = dp.get("bpm", section.get("bpm", 120.0))
    beat_dur = 60.0 / bpm
    step_dur = beat_dur / 4.0
    bars = dp.get("bars", 1)

    kick_patterns = dp.get("kick", [])
    snare_patterns = dp.get("snare", [])
    hihat_patterns = dp.get("hihat", [])
    clap_patterns = dp.get("clap", [])

    lines: list[str] = []
    node_id = 1000 + start_bar * 100

    synth_map = {
        "kick": ("kick", kick_patterns),
        "snare": ("snare", snare_patterns),
        "hihat": ("hihat", hihat_patterns),
        "clap": ("clap", clap_patterns),
    }

    for name, (synthdef_name, patterns) in synth_map.items():
        for bar_idx in range(min(bars, len(patterns))):
            pattern = patterns[bar_idx]
            for step, hit in enumerate(pattern):
                if hit:
                    t = (start_bar + bar_idx) * beat_dur * 4 + step * step_dur
                    t = round(t, 6)
                    node_id += 1
                    lines.append(
                        f"~score.add([{t}, [\\s_new, \\{synthdef_name}, {node_id}, 0, 100]]);"
                    )

    return lines


def _build_bass_events(section: dict, start_bar: int) -> list[str]:
    """Generate Score entries for bass notes."""
    if not section.get("bass_present", False):
        return []

    bass_notes = section.get("bass_notes", [])
    if not bass_notes:
        return []

    lines: list[str] = []
    node_id = 3000 + start_bar * 100

    for note in bass_notes:
        midi = note.get("pitch_midi", 36)
        start_t = round(note.get("start", 0.0), 6)
        dur = max(0.05, round(note.get("duration", 0.5), 6))
        end_t = round(start_t + dur, 6)
        freq = round(440.0 * (2.0 ** ((midi - 69) / 12.0)), 4)
        node_id += 1

        lines.append(
            f"~score.add([{start_t}, [\\s_new, \\bassline, {node_id}, 0, 100, "
            f"\\freq, {freq}, \\gate, 1, \\amp, 0.5]]);"
        )
        lines.append(
            f"~score.add([{end_t}, [\\n_set, {node_id}, \\gate, 0]]);"
        )

    return lines


def _build_chord_events(section: dict, start_bar: int) -> list[str]:
    """Generate Score entries for chord pads."""
    if not section.get("pad_present", False):
        return []

    chords = section.get("chord_sequence", [])
    if not chords:
        return []

    lines: list[str] = []
    node_id = 5000 + start_bar * 100

    note_to_midi = {
        "C": 60, "C#": 61, "D": 62, "D#": 63, "E": 64, "F": 65,
        "F#": 66, "G": 67, "G#": 68, "A": 69, "A#": 70, "B": 71,
    }

    for chord_info in chords:
        chord_name = chord_info.get("chord", "Cmaj")
        start_t = round(chord_info.get("start", 0.0), 6)
        end_t = chord_info.get("end", start_t + 1.0)
        dur = max(0.1, round(end_t - start_t, 6))

        # Parse chord name: root note + quality
        root_name = chord_name.rstrip("majmindimaug")
        if not root_name:
            root_name = "C"
        # Handle sharp notes
        if len(chord_name) > 1 and chord_name[1] == "#":
            root_name = chord_name[:2]
        elif len(root_name) == 0:
            root_name = chord_name[0]

        root_midi = note_to_midi.get(root_name, 60)
        freq = round(440.0 * (2.0 ** ((root_midi - 69) / 12.0)), 4)

        end_t = round(start_t + dur, 6)
        node_id += 1
        lines.append(
            f"~score.add([{start_t}, [\\s_new, \\padSynth, {node_id}, 0, 100, "
            f"\\freq, {freq}, \\gate, 1, \\amp, 0.3]]);"
        )
        lines.append(
            f"~score.add([{end_t}, [\\n_set, {node_id}, \\gate, 0]]);"
        )

    return lines


def _build_filter_automation(section: dict, start_bar: int) -> list[str]:
    """Generate Score entries for filter cutoff changes."""
    fa = section.get("filter_automation")
    if fa is None:
        return []

    timestamps = fa.get("timestamps", [])
    cutoffs = fa.get("cutoff_values", [])
    if len(timestamps) < 2 or len(cutoffs) < 2:
        return []

    lines: list[str] = []

    # Create a filter sweep synth at section start
    node_id = 7000 + start_bar * 100
    start_t = round(timestamps[0], 6)
    initial_cutoff = round(cutoffs[0], 2)
    resonance = round(fa.get("resonance_estimate", 0.3), 4)
    end_t = round(timestamps[-1], 6)
    dur = max(0.1, round(end_t - start_t, 6))

    lines.append(
        f"~score.add([{start_t}, [\\s_new, \\filterSweep, {node_id}, 0, 200, "
        f"\\cutoff, {initial_cutoff}, \\res, {resonance}, \\dur, {dur}]]);"
    )

    # Schedule n_set messages for cutoff automation at key points
    # Subsample to at most 20 automation points
    step = max(1, len(timestamps) // 20)
    for i in range(0, len(timestamps), step):
        t = round(timestamps[i], 6)
        cutoff = round(cutoffs[i], 2)
        lines.append(
            f"~score.add([{t}, [\\n_set, {node_id}, \\cutoff, {cutoff}]]);"
        )

    return lines


def _build_effects_chain(analysis: dict) -> list[str]:
    """Generate Score entries for effects buses (sidechain, compression, bitcrushing)."""
    effects = analysis.get("effects_estimate", {})
    lines: list[str] = []

    # Sidechain compressor (depth is 0-1, convert from dB)
    sc_depth_db = abs(effects.get("sidechain_depth", 0.0))
    if sc_depth_db > 1.0:
        depth_linear = min(0.9, sc_depth_db / 12.0)
        lines.append(
            f"~score.add([0.0, [\\s_new, \\sidechain, 9000, 0, 200, "
            f"\\depth, {round(depth_linear, 4)}]]);"
        )

    # Bitcrusher
    if effects.get("bitcrushing_detected", False):
        lines.append(
            f"~score.add([0.0, [\\s_new, \\sp1200, 9001, 0, 200, "
            f"\\mix, 0.4]]);"
        )

    # Master compressor
    compression = analysis.get("compression_estimate", {})
    crest = compression.get("crest_factor", 3.0)
    # Lower crest factor means heavier compression was applied
    if crest < 6.0:
        ratio = round(min(8.0, max(1.5, 12.0 / crest)), 2)
        lines.append(
            f"~score.add([0.0, [\\s_new, \\alesis3630, 9002, 0, 200, "
            f"\\ratio, {ratio}]]);"
        )

    lines.append("")
    return lines
