"""Hand-crafted Da Funk transcription.

Generates SuperCollider NRT code for Daft Punk - Da Funk (1997).
Uses musical knowledge of the actual song rather than automatic analysis.

BPM: 113, Key: E minor
Equipment: TB-303 bass through distortion + filter, TR-909 drums
"""

from __future__ import annotations
import math


def _midi_to_freq(midi: int) -> float:
    return round(440.0 * (2.0 ** ((midi - 69) / 12.0)), 4)


def generate_da_funk(duration: float = 330.0, alive: bool = False) -> str:
    """Generate SC code for Da Funk.

    Args:
        duration: total track duration in seconds
        alive: if True, generate the Alive 2007 version with more energy
    """
    bpm = 113.0
    beat = 60.0 / bpm          # 0.53097s
    step = beat / 4.0          # 0.13274s (16th note)
    bar = beat * 4.0           # 2.12389s

    # The Da Funk riff — 2-bar pattern
    # Each entry: (step_offset, midi_note, duration_in_steps)
    # E2=40, G2=43, A2=45, Bb2=46
    riff_pattern = [
        # Bar 1: E E . E . E G . A . G . E . . .
        (0,  40, 1),
        (1,  40, 1),
        (3,  40, 1),
        (5,  40, 1),
        (6,  43, 1),
        (8,  45, 1),
        (10, 43, 1),
        (12, 40, 2),
        # Bar 2: E E . E . E G . A . Bb . A . G .
        (16, 40, 1),
        (17, 40, 1),
        (19, 40, 1),
        (21, 40, 1),
        (22, 43, 1),
        (24, 45, 1),
        (26, 46, 1),
        (28, 45, 1),
        (30, 43, 2),
    ]

    # 909 drum patterns (16-step, 1 bar each)
    kick_pattern =  [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0]  # four-on-the-floor
    hihat_pattern = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]  # 8th notes
    clap_pattern =  [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]  # backbeat (2 & 4)
    snare_pattern = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # none in Da Funk

    # === ARRANGEMENT ===
    # Define sections as (start_bar, end_bar, label, elements)
    # elements: r=riff, k=kick, h=hihat, c=clap
    total_bars = int(math.ceil(duration / bar))

    if not alive:
        # Original Da Funk structure
        sections = [
            (0,   8,  "intro",      "r",    0.08),  # riff alone, filter very closed
            (8,   16, "build1",     "r",    0.20),  # filter opening
            (16,  22, "build2",     "r",    0.40),  # filter more open
            (22,  30, "drums_in",   "rkh",  0.55),  # kick + hihat enter
            (30,  50, "full1",      "rkhc", 0.80),  # full groove
            (50,  70, "full2",      "rkhc", 1.00),  # filter wide open, peak energy
            (70,  80, "breakdown",  "r",    0.30),  # drums drop, filter closes
            (80, 100, "full3",      "rkhc", 0.90),  # back to full
            (100,120, "full4",      "rkhc", 1.00),  # peak
            (120,130, "breakdown2", "r",    0.25),  # another breakdown
            (130,145, "full5",      "rkhc", 0.85),  # back up
            (145,150, "outro_build","rkh",  0.50),  # drums thinning
            (150,155, "outro",      "r",    0.15),  # riff alone, filter closing
        ]
    else:
        # Alive 2007 version — more energy, crowd noise, transitions
        sections = [
            (0,   6,  "intro",      "r",    0.10),
            (6,   12, "build",      "r",    0.30),
            (12,  20, "drums_in",   "rkh",  0.60),
            (20,  50, "full1",      "rkhc", 0.90),
            (50,  70, "peak1",      "rkhc", 1.00),
            (70,  80, "breakdown",  "r",    0.25),
            (80, 100, "full2",      "rkhc", 0.95),
            (100,130, "peak2",      "rkhc", 1.00),
            (130,145, "daftendirekt","rkhc", 0.85),  # transition section
            (145,160, "full3",      "rkhc", 0.90),
            (160,170, "breakdown2", "r",    0.20),
            (170,180, "outro",      "rkh",  0.40),
            (180,187, "fadeout",    "r",    0.10),
        ]

    # Clamp sections to actual duration
    clamped = []
    for s_bar, e_bar, label, elems, filter_open in sections:
        if s_bar >= total_bars:
            break
        e_bar = min(e_bar, total_bars)
        clamped.append((s_bar, e_bar, label, elems, filter_open))
    sections = clamped

    # === GENERATE SC CODE ===
    lines = []
    lines.append(f"// Da Funk {'(Alive 2007)' if alive else '(Original)'}")
    lines.append(f"// BPM: {bpm}, Key: Em, Duration: {duration:.1f}s")
    lines.append("~score = Score.new;")
    lines.append("")

    # Keeper synth
    nid = 99
    lines.append(f"~score.add([0.0, [\\s_new, \\crowdNoise, {nid}, 0, 0, \\amp, 0.0]]);")
    lines.append(f"~score.add([{round(duration - 0.05, 4)}, [\\n_free, {nid}]]);")
    lines.append("")

    node_id = [1000]  # mutable counter

    def next_id():
        node_id[0] += 1
        return node_id[0]

    for s_bar, e_bar, label, elems, filter_open in sections:
        s_time = s_bar * bar
        e_time = min(e_bar * bar, duration)
        n_bars = e_bar - s_bar

        # Filter cutoff for this section — key to the Da Funk sound
        # Low cutoff = muffled/closed, high cutoff = bright/open
        base_cutoff = 150 + filter_open * 3500  # 150 Hz to 3650 Hz
        filter_res = 0.55 + filter_open * 0.15   # higher res when more open

        lines.append(f"// --- {label} (bar {s_bar}-{e_bar}, "
                     f"filter: {base_cutoff:.0f}Hz) ---")

        # RIFF
        if "r" in elems:
            for bar_idx in range(n_bars):
                bar_time = s_time + bar_idx * bar
                if bar_time >= e_time:
                    break

                # Gradual filter sweep within section
                progress = bar_idx / max(1, n_bars - 1)
                # Filter opens gradually within build sections
                if "build" in label:
                    cutoff = base_cutoff * (0.5 + 0.5 * progress)
                elif "outro" in label or "fade" in label:
                    cutoff = base_cutoff * (1.0 - 0.7 * progress)
                else:
                    # Slight movement for "life"
                    cutoff = base_cutoff * (0.9 + 0.2 * math.sin(progress * math.pi))

                for step_off, midi, dur_steps in riff_pattern:
                    t = round(bar_time + step_off * step, 6)
                    if t >= e_time:
                        break
                    end_t = round(t + dur_steps * step * 0.9, 6)  # slight gap between notes
                    if end_t > e_time:
                        end_t = round(e_time, 6)

                    freq = _midi_to_freq(midi)
                    nid = next_id()

                    # TB-303 params with per-note filter cutoff
                    fc = round(cutoff, 1)
                    drive = 4.0 if filter_open > 0.5 else 3.0
                    accent = 0.7 if filter_open > 0.3 else 0.4

                    lines.append(
                        f"~score.add([{t}, [\\s_new, \\bassline, {nid}, 0, 0, "
                        f"\\freq, {freq}, \\gate, 1, \\amp, 0.45, "
                        f"\\filterCutoff, {fc}, \\filterRes, {round(filter_res, 2)}, "
                        f"\\filterEnv, {round(cutoff * 1.5, 1)}, "
                        f"\\accent, {accent}, \\drive, {drive}, "
                        f"\\decay, 0.15, \\sustain, 0.6, \\release, 0.3]]);"
                    )
                    lines.append(
                        f"~score.add([{end_t}, [\\n_set, {nid}, \\gate, 0]]);"
                    )

        # DRUMS
        if any(d in elems for d in "khc"):
            drum_amp_scale = min(1.0, filter_open * 1.3 + 0.2)

            for bar_idx in range(n_bars):
                bar_time = s_time + bar_idx * bar
                if bar_time >= e_time:
                    break

                # Kick
                if "k" in elems:
                    for s, hit in enumerate(kick_pattern):
                        if hit:
                            t = round(bar_time + s * step, 6)
                            if t >= e_time:
                                break
                            nid = next_id()
                            lines.append(
                                f"~score.add([{t}, [\\s_new, \\kick, {nid}, 0, 0, "
                                f"\\amp, {round(0.85 * drum_amp_scale, 3)}, "
                                f"\\freq, 52, \\decay, 0.55, \\punch, 1.0]]);"
                            )

                # Hihat
                if "h" in elems:
                    for s, hit in enumerate(hihat_pattern):
                        if hit:
                            t = round(bar_time + s * step, 6)
                            if t >= e_time:
                                break
                            nid = next_id()
                            lines.append(
                                f"~score.add([{t}, [\\s_new, \\hihat, {nid}, 0, 0, "
                                f"\\amp, {round(0.18 * drum_amp_scale, 3)}, "
                                f"\\decay, 0.04]]);"
                            )

                # Clap (backbeat)
                if "c" in elems:
                    for s, hit in enumerate(clap_pattern):
                        if hit:
                            t = round(bar_time + s * step, 6)
                            if t >= e_time:
                                break
                            nid = next_id()
                            lines.append(
                                f"~score.add([{t}, [\\s_new, \\clap, {nid}, 0, 0, "
                                f"\\amp, {round(0.35 * drum_amp_scale, 3)}, "
                                f"\\decay, 0.18]]);"
                            )

        lines.append("")

    lines.append(f"~score.add([{round(duration, 4)}, [\\c_set, 0, 0]]);")
    return "\n".join(lines)


if __name__ == "__main__":
    # Generate both versions
    from pathlib import Path
    from copywrite.config import load_config
    from copywrite.engine import NRTRenderer
    import soundfile as sf

    config = load_config()
    renderer = NRTRenderer(config)

    for name, alive, dur in [
        ("da_funk_v4", False, 330.0),
        ("da_funk_daftendirekt_alive_v4", True, 397.0),
    ]:
        print(f"\n=== Generating {name} ===")
        code = generate_da_funk(duration=dur, alive=alive)
        scd = Path(f"data/transcriptions/{name}.scd")
        wav = Path(f"data/transcriptions/{name}.wav")
        with open(scd, "w") as f:
            f.write(code)
        print(f"  Events: {code.count('s_new')}")
        out = renderer.render(code, wav, duration=dur)
        data, sr = sf.read(str(out))
        print(f"  Rendered: {len(data)/sr:.1f}s, Peak: {abs(data).max():.4f}")
