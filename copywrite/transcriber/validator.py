"""Transcription validation loop — render, compare, revise."""

from __future__ import annotations

import re
from pathlib import Path

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

from copywrite.config import CopywriteConfig, load_config
from copywrite.engine import NRTRenderer, SynthDefManager
from copywrite.scoring import extract_features, compare_features, AudioFeatures, TranscriptionScore
from .analyzer import analyze_track, TrackAnalysis
from .codegen import generate_sc_code

console = Console()


def transcription_loop(
    audio_path: Path,
    config: CopywriteConfig,
    max_iterations: int | None = None,
    score_threshold: float | None = None,
) -> dict:
    """Run the full transcription pipeline for one track.

    1. Analyze the reference track
    2. Generate initial SC code
    3. Loop:
       a. Render SC code to wav
       b. Extract features from rendered wav
       c. Compare to reference features
       d. If score >= threshold, stop
       e. Otherwise, generate diagnostic feedback and revise code
    4. Save best result

    Returns dict with:
        - analysis: TrackAnalysis dict
        - best_code: str (SC code)
        - best_score: TranscriptionScore
        - iterations: int
        - output_wav: Path
    """
    audio_path = Path(audio_path)
    max_iter = max_iterations or config.transcribe_max_iterations
    threshold = score_threshold or config.transcribe_score_threshold

    config.ensure_dirs()
    track_name = audio_path.stem
    track_dir = config.transcriptions_dir / track_name
    track_dir.mkdir(parents=True, exist_ok=True)

    renderer = NRTRenderer(config)
    synthdef_mgr = SynthDefManager(config)
    synthdef_docs = synthdef_mgr.get_synthdef_docs()

    # Step 1: Analyze
    console.print(f"[cyan]Analyzing reference track: {audio_path.name}[/cyan]")
    analysis = analyze_track(audio_path, sr=config.sc_sample_rate)
    analysis_dict = analysis.to_dict()
    analysis.save(track_dir / "analysis.json")
    console.print(f"[green]Analysis complete:[/green] {analysis.duration:.1f}s, "
                  f"{analysis.bpm:.0f} BPM, {analysis.key}, "
                  f"{len(analysis.sections)} sections")

    # Step 2: Extract reference features
    ref_features = extract_features(audio_path, sr=config.sc_sample_rate)

    # Step 3: Generate initial SC code
    console.print("[cyan]Generating initial SuperCollider code...[/cyan]")
    sc_code = generate_sc_code(analysis_dict, synthdef_docs, track_title=track_name)
    _save_code(track_dir / "iteration_0.scd", sc_code)

    best_code = sc_code
    best_score: TranscriptionScore | None = None
    best_iteration = 0
    best_wav: Path | None = None

    # Step 4: Iteration loop
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        console=console,
    ) as progress:
        task = progress.add_task("Transcription loop", total=max_iter)

        for iteration in range(1, max_iter + 1):
            progress.update(task, description=f"Iteration {iteration}/{max_iter}")

            # 4a. Render
            output_wav = track_dir / f"render_{iteration}.wav"
            try:
                renderer.render(sc_code, output_wav, duration=analysis.duration)
            except RuntimeError as exc:
                console.print(f"[red]Render failed (iteration {iteration}): {exc}[/red]")
                progress.advance(task)
                continue

            # 4b. Extract rendered features
            try:
                rendered_features = extract_features(output_wav, sr=config.sc_sample_rate)
            except Exception as exc:
                console.print(f"[red]Feature extraction failed (iteration {iteration}): {exc}[/red]")
                progress.advance(task)
                continue

            # 4c. Compare
            score = compare_features(ref_features, rendered_features)
            console.print(f"  [dim]Iteration {iteration}:[/dim] "
                          f"score={score.overall:.3f} "
                          f"(R={score.rhythm_score:.2f} H={score.harmony_score:.2f} "
                          f"S={score.spectral_score:.2f} D={score.dynamics_score:.2f} "
                          f"St={score.structure_score:.2f})")

            # Track best
            if best_score is None or score.overall > best_score.overall:
                best_score = score
                best_code = sc_code
                best_iteration = iteration
                best_wav = output_wav

            # 4d. Check threshold
            if score.overall >= threshold:
                console.print(f"[green]Threshold reached at iteration {iteration}! "
                              f"Score: {score.overall:.3f} >= {threshold}[/green]")
                progress.advance(task)
                break

            # 4e. Revise
            feedback = _generate_revision_feedback(score, ref_features, rendered_features)
            sc_code = _revise_sc_code(sc_code, feedback, analysis_dict, synthdef_docs)
            _save_code(track_dir / f"iteration_{iteration}.scd", sc_code)

            progress.advance(task)

    # Step 5: Save best result
    if best_score is None:
        console.print("[red]No successful renders were produced.[/red]")
        return {
            "analysis": analysis_dict,
            "best_code": best_code,
            "best_score": None,
            "iterations": max_iter,
            "output_wav": None,
        }

    _save_code(track_dir / "best.scd", best_code)
    console.print(f"\n[green]Best result: iteration {best_iteration}, "
                  f"score {best_score.overall:.3f}[/green]")
    console.print(best_score.summary())

    return {
        "analysis": analysis_dict,
        "best_code": best_code,
        "best_score": best_score,
        "iterations": best_iteration,
        "output_wav": best_wav,
    }


def _save_code(path: Path, code: str) -> None:
    """Write SC code to a file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write(code)


def _generate_revision_feedback(
    score: TranscriptionScore,
    reference: AudioFeatures,
    rendered: AudioFeatures,
) -> str:
    """Generate specific feedback about what to fix in the SC code.

    Looks at which dimensions scored lowest and generates targeted
    instructions for revision.
    """
    feedback_parts: list[str] = []

    # Rhythm feedback
    if score.rhythm_score < 0.5:
        ref_tempo = reference.rhythm.tempo
        ren_tempo = rendered.rhythm.tempo
        if abs(ref_tempo - ren_tempo) > 2.0:
            feedback_parts.append(
                f"TEMPO: Reference is {ref_tempo:.1f} BPM but rendered is "
                f"{ren_tempo:.1f} BPM. Adjust beat timing by factor "
                f"{ref_tempo / (ren_tempo + 1e-10):.4f}."
            )
        ref_density = reference.rhythm.onset_density
        ren_density = rendered.rhythm.onset_density
        if abs(ref_density - ren_density) > ref_density * 0.3:
            if ren_density < ref_density:
                feedback_parts.append(
                    "DENSITY: Rendered has fewer onsets than reference. "
                    "Add more drum hits or shorter note durations."
                )
            else:
                feedback_parts.append(
                    "DENSITY: Rendered has too many onsets. "
                    "Remove some drum hits or increase note spacing."
                )

    # Harmony feedback
    if score.harmony_score < 0.5:
        ref_key = reference.harmony.key
        ren_key = rendered.harmony.key
        if ref_key != ren_key:
            feedback_parts.append(
                f"KEY: Reference is in {ref_key} but rendered sounds like {ren_key}. "
                f"Transpose all pitched elements."
            )
        ref_chroma = reference.harmony.chroma_mean
        ren_chroma = rendered.harmony.chroma_mean
        if ref_chroma is not None and ren_chroma is not None:
            chroma_diff = [abs(a - b) for a, b in zip(ref_chroma, ren_chroma)]
            worst_note = int(max(range(len(chroma_diff)), key=lambda i: chroma_diff[i]))
            note_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
            if worst_note < len(note_names):
                feedback_parts.append(
                    f"CHROMA: Largest harmonic mismatch near {note_names[worst_note]}. "
                    f"Check pitch accuracy of bass and chord voicings."
                )

    # Spectral feedback
    if score.spectral_score < 0.5:
        ref_centroid = reference.spectral.spectral_centroid_mean
        ren_centroid = rendered.spectral.spectral_centroid_mean
        if ref_centroid > 0 and ren_centroid > 0:
            ratio = ren_centroid / ref_centroid
            if ratio > 1.3:
                feedback_parts.append(
                    f"BRIGHTNESS: Rendered is too bright (centroid ratio {ratio:.2f}). "
                    f"Lower filter cutoffs or reduce high-frequency content."
                )
            elif ratio < 0.7:
                feedback_parts.append(
                    f"BRIGHTNESS: Rendered is too dark (centroid ratio {ratio:.2f}). "
                    f"Raise filter cutoffs or add high-frequency content."
                )

        ref_flatness = reference.spectral.spectral_flatness_mean
        ren_flatness = rendered.spectral.spectral_flatness_mean
        if abs(ref_flatness - ren_flatness) > 0.1:
            if ren_flatness > ref_flatness:
                feedback_parts.append(
                    "TEXTURE: Rendered is too noisy/flat. Use more tonal synths, "
                    "reduce noise components."
                )
            else:
                feedback_parts.append(
                    "TEXTURE: Rendered is too tonal/peaked. Add noise or "
                    "widen the spectral content."
                )

    # Dynamics feedback
    if score.dynamics_score < 0.5:
        ref_rms = reference.dynamics.rms_mean
        ren_rms = rendered.dynamics.rms_mean
        if ref_rms > 0:
            level_ratio = ren_rms / (ref_rms + 1e-10)
            if level_ratio > 1.5:
                feedback_parts.append(
                    f"LEVEL: Rendered is too loud (ratio {level_ratio:.2f}). "
                    f"Reduce amp parameters."
                )
            elif level_ratio < 0.5:
                feedback_parts.append(
                    f"LEVEL: Rendered is too quiet (ratio {level_ratio:.2f}). "
                    f"Increase amp parameters."
                )

        ref_crest = reference.dynamics.crest_factor
        ren_crest = rendered.dynamics.crest_factor
        if abs(ref_crest - ren_crest) > ref_crest * 0.3:
            if ren_crest > ref_crest:
                feedback_parts.append(
                    "COMPRESSION: Rendered has more transient peaks. "
                    "Increase compressor ratio or lower threshold."
                )
            else:
                feedback_parts.append(
                    "COMPRESSION: Rendered is over-compressed. "
                    "Reduce compressor ratio or raise threshold."
                )

    # Structure feedback
    if score.structure_score < 0.5:
        ref_dur = reference.structure.duration
        ren_dur = rendered.structure.duration
        if abs(ref_dur - ren_dur) > 2.0:
            feedback_parts.append(
                f"DURATION: Reference is {ref_dur:.1f}s but rendered is {ren_dur:.1f}s. "
                f"Adjust the total score duration."
            )

    if not feedback_parts:
        feedback_parts.append(
            "No specific issues identified. Try adjusting overall mix balance "
            "and effects parameters."
        )

    return "\n".join(feedback_parts)


def _revise_sc_code(
    current_code: str,
    feedback: str,
    analysis_dict: dict,
    synthdef_docs: str,
) -> str:
    """Revise SC code based on comparison feedback.

    Applies rule-based revisions:
    - If tempo is wrong, adjust beat timing
    - If key is wrong, transpose
    - If spectral profile is off, adjust filter settings
    - If dynamics are off, adjust compression/levels
    """
    revised = current_code

    # Parse feedback for specific issues
    if "TEMPO:" in feedback:
        match = re.search(r"factor ([\d.]+)", feedback)
        if match:
            factor = float(match.group(1))
            revised = _adjust_tempo(revised, factor)

    if "KEY:" in feedback:
        match = re.search(r"Reference is in (.+?) but rendered sounds like (.+?)\.", feedback)
        if match:
            ref_key = match.group(1).strip()
            ren_key = match.group(2).strip()
            semitones = _key_distance(ren_key, ref_key)
            if semitones != 0:
                revised = _transpose_code(revised, semitones)

    if "BRIGHTNESS:" in feedback:
        if "too bright" in feedback:
            revised = _adjust_filter_cutoffs(revised, factor=0.75)
        elif "too dark" in feedback:
            revised = _adjust_filter_cutoffs(revised, factor=1.4)

    if "LEVEL:" in feedback:
        if "too loud" in feedback:
            revised = _adjust_amplitudes(revised, factor=0.7)
        elif "too quiet" in feedback:
            revised = _adjust_amplitudes(revised, factor=1.4)

    if "COMPRESSION:" in feedback:
        if "over-compressed" in feedback:
            revised = _adjust_compressor(revised, ratio_delta=-1.0)
        elif "more transient peaks" in feedback:
            revised = _adjust_compressor(revised, ratio_delta=1.0)

    if "DENSITY:" in feedback:
        if "fewer onsets" in feedback:
            # Regenerate with analysis data to add more events
            revised = generate_sc_code(analysis_dict, synthdef_docs)
        # If too many onsets, keep current code — removal is harder to do reliably

    # If no changes were made, regenerate entirely as a fresh attempt
    if revised == current_code:
        revised = generate_sc_code(analysis_dict, synthdef_docs)

    return revised


def _adjust_tempo(code: str, factor: float) -> str:
    """Scale all time values in score entries by a factor."""
    def _scale_time(m: re.Match) -> str:
        t = float(m.group(1))
        new_t = round(t * factor, 6)
        return f"~score.add([{new_t},"

    return re.sub(r"~score\.add\(\[(\d+\.?\d*),", _scale_time, code)


def _key_distance(from_key: str, to_key: str) -> int:
    """Compute semitone distance between two keys."""
    key_map = {
        "C": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3,
        "E": 4, "F": 5, "F#": 6, "Gb": 6, "G": 7, "G#": 8,
        "Ab": 8, "A": 9, "A#": 10, "Bb": 10, "B": 11,
    }
    from_root = from_key.split()[0] if " " in from_key else from_key.rstrip("majmindimaug")
    to_root = to_key.split()[0] if " " in to_key else to_key.rstrip("majmindimaug")
    from_val = key_map.get(from_root, 0)
    to_val = key_map.get(to_root, 0)
    diff = (to_val - from_val) % 12
    if diff > 6:
        diff -= 12
    return diff


def _transpose_code(code: str, semitones: int) -> str:
    """Transpose all \\freq values by a number of semitones."""
    factor = 2.0 ** (semitones / 12.0)

    def _scale_freq(m: re.Match) -> str:
        freq = float(m.group(1))
        new_freq = round(freq * factor, 4)
        return f"\\freq, {new_freq}"

    return re.sub(r"\\freq,\s*([\d.]+)", _scale_freq, code)


def _adjust_filter_cutoffs(code: str, factor: float) -> str:
    """Scale all \\cutoff values by a factor."""
    def _scale_cutoff(m: re.Match) -> str:
        val = float(m.group(1))
        new_val = round(val * factor, 2)
        return f"\\cutoff, {new_val}"

    return re.sub(r"\\cutoff,\s*([\d.]+)", _scale_cutoff, code)


def _adjust_amplitudes(code: str, factor: float) -> str:
    """Scale all \\amp values by a factor."""
    def _scale_amp(m: re.Match) -> str:
        val = float(m.group(1))
        new_val = round(min(1.0, val * factor), 4)
        return f"\\amp, {new_val}"

    return re.sub(r"\\amp,\s*([\d.]+)", _scale_amp, code)


def _adjust_compressor(code: str, ratio_delta: float) -> str:
    """Adjust \\ratio values on the compressor by a delta."""
    def _adjust_ratio(m: re.Match) -> str:
        val = float(m.group(1))
        new_val = round(max(1.0, min(20.0, val + ratio_delta)), 2)
        return f"\\ratio, {new_val}"

    return re.sub(r"\\ratio,\s*([\d.]+)", _adjust_ratio, code)
