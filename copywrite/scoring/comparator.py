"""Compare two AudioFeatures objects and produce a similarity score."""

from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

from .features import AudioFeatures, _RELATIVE_KEYS


@dataclass
class TranscriptionScore:
    """Detailed comparison score between reference and rendered audio."""
    overall: float
    rhythm_score: float
    harmony_score: float
    spectral_score: float
    structure_score: float
    dynamics_score: float
    diagnostics: dict

    def summary(self) -> str:
        """Human-readable summary of the score."""
        lines = [
            f"Overall score: {self.overall:.3f}",
            f"  Rhythm:    {self.rhythm_score:.3f}",
            f"  Harmony:   {self.harmony_score:.3f}",
            f"  Spectral:  {self.spectral_score:.3f}",
            f"  Structure: {self.structure_score:.3f}",
            f"  Dynamics:  {self.dynamics_score:.3f}",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _cosine_similarity(a: list[float] | np.ndarray,
                       b: list[float] | np.ndarray) -> float:
    a_arr = np.asarray(a, dtype=float)
    b_arr = np.asarray(b, dtype=float)
    norm_a = np.linalg.norm(a_arr)
    norm_b = np.linalg.norm(b_arr)
    if norm_a < 1e-10 or norm_b < 1e-10:
        return 0.0
    return float(np.dot(a_arr, b_arr) / (norm_a * norm_b))


def _ratio_similarity(a: float, b: float) -> float:
    """Returns a value between 0 and 1 where 1 means identical."""
    if a <= 0 and b <= 0:
        return 1.0
    if a <= 0 or b <= 0:
        return 0.0
    ratio = min(a, b) / max(a, b)
    return float(ratio)


def _sequence_edit_distance(seq_a: list[str], seq_b: list[str]) -> float:
    """Normalised Levenshtein distance between two string sequences.
    Returns similarity: 1.0 = identical, 0.0 = completely different.
    """
    n, m = len(seq_a), len(seq_b)
    if n == 0 and m == 0:
        return 1.0
    if n == 0 or m == 0:
        return 0.0

    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if seq_a[i - 1] == seq_b[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1,
                           dp[i][j - 1] + 1,
                           dp[i - 1][j - 1] + cost)

    max_len = max(n, m)
    return 1.0 - dp[n][m] / max_len


def _contour_correlation(a: list[float], b: list[float]) -> float:
    """Pearson correlation between two contours, resampled to same length."""
    if not a or not b:
        return 0.0
    a_arr = np.asarray(a, dtype=float)
    b_arr = np.asarray(b, dtype=float)
    # Resample to common length
    target_len = min(len(a_arr), len(b_arr), 64)
    if target_len < 2:
        return _ratio_similarity(float(a_arr.mean()), float(b_arr.mean()))
    a_res = np.interp(np.linspace(0, 1, target_len),
                      np.linspace(0, 1, len(a_arr)), a_arr)
    b_res = np.interp(np.linspace(0, 1, target_len),
                      np.linspace(0, 1, len(b_arr)), b_arr)
    if a_res.std() < 1e-10 or b_res.std() < 1e-10:
        return _ratio_similarity(float(a_res.mean()), float(b_res.mean()))
    corr = float(np.corrcoef(a_res, b_res)[0, 1])
    # Map from [-1, 1] to [0, 1]
    return (corr + 1.0) / 2.0


# ---------------------------------------------------------------------------
# Sub-scorers
# ---------------------------------------------------------------------------

def _score_rhythm(ref: AudioFeatures, ren: AudioFeatures) -> tuple[float, dict]:
    diag: dict = {}

    # Tempo ratio
    tempo_ratio = _ratio_similarity(ref.rhythm.tempo, ren.rhythm.tempo)
    diag["tempo_ref"] = ref.rhythm.tempo
    diag["tempo_ren"] = ren.rhythm.tempo
    diag["tempo_ratio"] = tempo_ratio

    # Pattern cosine similarity
    kick_sim = _cosine_similarity(ref.rhythm.kick_pattern, ren.rhythm.kick_pattern)
    snare_sim = _cosine_similarity(ref.rhythm.snare_pattern, ren.rhythm.snare_pattern)
    hihat_sim = _cosine_similarity(ref.rhythm.hihat_pattern, ren.rhythm.hihat_pattern)
    pattern_sim = (kick_sim + snare_sim + hihat_sim) / 3.0
    diag["kick_similarity"] = kick_sim
    diag["snare_similarity"] = snare_sim
    diag["hihat_similarity"] = hihat_sim
    diag["pattern_similarity"] = pattern_sim

    # Onset density ratio
    density_ratio = _ratio_similarity(ref.rhythm.onset_density, ren.rhythm.onset_density)
    diag["onset_density_ratio"] = density_ratio

    score = 0.4 * tempo_ratio + 0.4 * pattern_sim + 0.2 * density_ratio
    return float(score), diag


def _score_harmony(ref: AudioFeatures, ren: AudioFeatures) -> tuple[float, dict]:
    diag: dict = {}

    # Key match
    ref_key = ref.harmony.key
    ren_key = ren.harmony.key
    if ref_key == ren_key:
        key_score = 1.0
    elif _RELATIVE_KEYS.get(ref_key) == ren_key:
        key_score = 0.5
    else:
        key_score = 0.0
    diag["key_ref"] = ref_key
    diag["key_ren"] = ren_key
    diag["key_score"] = key_score

    # Chroma cosine similarity
    chroma_sim = _cosine_similarity(ref.harmony.chroma_mean, ren.harmony.chroma_mean)
    diag["chroma_similarity"] = chroma_sim

    # Chord sequence edit distance
    ref_chords = [c["chord"] for c in ref.harmony.chord_sequence]
    ren_chords = [c["chord"] for c in ren.harmony.chord_sequence]
    chord_sim = _sequence_edit_distance(ref_chords, ren_chords)
    diag["chord_sequence_similarity"] = chord_sim

    score = 0.3 * key_score + 0.35 * chroma_sim + 0.35 * chord_sim
    return float(score), diag


def _score_spectral(ref: AudioFeatures, ren: AudioFeatures) -> tuple[float, dict]:
    diag: dict = {}

    # MFCC cosine similarity
    mfcc_sim = _cosine_similarity(ref.spectral.mfcc_mean, ren.spectral.mfcc_mean)
    diag["mfcc_similarity"] = mfcc_sim

    # Centroid ratio
    centroid_ratio = _ratio_similarity(ref.spectral.spectral_centroid_mean,
                                       ren.spectral.spectral_centroid_mean)
    diag["centroid_ratio"] = centroid_ratio

    # Flatness difference
    flat_diff = abs(ref.spectral.spectral_flatness_mean - ren.spectral.spectral_flatness_mean)
    flatness_score = max(0.0, 1.0 - flat_diff * 10.0)  # 0.1 diff -> 0.0
    diag["flatness_diff"] = flat_diff
    diag["flatness_score"] = flatness_score

    # Bandwidth ratio
    bw_ratio = _ratio_similarity(ref.spectral.spectral_bandwidth_mean,
                                  ren.spectral.spectral_bandwidth_mean)
    diag["bandwidth_ratio"] = bw_ratio

    score = 0.4 * mfcc_sim + 0.25 * centroid_ratio + 0.15 * flatness_score + 0.2 * bw_ratio
    return float(score), diag


def _score_structure(ref: AudioFeatures, ren: AudioFeatures) -> tuple[float, dict]:
    diag: dict = {}

    # Number of sections match
    n_ref = len(ref.structure.section_boundaries)
    n_ren = len(ren.structure.section_boundaries)
    section_count_score = _ratio_similarity(float(n_ref), float(n_ren))
    diag["sections_ref"] = n_ref
    diag["sections_ren"] = n_ren
    diag["section_count_score"] = section_count_score

    # Section duration alignment (DTW-like via contour correlation)
    duration_ratio = _ratio_similarity(ref.structure.duration, ren.structure.duration)
    diag["duration_ratio"] = duration_ratio

    # Energy contour correlation
    energy_corr = _contour_correlation(ref.structure.energy_contour,
                                        ren.structure.energy_contour)
    diag["energy_contour_correlation"] = energy_corr

    # Section label alignment
    label_sim = _sequence_edit_distance(ref.structure.section_labels,
                                         ren.structure.section_labels)
    diag["label_similarity"] = label_sim

    score = 0.2 * section_count_score + 0.2 * duration_ratio + 0.3 * energy_corr + 0.3 * label_sim
    return float(score), diag


def _score_dynamics(ref: AudioFeatures, ren: AudioFeatures) -> tuple[float, dict]:
    diag: dict = {}

    # RMS contour correlation
    rms_corr = _contour_correlation(ref.dynamics.rms_contour, ren.dynamics.rms_contour)
    diag["rms_contour_correlation"] = rms_corr

    # Crest factor ratio
    crest_ratio = _ratio_similarity(ref.dynamics.crest_factor, ren.dynamics.crest_factor)
    diag["crest_factor_ratio"] = crest_ratio

    # Dynamic range similarity
    dr_diff = abs(ref.dynamics.dynamic_range - ren.dynamics.dynamic_range)
    dr_score = max(0.0, 1.0 - dr_diff / 30.0)  # 30 dB diff -> 0.0
    diag["dynamic_range_diff"] = dr_diff
    diag["dynamic_range_score"] = dr_score

    # Sidechain depth difference
    sc_diff = abs(ref.dynamics.sidechain_depth - ren.dynamics.sidechain_depth)
    sc_score = max(0.0, 1.0 - sc_diff / 12.0)  # 12 dB diff -> 0.0
    diag["sidechain_depth_diff"] = sc_diff
    diag["sidechain_score"] = sc_score

    score = 0.35 * rms_corr + 0.25 * crest_ratio + 0.2 * dr_score + 0.2 * sc_score
    return float(score), diag


# ---------------------------------------------------------------------------
# Main comparison entry point
# ---------------------------------------------------------------------------

_DEFAULT_WEIGHTS = {
    "rhythm": 0.25,
    "harmony": 0.20,
    "spectral": 0.25,
    "structure": 0.15,
    "dynamics": 0.15,
}


def compare_features(reference: AudioFeatures, rendered: AudioFeatures,
                     weights: dict | None = None) -> TranscriptionScore:
    """Compare two feature sets and return a detailed score.

    weights: dict with keys rhythm, harmony, spectral, structure, dynamics
             (values should sum to 1.0). Falls back to default weights.
    """
    w = weights if weights is not None else _DEFAULT_WEIGHTS

    rhythm_score, rhythm_diag = _score_rhythm(reference, rendered)
    harmony_score, harmony_diag = _score_harmony(reference, rendered)
    spectral_score, spectral_diag = _score_spectral(reference, rendered)
    structure_score, structure_diag = _score_structure(reference, rendered)
    dynamics_score, dynamics_diag = _score_dynamics(reference, rendered)

    overall = (
        w.get("rhythm", 0.25) * rhythm_score
        + w.get("harmony", 0.20) * harmony_score
        + w.get("spectral", 0.25) * spectral_score
        + w.get("structure", 0.15) * structure_score
        + w.get("dynamics", 0.15) * dynamics_score
    )

    diagnostics = {
        "rhythm": rhythm_diag,
        "harmony": harmony_diag,
        "spectral": spectral_diag,
        "structure": structure_diag,
        "dynamics": dynamics_diag,
        "weights": dict(w),
    }

    return TranscriptionScore(
        overall=float(overall),
        rhythm_score=float(rhythm_score),
        harmony_score=float(harmony_score),
        spectral_score=float(spectral_score),
        structure_score=float(structure_score),
        dynamics_score=float(dynamics_score),
        diagnostics=diagnostics,
    )
