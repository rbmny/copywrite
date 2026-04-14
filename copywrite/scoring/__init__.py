"""Scoring module for copywrite: feature extraction, comparison, and CLAP scoring."""

from .features import AudioFeatures, extract_features
from .comparator import compare_features, TranscriptionScore
