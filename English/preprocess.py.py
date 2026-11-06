#!/usr/bin/env python3
"""FastSpeech 2 preprocessing - English (Svarah).

Usage:
    python English/preprocess.py [--stage all|download|raw|align|features] [--limit N]

Stages:
    download - download Svarah from HuggingFace and subsample to ~5h train /
               ~1h val (writes corpus/ with manifest.csv)
    raw      - resample/normalise audio and clean transcripts (raw/)
    align    - CTC forced alignment -> phone durations (alignments/)
    features - mel / pitch / energy / duration features (preprocessed/)
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "fs2_core"))

from pipeline import run_preprocess  # noqa: E402

CFG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")

if __name__ == "__main__":
    run_preprocess(default_config=CFG)
