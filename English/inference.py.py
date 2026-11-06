#!/usr/bin/env python3
"""FastSpeech 2 inference - English (Svarah).

Usage:
    python English/inference.py [--restore_step N | --checkpoint PATH] [--text "..."]

Synthesizes the English assignment test sentences with the trained model and
writes audio samples plus the duration / pitch / energy variance-adaptor plots
to Results/English/.
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "fs2_core"))

import inference as inference_mod  # noqa: E402

CFG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")

if __name__ == "__main__":
    argv = sys.argv[1:]
    if "-c" not in argv and "--config" not in argv:
        argv = ["-c", CFG] + argv
    inference_mod.cli(argv)
