#!/usr/bin/env python3
"""FastSpeech 2 training - Telugu (IndicTTS Telugu).

Usage:
    python Telugu/train.py [--restore_step N] [--total_step N]

Trains the Telugu model using Telugu/config.yaml.  Checkpoints are saved to
Telugu/checkpoints/ and loss curves to Results/Telugu/.
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "fs2_core"))

import train as train_mod  # noqa: E402

CFG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")

if __name__ == "__main__":
    argv = sys.argv[1:]
    if "-c" not in argv and "--config" not in argv:
        argv = ["-c", CFG] + argv
    train_mod.cli(argv)
