#!/usr/bin/env python3

"""Repo-root entrypoint for the sample-aware TTA pipeline."""

from pathlib import Path
import os
import runpy
import sys


REPO_ROOT = Path(__file__).resolve().parent
MODEL_DIR = REPO_ROOT / "model"

if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

runpy.run_path(str(MODEL_DIR / "run_tta.py"), run_name="__main__")
