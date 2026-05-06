#!/usr/bin/env python3
"""Compatibility wrapper for merge_macrogene_rounds.py."""

from pathlib import Path
import runpy

runpy.run_path(str(Path(__file__).with_name("merge_macrogene_rounds.py")), run_name="__main__")
