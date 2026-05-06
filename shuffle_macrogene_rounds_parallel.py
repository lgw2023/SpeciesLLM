#!/usr/bin/env python3
"""Compatibility wrapper for shuffle_flatten_macrogene.py."""

from pathlib import Path
import runpy

runpy.run_path(str(Path(__file__).with_name("shuffle_flatten_macrogene.py")), run_name="__main__")
