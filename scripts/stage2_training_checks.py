#!/usr/bin/env python3
"""Compatibility wrapper for scripts/pretrain_checks.py."""

from pathlib import Path
import runpy

runpy.run_path(str(Path(__file__).with_name("pretrain_checks.py")), run_name="__main__")
