#!/usr/bin/env python3
"""Compatibility entry point for dependency training."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if __name__ == "__main__":
    from scripts.train import main

    raise SystemExit(main(["dependency", *sys.argv[1:]]))
