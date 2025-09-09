"""Test configuration for fks_transformer.
Adds local src and shared package path for tests.
"""
from __future__ import annotations
import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parent
SRC = ROOT / "src"
SHARED = ROOT / "shared" / "shared_python" / "src"
for p in (SRC, SHARED):
    if p.is_dir():
        sp = str(p)
        if sp not in sys.path:
            sys.path.insert(0, sp)
