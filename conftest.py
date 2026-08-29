"""
Pytest configuration.

Makes the project modules importable no matter which directory pytest is
invoked from (without this, collection only works by accident of the repo root
being the current working directory), and forces Qt onto the offscreen
platform plugin so the UI tests never need a display.
"""
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
