#!/usr/bin/env python
"""Run a figure script with editable-text font settings and save an SVG next to every PDF it writes.

Run:
    python tools/render_with_svg.py <script.py> [script args...]
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["svg.fonttype"] = "none"

from matplotlib.figure import Figure  # noqa: E402

_orig_savefig = Figure.savefig


def _savefig(self, fname, *args, **kwargs):
    result = _orig_savefig(self, fname, *args, **kwargs)
    try:
        s = str(fname)
        if s.lower().endswith(".pdf"):
            kw = dict(kwargs); kw.pop("format", None)
            _orig_savefig(self, s[:-4] + ".svg", *args, **kw)
            print(f"  [+svg] {Path(s).with_suffix('.svg').name}", flush=True)
    except Exception as e:  # never let SVG export break the run
        print(f"  [svg-export-failed] {e}", flush=True)
    return result


Figure.savefig = _savefig

if len(sys.argv) < 2:
    raise SystemExit("usage: render_with_svg.py <script.py> [args...]")

target = sys.argv[1]
sys.argv = sys.argv[1:]                       # hand the rest to the target script
sys.path.insert(0, str(Path(target).resolve().parent))  # for sibling imports
runpy.run_path(target, run_name="__main__")
