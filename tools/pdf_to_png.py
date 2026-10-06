#!/usr/bin/env python
"""Rasterize figure PDFs to PNGs next to the source files.

Run (any env with PyMuPDF):
    python tools/pdf_to_png.py figures/figure3/outputs/*.pdf
    python tools/pdf_to_png.py --dpi 150 figures/figure4/outputs
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import fitz  # PyMuPDF


def convert(pdf: Path, dpi: int) -> Path | None:
    try:
        doc = fitz.open(pdf)
        page = doc.load_page(0)
        pix = page.get_pixmap(dpi=dpi)
        out = pdf.with_suffix(".png")
        pix.save(out)
        doc.close()
        return out
    except Exception as e:  # noqa: BLE001
        print(f"  FAILED {pdf.name}: {e}", file=sys.stderr)
        return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+", help="PDF files and/or directories")
    ap.add_argument("--dpi", type=int, default=140)
    args = ap.parse_args()

    pdfs: list[Path] = []
    for p in map(Path, args.paths):
        if p.is_dir():
            pdfs.extend(sorted(p.glob("*.pdf")))
        elif p.suffix.lower() == ".pdf":
            pdfs.append(p)
    n = 0
    for pdf in pdfs:
        if convert(pdf, args.dpi):
            n += 1
    print(f"rasterized {n}/{len(pdfs)} PDF(s) at {args.dpi} dpi")


if __name__ == "__main__":
    main()
