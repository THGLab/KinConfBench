#!/usr/bin/env python3
"""
Combine four KinCoRe–geometric correlation panel PNGs into one 2×2 figure.

Layout (left-to-right, top-to-bottom):
  protein RMSD | ligand RMSD
  lDDT-CA      | lDDT-PLI

Panel positions preserve each image’s aspect ratio; the figure width is the
maximum of the two row widths.

Usage:
    python combine_kincore_geometric_correlation_panels.py \\
        --figures-dir run_output/figures_successful
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    repo = Path(__file__).resolve().parents[3]
    default_dir = repo / "run_output" / "figures_successful"
    p.add_argument(
        "--figures-dir",
        type=Path,
        default=default_dir,
        help="Directory containing the four kincore_geometric_correlation_*.png panels",
    )
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output path (.png / .pdf). Default: <figures-dir>/kincore_geometric_correlation_four_panel.png",
    )
    p.add_argument("--dpi", type=int, default=300, help="Raster output DPI (PNG only)")
    args = p.parse_args()

    fig_dir: Path = args.figures_dir.resolve()
    names = (
        "kincore_geometric_correlation_protein_rmsd.png",
        "kincore_geometric_correlation_ligand_rmsd.png",
        "kincore_geometric_correlation_lddt_ca.png",
        "kincore_geometric_correlation_lddt_pli.png",
    )
    paths = [fig_dir / n for n in names]
    for path in paths:
        if not path.is_file():
            raise SystemExit(f"Missing panel: {path}")

    images = [mpimg.imread(str(path)) for path in paths]
    hs = [im.shape[0] for im in images]
    ws = [im.shape[1] for im in images]

    h0 = max(hs[0], hs[1])
    h1 = max(hs[2], hs[3])
    w0 = sum(ws[i] * (h0 / hs[i]) for i in (0, 1))
    w1 = sum(ws[i] * (h1 / hs[i]) for i in (2, 3))
    total_w = max(w0, w1)
    total_h = h0 + h1

    px_per_inch = 120
    fig = plt.figure(figsize=(total_w / px_per_inch, total_h / px_per_inch))

    def add_row(
        indices: tuple[int, int], row_h: int, y_bottom_frac: float, row_total_w: float
    ) -> None:
        x_px = max(0.0, (total_w - row_total_w) / 2.0)
        for idx in indices:
            hi, wi = hs[idx], ws[idx]
            wi_scaled = wi * (row_h / hi)
            ax = fig.add_axes(
                [
                    x_px / total_w,
                    y_bottom_frac,
                    wi_scaled / total_w,
                    row_h / total_h,
                ]
            )
            ax.imshow(images[idx])
            ax.axis("off")
            x_px += wi_scaled

    # Bottom row: lDDT panels; top row: RMSD panels (matplotlib y is from bottom)
    add_row((2, 3), h1, 0.0, w1)
    add_row((0, 1), h0, h1 / total_h, w0)

    out = args.output
    if out is None:
        out = fig_dir / "kincore_geometric_correlation_four_panel.png"
    else:
        out = out.resolve()

    out.parent.mkdir(parents=True, exist_ok=True)
    suffix = out.suffix.lower()
    if suffix == ".pdf":
        fig.savefig(out, format="pdf", bbox_inches="tight", pad_inches=0.02)
    else:
        fig.savefig(out, dpi=args.dpi, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
