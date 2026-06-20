"""
Shared plot styling for KinConfBench analysis (methods, seaborn colors).

Use PRIMARY_METHODS + method_colors() for Boltz-2 / Chai-1 / Protenix-only figures.
"""

from __future__ import annotations

PRIMARY_METHODS: tuple[str, ...] = ("boltz", "chai", "protenix")

METHOD_DISPLAY_NAMES: dict[str, str] = {
    "boltz": "Boltz-2",
    "chai": "Chai-1",
    "protenix": "Protenix",
    "rfaa": "RoseTTAFold-AA",
}


def parse_methods_arg(s: str | None, default: tuple[str, ...] = PRIMARY_METHODS) -> tuple[str, ...]:
    """Comma-separated lowercase method keys."""
    if not s or not str(s).strip():
        return default
    return tuple(x.strip().lower() for x in str(s).split(",") if x.strip())


def method_colors(
    methods: tuple[str, ...] | None = None,
    *,
    palette: str = "deep",
) -> dict[str, tuple]:
    """
    Map each method to a color (RGBA from seaborn, or hex from matplotlib tab10 fallback).
    """
    methods = methods or PRIMARY_METHODS
    try:
        import seaborn as sns

        pal = sns.color_palette(palette, n_colors=len(methods))
        return {m: pal[i] for i, m in enumerate(methods)}
    except ImportError:
        import matplotlib.cm as cm

        cmap = cm.get_cmap("tab10")
        return {m: cmap(i % 10) for i, m in enumerate(methods)}


def labels_for_methods(methods: tuple[str, ...]) -> list[str]:
    return [METHOD_DISPLAY_NAMES.get(m, m) for m in methods]
