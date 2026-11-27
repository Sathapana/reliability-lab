"""Figures for reliability reports (requires the optional ``plot`` extra: matplotlib).

Reuses the light and dark dataviz token palettes of ``simulations/report.py``:

- :func:`plot_policy_forest`: horizontal interval forest plot across missing-data policies.
- :func:`plot_coincidence`: observed vs expected coincidence heatmaps (or residual difference).
- :func:`plot_coder_influence`: leave-one-coder-out influence chart from the coder jackknife.
"""

from __future__ import annotations

import math
import os
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import numpy.typing as npt

from reliability_lab.exceptions import DesignError
from reliability_lab.metrics.gwet import GwetResult
from reliability_lab.report import (
    AlphaEstimate,
    AlphaResult,
    BootstrapResult,
    ComparisonResult,
    JackknifeResult,
)

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

PlotMode = Literal["light", "dark"]
CoincidenceView = Literal["side_by_side", "residual"]

#: Chart tokens matching simulations/report.py (light and dark palettes).
TOKENS: dict[str, dict[str, str]] = {
    "light": {
        "surface": "#fcfcfb",
        "ink": "#0b0b0b",
        "ink2": "#52514e",
        "muted": "#898781",
        "grid": "#e1e0d9",
        "axis": "#c3c2b7",
        "band": "#f0efec",
        "primary": "#2a78d6",
        "accent": "#eb6834",
        "positive": "#1baf7a",
        "warn": "#eda100",
    },
    "dark": {
        "surface": "#1a1a19",
        "ink": "#f0efec",
        "ink2": "#c3c2b7",
        "muted": "#898781",
        "grid": "#2c2c2a",
        "axis": "#383835",
        "band": "#2c2c2a",
        "primary": "#3987e5",
        "accent": "#d95926",
        "positive": "#199e70",
        "warn": "#c98500",
    },
}

PX = 0.75  # points per pixel at 96 dpi


def _import_pyplot() -> Any:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "Plotting needs matplotlib; install it with: pip install 'reliability-lab[plot]'"
        ) from exc
    return plt


def _resolve_tokens(mode: str) -> dict[str, str]:
    if mode not in TOKENS:
        raise DesignError(f"unknown plot mode {mode!r}; choose 'light' or 'dark'")
    return TOKENS[mode]


def _style_horizontal_ax(ax: Axes, t: dict[str, str]) -> None:
    ax.set_facecolor(t["surface"])
    ax.grid(True, axis="x", color=t["grid"], linewidth=1 * PX)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(t["axis"])
    ax.tick_params(colors=t["muted"], labelcolor=t["ink2"], length=0, labelsize=9)


def _save_if_requested(fig: Figure, path: str | os.PathLike[str] | None, t: dict[str, str]) -> None:
    if path is not None:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=180, facecolor=t["surface"], bbox_inches="tight")


def plot_policy_forest(
    result: BootstrapResult | ComparisonResult | JackknifeResult | GwetResult | AlphaResult,
    *,
    mode: PlotMode = "light",
    ax: Axes | None = None,
    path: str | os.PathLike[str] | None = None,
) -> Figure:
    """Plot horizontal confidence intervals across missing-data policies.

    Accepts a :class:`BootstrapResult`, :class:`ComparisonResult`,
    :class:`JackknifeResult`, :class:`GwetResult` (using its analytical intervals),
    or :class:`AlphaResult` (via ``result.bootstrap()`` when callers pass a bootstrap
    result, or raises :class:`DesignError` if given an un-bootstrapped ``AlphaResult``).
    """
    plt = _import_pyplot()
    t = _resolve_tokens(mode)

    if isinstance(result, AlphaResult):
        raise DesignError(
            "AlphaResult holds point estimates only; pass result.bootstrap(...) or "
            "result.coder_jackknife(...) to plot_policy_forest()"
        )

    rows: list[tuple[str, float, float, float]] = []
    ref_lines: list[tuple[float, str, str]] = []

    if isinstance(result, BootstrapResult):
        for name, interval in result.intervals.items():
            rows.append((name, interval.point, interval.low, interval.high))
        xlabel = f"Agreement ({result.measurement}, {result.confidence:.0%} CI)"
        title = f"Missing-data policy forest ({result.design})"
        if result.alpha_min is not None and math.isfinite(result.alpha_min):
            ref_lines.append((result.alpha_min, t["accent"], f"min ({result.alpha_min:g})"))
    elif isinstance(result, ComparisonResult):
        for name, interval in result.differences.items():
            rows.append((name, interval.point, interval.low, interval.high))
        xlabel = f"Difference b - a ({result.measurement}, {result.arm_a.confidence:.0%} CI)"
        title = f"Paired comparison forest ({result.claim})"
        ref_lines.append((0.0, t["muted"], "0"))
        if result.margin is not None and math.isfinite(result.margin):
            ref_lines.append((-result.margin, t["accent"], f"-margin ({-result.margin:g})"))
            if result.test == "equivalence":
                ref_lines.append((result.margin, t["accent"], f"+margin ({result.margin:g})"))
    elif isinstance(result, JackknifeResult):
        for name, interval in result.intervals.items():
            rows.append((name, interval.point, interval.low, interval.high))
        xlabel = f"Alpha ({result.measurement}, {result.confidence:.0%} jackknife t-CI)"
        title = f"Coder jackknife policy forest ({result.design})"
    elif isinstance(result, GwetResult):
        for name, est in result.estimates.items():
            rows.append((name, est.ac, est.low, est.high))
        xlabel = f"Gwet's {result.coefficient_name} ({result.weights_name}, analytical CI)"
        title = f"Gwet's {result.coefficient_name} across missing-data policies ({result.design})"
    else:
        raise DesignError(f"unsupported result type {type(result).__name__} for plot_policy_forest")

    if ax is None:
        fig_h = max(2.2, 0.65 * len(rows) + 1.1)
        fig, ax_use = plt.subplots(figsize=(6.8, fig_h))
    else:
        ax_use = ax
        fig = ax_use.figure
    fig.patch.set_facecolor(t["surface"])
    _style_horizontal_ax(ax_use, t)

    y_positions = np.arange(len(rows))[::-1]
    labels = [r[0] for r in rows]

    for y, (_label, pt, lo, hi) in zip(y_positions, rows, strict=True):
        if math.isfinite(lo) and math.isfinite(hi):
            ax_use.plot(
                [lo, hi],
                [y, y],
                color=t["primary"],
                linewidth=2.5 * PX,
                solid_capstyle="round",
                zorder=2,
            )
        if math.isfinite(pt):
            ax_use.plot(
                [pt],
                [y],
                marker="o",
                color=t["primary"],
                markersize=8 * PX,
                markeredgecolor=t["surface"],
                markeredgewidth=1.5 * PX,
                zorder=3,
            )

    for ref_val, ref_col, _ref_lbl in ref_lines:
        ax_use.axvline(ref_val, color=ref_col, linestyle="--", linewidth=1.2 * PX, zorder=1)

    ax_use.set_yticks(y_positions)
    ax_use.set_yticklabels(labels, color=t["ink"])
    ax_use.set_ylim(-0.6, max(0.6, len(rows) - 0.4))
    ax_use.set_xlabel(xlabel, color=t["ink2"], fontsize=9)
    ax_use.set_title(title, color=t["ink"], fontsize=10, loc="left", pad=8)

    _save_if_requested(fig, path, t)
    return fig  # type: ignore[no-any-return]


def _expected_coincidence_matrix(observed: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Expected coincidences e_ck = n_c (n_k - delta_ck) / (n - 1) under Krippendorff's null."""
    n_c = observed.sum(axis=1)
    n = float(n_c.sum())
    v = len(n_c)
    if n <= 1.0 or v == 0:
        return np.zeros_like(observed, dtype=float)
    expected = np.outer(n_c, n_c) / (n - 1.0)
    diag_idx = np.arange(v)
    expected[diag_idx, diag_idx] = n_c * (n_c - 1.0) / (n - 1.0)
    return np.asarray(expected, dtype=float)


def plot_coincidence(
    result: AlphaResult | AlphaEstimate,
    *,
    policy: str = "as_missing",
    view: CoincidenceView = "side_by_side",
    mode: PlotMode = "light",
    ax: Axes | Sequence[Axes] | None = None,
    path: str | os.PathLike[str] | None = None,
) -> Figure:
    """Plot observed against expected coincidences for an AlphaResult or AlphaEstimate.

    When ``view="side_by_side"`` (the default), renders a 1x2 figure with Observed
    coincidences ``o_ck`` on the left and Expected coincidences ``e_ck`` on the right
    sharing a common scale. When ``view="residual"``, renders a single heatmap of
    ``o_ck - e_ck``.
    """
    plt = _import_pyplot()
    t = _resolve_tokens(mode)
    if view not in ("side_by_side", "residual"):
        raise DesignError(f"unknown coincidence view {view!r}; choose 'side_by_side' or 'residual'")

    if isinstance(result, AlphaResult):
        if policy not in result.estimates:
            raise DesignError(
                f"policy {policy!r} is not in result.estimates {list(result.estimates)}"
            )
        estimate = result.estimates[policy]
    elif isinstance(result, AlphaEstimate):
        estimate = result
    else:
        raise DesignError(
            f"plot_coincidence expects AlphaResult or AlphaEstimate, got {type(result).__name__}"
        )

    coinc_df = estimate.coincidence
    labels = [str(v) for v in coinc_df.index]
    obs = coinc_df.to_numpy(dtype=float)
    exp = _expected_coincidence_matrix(obs)
    v = len(labels)
    ticks = np.arange(v)

    if view == "side_by_side":
        if ax is None:
            fig, axes_arr = plt.subplots(1, 2, figsize=(8.8, 4.0))
            ax_obs, ax_exp = axes_arr[0], axes_arr[1]
        elif isinstance(ax, Sequence) and len(ax) == 2:
            ax_obs, ax_exp = ax[0], ax[1]
            fig = ax_obs.figure
        else:
            raise DesignError("view='side_by_side' requires ax=None or a sequence of 2 Axes")

        fig.patch.set_facecolor(t["surface"])
        vmax = float(max(float(np.max(obs, initial=0.0)), float(np.max(exp, initial=0.0)), 1e-9))
        cmap = "Blues" if mode == "light" else "cividis"

        for panel_ax, mat, panel_title in (
            (ax_obs, obs, f"Observed coincidences ({estimate.name})"),
            (ax_exp, exp, f"Expected coincidences ({estimate.name})"),
        ):
            panel_ax.set_facecolor(t["surface"])
            panel_ax.imshow(mat, vmin=0.0, vmax=vmax, cmap=cmap, aspect="equal")
            panel_ax.set_xticks(ticks)
            panel_ax.set_xticklabels(labels, color=t["ink2"], fontsize=8.5)
            panel_ax.set_yticks(ticks)
            panel_ax.set_yticklabels(labels, color=t["ink2"], fontsize=8.5)
            panel_ax.set_title(panel_title, color=t["ink"], fontsize=9.5, pad=6)
            for spine in panel_ax.spines.values():
                spine.set_color(t["axis"])
            if v <= 12:
                for r_i in range(v):
                    for c_i in range(v):
                        val = mat[r_i, c_i]
                        txt_col = (
                            t["surface"]
                            if (mode == "light" and val > 0.6 * vmax)
                            or (mode == "dark" and val > 0.7 * vmax)
                            else t["ink"]
                        )
                        panel_ax.text(
                            c_i,
                            r_i,
                            f"{val:.1f}",
                            ha="center",
                            va="center",
                            fontsize=7.5,
                            color=txt_col,
                        )
    else:
        if ax is None:
            fig, ax_res = plt.subplots(figsize=(4.8, 4.2))
        elif not isinstance(ax, Sequence):
            ax_res = ax
            fig = ax_res.figure
        else:
            raise DesignError("view='residual' requires a single Axes or ax=None")

        fig.patch.set_facecolor(t["surface"])
        diff = obs - exp
        bound = float(max(float(np.max(np.abs(diff), initial=0.0)), 1e-9))
        ax_res.set_facecolor(t["surface"])
        ax_res.imshow(diff, vmin=-bound, vmax=bound, cmap="coolwarm", aspect="equal")
        ax_res.set_xticks(ticks)
        ax_res.set_xticklabels(labels, color=t["ink2"], fontsize=8.5)
        ax_res.set_yticks(ticks)
        ax_res.set_yticklabels(labels, color=t["ink2"], fontsize=8.5)
        ax_res.set_title(
            f"Observed - Expected coincidences ({estimate.name})",
            color=t["ink"],
            fontsize=9.5,
            pad=6,
        )
        for spine in ax_res.spines.values():
            spine.set_color(t["axis"])
        if v <= 12:
            for r_i in range(v):
                for c_i in range(v):
                    val = diff[r_i, c_i]
                    ax_res.text(
                        c_i,
                        r_i,
                        f"{val:+.1f}",
                        ha="center",
                        va="center",
                        fontsize=7.5,
                        color=t["ink"],
                    )

    _save_if_requested(fig, path, t)
    return fig  # type: ignore[no-any-return]


def plot_coder_influence(
    result: JackknifeResult | AlphaResult,
    *,
    policy: str = "as_missing",
    mode: PlotMode = "light",
    ax: Axes | None = None,
    path: str | os.PathLike[str] | None = None,
) -> Figure:
    """Plot each coder's leave-one-out influence (``alpha - alpha_without``).

    Accepts a :class:`JackknifeResult` or an :class:`AlphaResult` (calling
    ``result.coder_jackknife()`` automatically).
    """
    plt = _import_pyplot()
    t = _resolve_tokens(mode)

    if isinstance(result, AlphaResult):
        jack = result.coder_jackknife()
    elif isinstance(result, JackknifeResult):
        jack = result
    else:
        raise DesignError(
            f"plot_coder_influence expects JackknifeResult or AlphaResult, got "
            f"{type(result).__name__}"
        )

    sub = jack.influence[jack.influence["estimate"] == policy].copy()
    if sub.empty:
        raise DesignError(
            f"policy {policy!r} is not in jackknife influence table "
            f"{sorted(jack.influence['estimate'].unique().tolist())}"
        )

    sub = sub.sort_values("influence", ascending=True, kind="mergesort")
    coders = [str(c) for c in sub["coder"].tolist()]
    influences = [float(v) for v in sub["influence"].tolist()]

    if ax is None:
        fig_h = max(2.4, 0.45 * len(coders) + 1.2)
        fig, ax_use = plt.subplots(figsize=(6.6, fig_h))
    else:
        ax_use = ax
        fig = ax_use.figure
    fig.patch.set_facecolor(t["surface"])
    _style_horizontal_ax(ax_use, t)

    y_positions = np.arange(len(coders))
    ax_use.axvline(0.0, color=t["muted"], linestyle="--", linewidth=1.2 * PX, zorder=1)

    for y, inf in zip(y_positions, influences, strict=True):
        if math.isfinite(inf):
            col = t["positive"] if inf >= 0 else t["accent"]
            ax_use.plot(
                [0.0, inf],
                [y, y],
                color=col,
                linewidth=2.2 * PX,
                solid_capstyle="round",
                zorder=2,
            )
            ax_use.plot(
                [inf],
                [y],
                marker="o",
                color=col,
                markersize=7.5 * PX,
                markeredgecolor=t["surface"],
                markeredgewidth=1.4 * PX,
                zorder=3,
            )

    ax_use.set_yticks(y_positions)
    ax_use.set_yticklabels(coders, color=t["ink"])
    ax_use.set_ylim(-0.6, max(0.6, len(coders) - 0.4))
    ax_use.set_xlabel(
        f"Influence on {policy} (alpha - alpha without coder)",
        color=t["ink2"],
        fontsize=9,
    )
    ax_use.set_title(
        f"Coder influence ({jack.design}, {jack.measurement})",
        color=t["ink"],
        fontsize=10,
        loc="left",
        pad=8,
    )

    _save_if_requested(fig, path, t)
    return fig  # type: ignore[no-any-return]
