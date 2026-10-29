"""Render the bootstrap coverage guidance from simulation results.

    uv run --group sim python -m simulations.report

Reads ``coverage.csv`` and ``run_info.json`` from ``simulations/results/`` (the
P5 grid) and ``simulations/results/coders/`` (coders that differ), draws the
figures (light and dark PNGs, served through ``<picture>``), and fills the
template ``simulations/templates/bootstrap-coverage.md``. The output is
``docs/guidance/bootstrap-coverage.md``. Edit the template, not the output.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "simulations" / "results"
TEMPLATE = ROOT / "simulations" / "templates" / "bootstrap-coverage.md"
OUTPUT = ROOT / "docs" / "guidance" / "bootstrap-coverage.md"

#: Chart tokens from the dataviz reference palette; categorical slots 1-3 in order.
TOKENS: dict[str, dict[str, str]] = {
    "light": {
        "surface": "#fcfcfb",
        "ink": "#0b0b0b",
        "ink2": "#52514e",
        "muted": "#898781",
        "grid": "#e1e0d9",
        "axis": "#c3c2b7",
        "band": "#f0efec",
        "units": "#2a78d6",
        "kalpha": "#eb6834",
        "bca": "#1baf7a",
        "coders": "#eda100",
        "two_way": "#e87ba4",
        "jackknife": "#008300",
    },
    "dark": {
        "surface": "#1a1a19",
        "ink": "#f0efec",
        "ink2": "#c3c2b7",
        "muted": "#898781",
        "grid": "#2c2c2a",
        "axis": "#383835",
        "band": "#2c2c2a",
        "units": "#3987e5",
        "kalpha": "#d95926",
        "bca": "#199e70",
        "coders": "#c98500",
        "two_way": "#d55181",
        "jackknife": "#008300",
    },
}
SCHEMES = ("units", "kalpha", "bca")
#: Coder-grid schemes; their colours are slots 1, 4, 5 and 6, so units keeps its blue
#: and no colour means one scheme here and another in the P5 figures.
CODER_SCHEMES = ("units", "coders", "two_way", "jackknife")
LABELS = {
    "units": "units (percentile)",
    "units_group": "units, group level",
    "kalpha": "KALPHA",
    "bca": "units (BCa)",
    "coders": "coders",
    "two_way": "two_way",
    "jackknife": "coder jackknife",
}
MARKERS = {"units": "o", "kalpha": "s", "bca": "^", "coders": "s", "two_way": "^", "jackknife": "D"}
FACTOR_LABELS = {
    "skewed": "Skewed prevalence (80/10/10)",
    "coders2": "2 coders",
    "coders5": "5 coders",
    "accuracy60": "Accuracy 0.6",
    "accuracy95": "Accuracy 0.95",
    "mcar": "20% failures, at random",
    "informative": "Failures on hard units",
    "correlated": "Correlated errors",
    "clustered": "Clustered units",
}
PX = 0.75  # points per pixel at 96 dpi


def load(results: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    data = pd.read_csv(results / "coverage.csv")
    info = json.loads((results / "run_info.json").read_text(encoding="utf-8"))
    return data, info


# -- figures --------------------------------------------------------------------------


def _floor(coverage: pd.Series) -> float:
    """A coverage axis floor at a 5-point step below the lowest value (at most 60%)."""
    return min(0.6, float(int(coverage.min() * 20) / 20) - 0.05)


def _style(ax: plt.Axes, t: dict[str, str]) -> None:
    ax.set_facecolor(t["surface"])
    ax.grid(True, axis="y", color=t["grid"], linewidth=1 * PX)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(t["axis"])
    ax.tick_params(colors=t["muted"], labelcolor=t["ink2"], length=0, labelsize=8.5)


def _legend(fig: plt.Figure, t: dict[str, str], handles: list[Any]) -> None:
    legend = fig.legend(
        handles=handles,
        loc="upper center",
        ncol=len(handles),
        frameon=False,
        fontsize=9,
        bbox_to_anchor=(0.5, 0.995),
        handlelength=1.6,
    )
    for text in legend.get_texts():
        text.set_color(t["ink"])


def _series(ax: plt.Axes, t: dict[str, str], x: list[int], y: list[float], scheme: str) -> Any:
    (line,) = ax.plot(
        x,
        y,
        color=t[scheme],
        linewidth=2 * PX,
        marker=MARKERS[scheme],
        markersize=8 * PX,
        markeredgecolor=t["surface"],
        markeredgewidth=2 * PX,
        solid_capstyle="round",
        solid_joinstyle="round",
        label=LABELS[scheme],
    )
    return line


def figure_coverage_by_n(data: pd.DataFrame, mode: str, path: Path) -> None:
    """Coverage (top) and mean width (bottom) against n, for nominal and ordinal data."""
    t = TOKENS[mode]
    fig, axes = plt.subplots(2, 2, figsize=(8.4, 5.6), sharex=True, dpi=200)
    fig.patch.set_facecolor(t["surface"])
    handles = []
    for col, name in enumerate(("baseline", "ordinal")):
        cells = data[data["scenario_name"] == name]
        sizes = sorted(cells["scenario_n_units"].unique())
        positions = list(range(len(sizes)))
        for row, metric in enumerate(("coverage", "mean_width")):
            ax = axes[row][col]
            _style(ax, t)
            if metric == "coverage":
                # The reference line stops at the data; its label sits in the free
                # space to the right, clear of every series.
                ax.plot(
                    [-0.3, positions[-1] + 0.15],
                    [0.95, 0.95],
                    color=t["ink2"],
                    linewidth=1 * PX,
                    zorder=1,
                )
                ax.text(
                    positions[-1] + 0.25,
                    0.95,
                    "95%",
                    color=t["ink2"],
                    fontsize=8,
                    ha="left",
                    va="center",
                )
                ax.set_ylim(_floor(data["coverage"]), 1.0)
                ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
            for scheme in SCHEMES:
                rows = cells[cells["scheme"] == scheme].set_index("scenario_n_units")
                line = _series(ax, t, positions, [rows.loc[n, metric] for n in sizes], scheme)
                if row == 0 and col == 0:
                    handles.append(line)
            ax.set_xticks(positions, [f"n = {n}" for n in sizes])
            ax.set_xlim(-0.3, positions[-1] + 0.7)
            title = "Nominal, 3 categories" if name == "baseline" else "Ordinal, 5-point scale"
            if row == 0:
                ax.set_title(title, color=t["ink"], fontsize=10, loc="left")
            if col == 0:
                ylabel = "Coverage of 95% interval" if metric == "coverage" else "Mean width"
                ax.set_ylabel(ylabel, color=t["ink2"], fontsize=9)
    _legend(fig, t, handles)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, facecolor=t["surface"])
    plt.close(fig)


def figure_coverage_by_factor(data: pd.DataFrame, mode: str, path: Path) -> None:
    """Coverage per scheme for each one-factor variation (dot plot)."""
    t = TOKENS[mode]
    cells = data[data["scenario_name"].isin(FACTOR_LABELS)]
    order = [
        (name, n)
        for name in FACTOR_LABELS
        for n in sorted(cells[cells["scenario_name"] == name]["scenario_n_units"].unique())
    ]
    fig, ax = plt.subplots(figsize=(8.4, 0.34 * len(order) + 1.2), dpi=200)
    fig.patch.set_facecolor(t["surface"])
    ax.set_facecolor(t["surface"])
    ax.axvspan(0.94, 0.96, color=t["band"], zorder=0)
    ax.axvline(0.95, color=t["ink2"], linewidth=1 * PX, zorder=1)
    handles = []
    for scheme in SCHEMES:
        xs, ys = [], []
        for y, (name, n) in enumerate(order):
            row = cells[
                (cells["scenario_name"] == name)
                & (cells["scenario_n_units"] == n)
                & (cells["scheme"] == scheme)
            ]
            xs.append(float(row["coverage"].iloc[0]))
            ys.append(y)
        handles.append(
            ax.scatter(
                xs,
                ys,
                s=(8 * PX) ** 2 * 1.6,
                marker=MARKERS[scheme],
                color=t[scheme],
                edgecolors=t["surface"],
                linewidths=2 * PX,
                zorder=3,
                label=LABELS[scheme],
            )
        )
    ax.set_yticks(range(len(order)), [f"{FACTOR_LABELS[name]}, n = {n}" for name, n in order])
    ax.invert_yaxis()
    ax.set_xlim(_floor(cells["coverage"]), 1.0)
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.grid(True, axis="x", color=t["grid"], linewidth=1 * PX)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors=t["muted"], labelcolor=t["ink2"], length=0, labelsize=8.5)
    ax.set_xlabel(
        "Coverage of the 95% interval (shaded: within ±1 point of nominal)",
        color=t["ink2"],
        fontsize=9,
    )
    _legend(fig, t, handles)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.5 / (0.34 * len(order) + 1.2)))
    fig.savefig(path, facecolor=t["surface"])
    plt.close(fig)


def figure_coverage_by_spread(data: pd.DataFrame, mode: str, path: Path) -> None:
    """Coverage against coder spread, per panel size, at n = 100 (nominal).

    Top row: each scheme against the claim it is built for. Bottom row: every
    scheme against new coders on new questions.
    """
    t = TOKENS[mode]
    cells = data[(data["scenario_name"] == "nominal") & (data["scenario_n_units"] == 100)]
    spreads = sorted(cells["coder_spread"].unique())
    panels = sorted(cells["scenario_n_coders"].unique())
    positions = list(range(len(spreads)))
    rows = (
        ("coverage_own", "Coverage of its own claim"),
        ("coverage_both_random", "Coverage of new coders\non new questions"),
    )
    fig, axes = plt.subplots(
        len(rows), len(panels), figsize=(8.4, 5.6), sharex=True, sharey=True, dpi=200
    )
    fig.patch.set_facecolor(t["surface"])
    floor = _floor(pd.concat([cells[metric] for metric, _ in rows]))
    handles = []
    for row, (metric, ylabel) in enumerate(rows):
        for col, m in enumerate(panels):
            ax = axes[row][col]
            _style(ax, t)
            ax.plot([-0.3, positions[-1] + 0.15], [0.95, 0.95], color=t["ink2"], linewidth=1 * PX)
            if col == len(panels) - 1:
                ax.text(
                    positions[-1] + 0.25,
                    0.95,
                    "95%",
                    color=t["ink2"],
                    fontsize=8,
                    ha="left",
                    va="center",
                )
            panel = cells[cells["scenario_n_coders"] == m]
            for scheme in CODER_SCHEMES:
                series = panel[panel["scheme"] == scheme].set_index("coder_spread")
                line = _series(ax, t, positions, [series.loc[s, metric] for s in spreads], scheme)
                if row == 0 and col == 0:
                    handles.append(line)
            ax.set_ylim(floor, 1.0)
            ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
            ax.set_xticks(positions, [f"{TAU} = {s:g}" for s in spreads])
            ax.set_xlim(-0.3, positions[-1] + 0.7)
            if row == 0:
                ax.set_title(f"{m} coders, 100 questions", color=t["ink"], fontsize=10, loc="left")
            if col == 0:
                ax.set_ylabel(ylabel, color=t["ink2"], fontsize=9)
    _legend(fig, t, handles)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, facecolor=t["surface"])
    plt.close(fig)


#: name: (draw, alt text, results it is drawn from)
FIGURES = {
    "coverage_by_n": (
        figure_coverage_by_n,
        "Coverage and mean width against sample size for the three schemes",
        "base",
    ),
    "coverage_by_factor": (
        figure_coverage_by_factor,
        "Coverage of each scheme under each one-factor variation",
        "base",
    ),
    "coverage_by_spread": (
        figure_coverage_by_spread,
        "Coverage of the units, coders and two-way bootstraps and the coder jackknife "
        "as coders differ more",
        "coders",
    ),
}


def picture(name: str, alt: str) -> str:
    return (
        "<picture>\n"
        f'  <source media="(prefers-color-scheme: dark)" srcset="figures/{name}-dark.png">\n'
        f'  <img alt="{alt}" src="figures/{name}-light.png" width="840">\n'
        "</picture>"
    )


# -- tables ---------------------------------------------------------------------------


#: Shown for values that are undefined or not applicable (an en dash).
MISSING = chr(0x2013)
TIMES = chr(0x00D7)
TAU = chr(0x03C4)


def _pct(x: float) -> str:
    return MISSING if pd.isna(x) else f"{x:.1%}"


def _num(x: float, digits: int = 3) -> str:
    return MISSING if pd.isna(x) else f"{x:.{digits}f}"


def _markdown(frame: pd.DataFrame) -> str:
    header = "| " + " | ".join(frame.columns) + " |"
    rule = "| " + " | ".join("---" for _ in frame.columns) + " |"
    rows = ["| " + " | ".join(str(v) for v in row) + " |" for row in frame.to_numpy()]
    return "\n".join([header, rule, *rows])


def _cell(data: pd.DataFrame, name: str, n: int, scheme: str) -> pd.Series:
    rows = data[
        (data["scenario_name"] == name)
        & (data["scenario_n_units"] == n)
        & (data["scheme"] == scheme)
    ]
    return rows.iloc[0]


def _scenario_label(name: str, n: int) -> str:
    base = {"baseline": "Nominal (baseline)", "ordinal": "Ordinal 5-point"}
    return f"{base.get(name) or FACTOR_LABELS[name]}, n = {n}"


def table_sample_size(data: pd.DataFrame) -> str:
    rows = []
    for name in ("baseline", "ordinal"):
        for n in sorted(data[data["scenario_name"] == name]["scenario_n_units"].unique()):
            cells = {s: _cell(data, name, n, s) for s in SCHEMES}
            rows.append(
                {
                    "Scenario": _scenario_label(name, n),
                    "True alpha": _num(cells["units"]["true_alpha"]),
                    **{f"{LABELS[s]} coverage": _pct(cells[s]["coverage"]) for s in SCHEMES},
                    **{f"{LABELS[s]} width": _num(cells[s]["mean_width"], 2) for s in SCHEMES},
                }
            )
    return _markdown(pd.DataFrame(rows))


def table_factors(data: pd.DataFrame) -> str:
    rows = []
    for name in FACTOR_LABELS:
        for n in sorted(data[data["scenario_name"] == name]["scenario_n_units"].unique()):
            cells = {s: _cell(data, name, n, s) for s in SCHEMES}
            rows.append(
                {
                    "Scenario": _scenario_label(name, n),
                    "True alpha": _num(cells["units"]["true_alpha"]),
                    **{LABELS[s]: _pct(cells[s]["coverage"]) for s in SCHEMES},
                    "BCa undefined": _pct(cells["bca"]["undefined_rate"]),
                    "Units below / above": f"{_pct(cells['units']['miss_below'])} / "
                    f"{_pct(cells['units']['miss_above'])}",
                }
            )
    return _markdown(pd.DataFrame(rows))


def table_clusters(data: pd.DataFrame) -> str:
    rows = []
    for n in sorted(data[data["scenario_name"] == "clustered"]["scenario_n_units"].unique()):
        for scheme in ("units", "units_group", "kalpha", "bca"):
            cell = _cell(data, "clustered", n, scheme)
            rows.append(
                {
                    "Units (clusters)": f"{n} ({n // 5})",
                    "Scheme": LABELS[scheme] + (" (group level)" if scheme == "bca" else ""),
                    "Coverage": _pct(cell["coverage"]),
                    "Mean width": _num(cell["mean_width"], 3),
                }
            )
    return _markdown(pd.DataFrame(rows))


def table_failures(data: pd.DataFrame) -> str:
    rows = []
    for name in ("mcar", "informative"):
        for n in sorted(data[data["scenario_name"] == name]["scenario_n_units"].unique()):
            cell = _cell(data, name, n, "units")
            rows.append(
                {
                    "Scenario": _scenario_label(name, n),
                    "Complete-data alpha": _num(cell["complete_alpha"]),
                    "Alpha with failures dropped": _num(cell["true_alpha"]),
                    "Gap": f"{cell['true_alpha'] - cell['complete_alpha']:+.3f}",
                    "Units coverage of the dropped-failures value": _pct(cell["coverage"]),
                    "Policy range covers complete-data alpha": _pct(
                        cell["sensitivity_covers_complete"]
                    ),
                }
            )
    return _markdown(pd.DataFrame(rows))


def table_coders(data: pd.DataFrame) -> str:
    """Coverage (mean width) of each scheme against its own claim, per coder-grid cell."""
    rows = []
    for _, cell in data.drop_duplicates("cell").iterrows():
        cells = data[data["cell"] == cell["cell"]].set_index("scheme")

        def scored(scheme: str, cells: pd.DataFrame = cells) -> str:
            row = cells.loc[scheme]
            return f"{_pct(row['coverage_own'])} ({_num(row['mean_width'], 2)})"

        rows.append(
            {
                "Scenario": f"{cell['scenario_name'].capitalize()}, {TAU} = "
                f"{cell['coder_spread']:g}, {cell['scenario_n_coders']} coders, "
                f"n = {cell['scenario_n_units']}",
                "Population alpha": _num(cell["both_random"]),
                "units: same coders": scored("units"),
                "units: new coders": _pct(cells.loc["units", "coverage_both_random"]),
                "coders: same questions": scored("coders"),
                "jackknife: same questions": scored("jackknife"),
                "two_way: new coders and questions": scored("two_way"),
            }
        )
    return _markdown(pd.DataFrame(rows))


def run_info_text(info: dict[str, Any], data: pd.DataFrame) -> str:
    truth = "both_random" if info.get("grid") == "coders" else "true_alpha"
    mc_error = data.drop_duplicates("cell")[f"{truth}_mc_error"].max()
    v = info["versions"]
    return (
        f"{info['scenarios']} scenarios {TIMES} {info['n_sim']:,} replications {TIMES} "
        f"{info['n_resamples']:,} bootstrap resamples, seed {info['seed']}, "
        f"reliability-lab {v['reliability_lab']} (numpy {v['numpy']}, scipy {v['scipy']}, "
        f"pandas {v['pandas']}); {info['wall_seconds'] / 60:.0f} minutes on "
        f"{info['workers']} workers. The largest Monte Carlo error of a population alpha is "
        f"{mc_error:.4f}, and a coverage near 95% has a Monte Carlo standard error of "
        f"about {(0.95 * 0.05 / info['n_sim']) ** 0.5:.1%}."
    )


#: name: (table, results it is drawn from)
TABLES = {
    "sample_size": (table_sample_size, "base"),
    "factors": (table_factors, "base"),
    "clusters": (table_clusters, "base"),
    "failures": (table_failures, "base"),
    "coders": (table_coders, "coders"),
}


def render(results: Path = RESULTS, template: Path = TEMPLATE, output: Path = OUTPUT) -> Path:
    """``results`` holds the P5 grid, and its ``coders/`` folder the coder grid."""
    sources = {"base": load(results), "coders": load(results / "coders")}
    figures = output.parent / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    for name, (draw, _, source) in FIGURES.items():
        for mode in ("light", "dark"):
            draw(sources[source][0], mode, figures / f"{name}-{mode}.png")

    def fill(match: re.Match[str]) -> str:
        kind, _, name = match.group(1).partition(":")
        if kind == "table":
            table, source = TABLES[name]
            return table(sources[source][0])
        if kind == "figure":
            return picture(name, FIGURES[name][1])
        if kind == "run_info":
            data, info = sources[name or "base"]
            return run_info_text(info, data)
        raise KeyError(match.group(0))

    text = template.read_text(encoding="utf-8")
    body = re.sub(r"\{\{([a-z_]+(?::[a-z_]+)?)\}\}", fill, text)
    notice = (
        "<!-- Generated by simulations/report.py from simulations/results; "
        "edit simulations/templates/bootstrap-coverage.md instead. -->\n\n"
    )
    output.write_text(notice + body, encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results", type=Path, default=RESULTS)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    print(f"wrote {render(args.results, TEMPLATE, args.output)}")


if __name__ == "__main__":
    main()
