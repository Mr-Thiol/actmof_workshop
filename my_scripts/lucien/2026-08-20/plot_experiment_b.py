#!/usr/bin/env python3
"""Plot Experiment B catastrophic-error trends from the main table."""

from __future__ import annotations

import argparse
import os
import re
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "actmof-matplotlib"))

import matplotlib.pyplot as plt
import pandas as pd


# Edit this path when you want to pin the script to one table.
# If this relative path is not found, the script searches for experiment_b_main_table.tex.
TABLE_PATH = Path("../experiment_b_main_table.tex")

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
SEARCH_ROOT = REPO_ROOT / "results" / "lucien" / "2026-08-20" / "experiment_b"
FIGURE_DIR = REPO_ROOT / "figures" / "lucien" / "2026-08-20" / "experiment_b"
ROUND_RE = re.compile(r"^(?:Initial\s+(?P<initial>\d+)|\+(?P<added>\d+))$")
CELL_RE = re.compile(
    r"(?P<mean>[+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*"
    r"(?:\\pm|\+/-|±)\s*"
    r"(?P<err>[+-]?(?:\d+(?:\.\d*)?|\.\d+))"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table-path", type=Path, default=TABLE_PATH)
    parser.add_argument("--search-root", type=Path, default=SEARCH_ROOT)
    parser.add_argument("--output-dir", type=Path, default=FIGURE_DIR)
    parser.add_argument("--output-name", default="catastrophic_error_rate_trend.png")
    parser.add_argument(
        "--use-table-error",
        action="store_true",
        help="shade the +/- value in the LaTeX table instead of the adjacent summary CSV std",
    )
    parser.add_argument(
        "--disable-std",
        action="store_true",
        help="plot only mean curves, without shaded std bands",
    )
    return parser.parse_args()


def resolve_table_path(path: Path, search_root: Path) -> Path:
    candidates = []
    if path.is_absolute():
        candidates.append(path)
    else:
        candidates.extend([Path.cwd() / path, SCRIPT_DIR / path])
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    matches = sorted(search_root.glob("**/experiment_b_main_table.tex"))
    if not matches:
        raise FileNotFoundError(
            f"Could not find {path} or any experiment_b_main_table.tex under {search_root}"
        )
    if len(matches) > 1:
        print("Found multiple Experiment B tables; using the newest by path sort:")
        for match in matches:
            print(f"  {match}")
    return matches[-1].resolve()


def clean_latex_text(text: str) -> str:
    return (
        text.replace(r"\_", "_")
        .replace(r"\%", "%")
        .replace(r"\&", "&")
        .replace("$", "")
        .strip()
    )


def parse_round_points(labels: list[str]) -> list[int]:
    points: list[int] = []
    initial_points: int | None = None
    for label in labels:
        match = ROUND_RE.match(label.strip())
        if not match:
            raise ValueError(f"Could not parse round label {label!r}")
        if match.group("initial") is not None:
            initial_points = int(match.group("initial"))
            points.append(initial_points)
            continue
        if initial_points is None:
            raise ValueError("Found acquisition label before an Initial N label")
        points.append(initial_points + int(match.group("added")))
    return points


def parse_latex_table(table_path: Path) -> tuple[pd.DataFrame, list[str], list[int]]:
    rows = []
    columns: list[str] | None = None
    for raw_line in table_path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("%") or line.startswith("\\"):
            continue
        line = line.rstrip("\\").strip()
        cells = [clean_latex_text(cell) for cell in line.split("&")]
        if cells[0] == "Method":
            columns = cells
            continue
        if columns is None or len(cells) != len(columns):
            continue
        row = dict(zip(columns, cells))
        rows.append(row)

    if not rows or columns is None:
        raise ValueError(f"Could not parse table rows from {table_path}")

    round_labels = [label for label in columns[1:] if label != "Error AUC"]
    explored_points = parse_round_points(round_labels)
    records = []
    for row in rows:
        for label, explored in zip(round_labels, explored_points):
            match = CELL_RE.search(row[label])
            if not match:
                raise ValueError(f"Could not parse mean/error cell {row[label]!r}")
            records.append(
                {
                    "method_label": row["Method"],
                    "round_label": label,
                    "explored_points": explored,
                    "catastrophic_error_rate_mean": float(match.group("mean")),
                    "catastrophic_error_rate_table_error": float(match.group("err")),
                }
            )
    return pd.DataFrame(records), round_labels, explored_points


def load_plot_data(table_path: Path, use_table_error: bool) -> pd.DataFrame:
    table_df, round_labels, explored_points = parse_latex_table(table_path)
    summary_path = table_path.with_name("experiment_b_summary.csv")
    if use_table_error or not summary_path.exists():
        if not summary_path.exists():
            print(f"Adjacent summary CSV not found; using table +/- values from {table_path}")
        table_df["catastrophic_error_rate_std"] = table_df["catastrophic_error_rate_table_error"]
        return table_df

    summary = pd.read_csv(summary_path)
    required = {
        "method_label",
        "round",
        "catastrophic_error_rate_mean",
        "catastrophic_error_rate_std",
    }
    missing = required - set(summary.columns)
    if missing:
        raise ValueError(f"{summary_path} is missing columns: {', '.join(sorted(missing))}")
    point_by_round = dict(enumerate(explored_points))
    label_by_round = dict(enumerate(round_labels))
    summary = summary.copy()
    summary["explored_points"] = summary["round"].map(point_by_round)
    summary["round_label"] = summary["round"].map(label_by_round)
    return summary[list(required) + ["explored_points", "round_label"]].dropna(subset=["explored_points"])


def set_zoomed_ylim(ax: plt.Axes, df: pd.DataFrame) -> None:
    means = df["catastrophic_error_rate_mean"].dropna()
    if means.empty:
        ax.set_ylim(0.0, 1.0)
        return
    y_min = float(means.min())
    y_max = float(means.max())
    span = max(y_max - y_min, 0.02)
    padding = max(span * 0.18, 0.01)
    ax.set_ylim(max(0.0, y_min - padding), min(1.0, y_max + padding))


def plot_catastrophic_error(df: pd.DataFrame, output_path: Path, title: str, show_std: bool) -> None:
    fig, ax = plt.subplots(figsize=(7.4, 4.5))
    colors = plt.get_cmap("tab10").colors
    for idx, (label, group) in enumerate(df.groupby("method_label", sort=False)):
        group = group.sort_values("explored_points")
        x = group["explored_points"]
        mean = group["catastrophic_error_rate_mean"]
        std = group["catastrophic_error_rate_std"].fillna(0.0)
        color = colors[idx % len(colors)]
        ax.plot(x, mean, marker="o", linewidth=2.0, markersize=4.5, label=label, color=color)
        if show_std:
            ax.fill_between(x, (mean - std).clip(lower=0.0), (mean + std).clip(upper=1.0), color=color, alpha=0.16)

    ax.set_xlabel("Explored points")
    ax.set_ylabel("Catastrophic error rate")
    ax.set_title(title)
    set_zoomed_ylim(ax, df)
    ax.grid(True, alpha=0.25, linewidth=0.8)
    ax.legend(fontsize=8)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    table_path = resolve_table_path(args.table_path, args.search_root)
    df = load_plot_data(table_path, args.use_table_error)
    output_dir = args.output_dir / table_path.parent.name if args.output_dir == FIGURE_DIR else args.output_dir
    output_path = output_dir / args.output_name
    plot_catastrophic_error(
        df,
        output_path,
        f"Experiment B catastrophic-error trend ({table_path.parent.name})",
        show_std=not args.disable_std,
    )
    print(f"Read table from {table_path}")
    if args.disable_std:
        print("Std bands disabled")
    elif table_path.with_name("experiment_b_summary.csv").exists() and not args.use_table_error:
        print(f"Used std bands from {table_path.with_name('experiment_b_summary.csv')}")
    print(f"Saved figure to {output_path}")


if __name__ == "__main__":
    main()
