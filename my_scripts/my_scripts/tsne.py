#!/usr/bin/env python3
"""Create five t-SNE visualizations of the 95 ActMOF experiments.

The five synthesis variables are embedded into two dimensions once.  The same
embedding is then colored by intensity, FWHM, q, absolute prediction error,
and log absolute prediction error.
"""

from __future__ import annotations

import argparse
import ast
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler


FEATURES = [
    "metal_amount",
    "modulator",
    "add_solvent",
    "reaction_time",
    "reaction_temperature",
]
EXPECTED_RAW_ROWS = 96
EXPECTED_UNIQUE_ROWS = 95
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_SOURCE = PROJECT_ROOT / "Synthesis App" / "student_bo_app_v109.py"
DEFAULT_ERROR_DATA = PROJECT_ROOT / "Error_data.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "figures" / "tsne"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE,
                        help="Python file containing the REFERENCE_* arrays")
    parser.add_argument("--error-data", type=Path, default=DEFAULT_ERROR_DATA,
                        help="CSV containing y_abs_error and y_log_abs_error")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                        help="directory in which the five PNG files are written")
    parser.add_argument("--seed", type=int, default=42, help="t-SNE random seed")
    parser.add_argument("--perplexity", type=float, default=30.0,
                        help="t-SNE perplexity (must be smaller than 95)")
    parser.add_argument("--dpi", type=int, default=300, help="output image DPI")
    return parser.parse_args()


def extract_array(source: Path, variable: str) -> np.ndarray:
    """Read the literal passed to np.array without importing the GUI app."""
    tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    for node in tree.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        if not any(isinstance(target, ast.Name) and target.id == variable
                   for target in targets):
            continue
        value = node.value
        if not isinstance(value, ast.Call) or not value.args:
            break
        return np.asarray(ast.literal_eval(value.args[0]), dtype=float)
    raise ValueError(f"Could not find a literal {variable} array in {source}")


def calculate_q(intensity: pd.Series, fwhm: pd.Series) -> np.ndarray:
    """Match the q calculation used by student_bo_app_v109.py."""
    valid = (intensity > 0) & (fwhm > 0) & (fwhm != 30)
    q = np.zeros(len(intensity), dtype=int)
    q[valid] = np.rint(
        intensity.loc[valid].to_numpy() / fwhm.loc[valid].to_numpy()
    ).astype(int)
    return q


def load_data(source: Path, error_data: Path) -> pd.DataFrame:
    x = extract_array(source, "REFERENCE_X")
    intensity = extract_array(source, "REFERENCE_INTENSITY")
    fwhm = extract_array(source, "REFERENCE_FWHM")
    if not (len(x) == len(intensity) == len(fwhm) == EXPECTED_RAW_ROWS):
        raise ValueError(
            "Expected 96 source rows, but found "
            f"X={len(x)}, intensity={len(intensity)}, FWHM={len(fwhm)}"
        )

    raw = pd.DataFrame(x, columns=FEATURES)
    raw["Intensity"] = intensity
    raw["FWHM"] = fwhm
    duplicate_count = int(raw.duplicated(subset=FEATURES).sum())
    data = raw.drop_duplicates(subset=FEATURES, keep="first").copy()
    if duplicate_count != 1 or len(data) != EXPECTED_UNIQUE_ROWS:
        raise ValueError(
            f"Expected one repeated condition and 95 unique points; found "
            f"{duplicate_count} repeated rows and {len(data)} unique points"
        )
    data["q_value"] = calculate_q(data["Intensity"], data["FWHM"])

    errors = pd.read_csv(error_data)
    required = FEATURES + ["y_abs_error", "y_log_abs_error"]
    missing = [column for column in required if column not in errors.columns]
    if missing:
        raise ValueError(f"Missing columns in {error_data}: {', '.join(missing)}")
    errors = errors[required]
    if errors.duplicated(subset=FEATURES).any():
        raise ValueError(f"{error_data} contains duplicate synthesis conditions")

    data = data.merge(errors, on=FEATURES, how="left", validate="one_to_one")
    if data[["y_abs_error", "y_log_abs_error"]].isna().any().any():
        raise ValueError("Some source experiments have no matching error-data row")
    if len(data) != EXPECTED_UNIQUE_ROWS:
        raise ValueError(f"Expected 95 merged data points, found {len(data)}")
    return data.reset_index(drop=True)


def save_plot(embedding: np.ndarray, values: pd.Series, title: str,
              colorbar_label: str, cmap: str, output: Path, dpi: int) -> None:
    fig, ax = plt.subplots(figsize=(8, 6.5), constrained_layout=True)
    points = ax.scatter(
        embedding[:, 0], embedding[:, 1], c=values, cmap=cmap,
        s=62, alpha=0.9, edgecolors="white", linewidths=0.45,
    )
    colorbar = fig.colorbar(points, ax=ax, pad=0.02)
    colorbar.set_label(colorbar_label)
    ax.set(title=title, xlabel="t-SNE dimension 1", ylabel="t-SNE dimension 2")
    ax.grid(alpha=0.18, linewidth=0.6)
    fig.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if not 0 < args.perplexity < EXPECTED_UNIQUE_ROWS:
        raise ValueError("--perplexity must be greater than 0 and smaller than 95")

    data = load_data(args.source.resolve(), args.error_data.resolve())
    scaled_x = StandardScaler().fit_transform(data[FEATURES].to_numpy(dtype=float))
    embedding = TSNE(
        n_components=2,
        perplexity=args.perplexity,
        init="pca",
        learning_rate="auto",
        random_state=args.seed,
    ).fit_transform(scaled_x)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    plots = [
        ("Intensity", "t-SNE colored by Intensity", "Intensity", "viridis", "tsne_intensity.png"),
        ("FWHM", "t-SNE colored by FWHM", "FWHM", "plasma", "tsne_fwhm.png"),
        ("q_value", "t-SNE colored by q-value", "q-value", "turbo", "tsne_q_value.png"),
        ("y_abs_error", "t-SNE colored by absolute error", "y_abs_error", "magma", "tsne_y_abs_error.png"),
        ("y_log_abs_error", "t-SNE colored by log absolute error", "y_log_abs_error", "cividis", "tsne_y_log_abs_error.png"),
    ]
    for column, title, label, cmap, filename in plots:
        save_plot(embedding, data[column], title, label, cmap,
                  output_dir / filename, args.dpi)

    print(f"Created {len(plots)} t-SNE images from {len(data)} unique data points in {output_dir}")


if __name__ == "__main__":
    main()
