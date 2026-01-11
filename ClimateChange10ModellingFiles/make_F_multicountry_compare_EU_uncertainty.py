#!/usr/bin/env python3
"""
Climate Change Assistant: Can you give me the Python program to make this plot?
Gerald Schuller, December 2025

python3 make_F_multicountry_compare_EU_uncertainty.py electric-car-sales-share.csv

"""
"""
Make Figure F: multi-country EV adoption comparison INCLUDING EU-27,
with logistic fits + 68% bootstrap uncertainty bands.

Input (OWID CSV): electric-car-sales-share.csv
Expected columns:
- Entity
- Code
- Year
- Share of new cars that are electric

Output:
- F_multicountry_compare_EU_uncertainty.png
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


# -----------------------------
# Model
# -----------------------------
def logistic(t: np.ndarray, L: float, k: float, t0: float) -> np.ndarray:
    return L / (1.0 + np.exp(-k * (t - t0)))


# Bounds used in the episode (conservative)
BOUNDS = ([0.5, 0.0, 1990.0], [1.05, 5.0, 2050.0])


def fit_params(years: np.ndarray, shares: np.ndarray, p0: list[float]) -> np.ndarray:
    params, _ = curve_fit(
        logistic, years, shares,
        p0=p0, bounds=BOUNDS,
        maxfev=20000
    )
    return params


def bootstrap_band(
    years: np.ndarray,
    shares: np.ndarray,
    t_grid: np.ndarray,
    p0: list[float],
    n_boot: int = 500,
    seed: int = 0
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray]:
    """
    Returns (lo68, hi68, p_hat). Bands are 16–84 percentiles.
    If too few fits succeed, returns (None, None, p_hat).
    """
    p_hat = fit_params(years, shares, p0=p0)

    rng = np.random.default_rng(seed)
    curves = []

    for _ in range(n_boot):
        idx = rng.integers(0, len(years), len(years))
        yb = years[idx]
        sb = shares[idx]
        try:
            pb = fit_params(yb, sb, p0=list(p_hat))
            curves.append(logistic(t_grid, *pb))
        except Exception:
            pass

    if len(curves) < 20:
        return None, None, p_hat

    curves = np.array(curves)
    lo68 = np.percentile(curves, 16, axis=0)
    hi68 = np.percentile(curves, 84, axis=0)
    return lo68, hi68, p_hat


# -----------------------------
# Data loading
# -----------------------------
def load_owid(csv_path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    col = "Share of new cars that are electric"
    if col not in df.columns:
        raise ValueError(f"Missing column '{col}'. Found columns: {list(df.columns)}")

    # Convert % -> fraction if needed
    s = df[col].astype(float)
    if s.max(skipna=True) > 1.5:
        s = s / 100.0

    df = df.copy()
    df["share"] = s
    return df


def get_series(df: pd.DataFrame, entity: str) -> tuple[np.ndarray, np.ndarray]:
    g = df[df["Entity"] == entity].dropna(subset=["Year", "share"]).sort_values("Year")
    if g.empty:
        raise ValueError(f"No data for entity: {entity}")
    years = g["Year"].values.astype(float)
    shares = g["share"].values.astype(float)
    return years, shares


# -----------------------------
# Main plot
# -----------------------------
def make_plot(
    csv_path: str | Path,
    out_path: str | Path = "F_multicountry_compare_EU_uncertainty.png",
    x_min: float = 2005,
    x_max: float = 2035
) -> Path:
    df = load_owid(csv_path)

    # Countries to plot (adjust as desired)
    preferred = [
        "Norway",
        "European Union (27)",
        "China",
        "Germany",
        "United States",
        "India",
    ]
    countries = [c for c in preferred if c in set(df["Entity"])]
    if not countries:
        raise ValueError("None of the preferred entities were found in the CSV.")

    t_grid = np.linspace(x_min, x_max, 500)

    fig = plt.figure(figsize=(19.2, 10.8), dpi=100)  # 1920x1080
    ax = fig.add_subplot(111)

    for i, c in enumerate(countries):
        years, shares = get_series(df, c)
        if len(years) < 8:
            # too few points for a stable bootstrap
            continue

        p0 = [1.0, 0.4, float(np.median(years))]
        lo, hi, p_hat = bootstrap_band(
            years, shares, t_grid,
            p0=p0, n_boot=500, seed=10 + i
        )

        line, = ax.plot(t_grid, logistic(t_grid, *p_hat) * 100,
                        linewidth=2.8, label=c)
        ax.scatter(years, shares * 100, s=55, alpha=0.9)

        if lo is not None:
            ax.fill_between(t_grid, lo * 100, hi * 100,
                            alpha=0.15, color=line.get_color())

        # label near last observed point
        ax.text(years[-1] + 0.2, shares[-1] * 100,
                c, fontsize=13, va="center")

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(0, 100)
    ax.set_xlabel("Year")
    ax.set_ylabel("EV share of new car sales (%)")
    ax.set_title(
        "EV adoption S-curves including the European Union (OWID)\n"
        "Observed data + logistic fits with 68% bootstrap uncertainty"
    )

    ax.axhline(50, linestyle="--", linewidth=1.8)
    ax.legend(loc="upper left", ncols=2, framealpha=0.8)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", transparent=True)
    plt.close(fig)
    return out_path


if __name__ == "__main__":
    # Example usage:
    # python make_F_multicountry_compare_EU_uncertainty.py electric-car-sales-share.csv
    import sys
    csv = sys.argv[1] if len(sys.argv) > 1 else "electric-car-sales-share.csv"
    out = make_plot(csv)
    print(f"Saved: {out}")

