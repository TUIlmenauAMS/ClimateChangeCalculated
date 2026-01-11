#!/usr/bin/env python3
"""
Climate Change assistant: Can you also give me the Python programs to make those plots?
Gerald Schuller, December 2025

python make_episode10_plots.py --csv electric-car-sales-share.csv --out fig --all
"""

"""
Episode 10 plot generator (OWID electric-car-sales share)

Input CSV expected columns:
- Entity
- Code
- Year
- Share of new cars that are electric

Generates:
A_scurve_cartoon.png
B_equation_card.png
C_shares_vs_counts_left.png
D_norway_fit.png
E_china_fit_2035.png
J_phase_k_vs_share.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


# -----------------------------
# Shared helpers
# -----------------------------
def load_owid_share(csv_path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    col = "Share of new cars that are electric"
    if col not in df.columns:
        raise ValueError(f"Expected column not found: {col}. Found: {list(df.columns)}")

    # Convert to fraction if in percent
    s = df[col].astype(float)
    if s.max(skipna=True) > 1.5:
        s = s / 100.0

    df = df.copy()
    df["share"] = s
    return df


def logistic(t: np.ndarray, L: float, k: float, t0: float) -> np.ndarray:
    return L / (1.0 + np.exp(-k * (t - t0)))


def fit_logistic(years: np.ndarray, shares: np.ndarray,
                 bounds=((0.5, 0.0, 1990.0), (1.05, 5.0, 2050.0)),
                 p0=(1.0, 0.4, 2025.0)) -> tuple[float, float, float]:
    params, _ = curve_fit(
        logistic,
        years,
        shares,
        p0=list(p0),
        bounds=bounds,
        maxfev=20000,
    )
    return float(params[0]), float(params[1]), float(params[2])


def bootstrap_band(years: np.ndarray, shares: np.ndarray, t_grid: np.ndarray,
                   bounds=((0.5, 0.0, 1990.0), (1.05, 5.0, 2050.0)),
                   p0=(1.0, 0.4, 2025.0),
                   n_boot: int = 500,
                   seed: int = 1) -> tuple[np.ndarray, np.ndarray, tuple[float, float, float]]:
    """
    Returns: lower68, upper68, point_params
    """
    L_hat, k_hat, t0_hat = fit_logistic(years, shares, bounds=bounds, p0=p0)
    point_params = (L_hat, k_hat, t0_hat)

    rng = np.random.default_rng(seed)
    curves = []

    for _ in range(n_boot):
        idx = rng.integers(0, len(years), len(years))
        yb = years[idx]
        sb = shares[idx]
        try:
            pb = fit_logistic(yb, sb, bounds=bounds, p0=point_params)
            curves.append(logistic(t_grid, *pb))
        except Exception:
            # Ignore failed fits
            pass

    if len(curves) < 10:
        raise RuntimeError("Too few successful bootstrap fits. Try reducing bounds or n_boot.")

    curves = np.array(curves)
    lower68 = np.percentile(curves, 16, axis=0)
    upper68 = np.percentile(curves, 84, axis=0)
    return lower68, upper68, point_params


def inv_logistic_year(p: float, L: float, k: float, t0: float) -> float:
    """
    Solve for t at logistic(t)=p.
    Note: only valid if 0 < p < L.
    """
    if not (0.0 < p < L):
        return float("nan")
    return float(t0 - (1.0 / k) * np.log(L / p - 1.0))


def savefig(fig, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", transparent=True)
    plt.close(fig)


# -----------------------------
# Plot A
# -----------------------------
def plot_A(out_dir: Path) -> Path:
    years = np.linspace(2010, 2040, 500)
    L, k, t0 = 1.0, 0.35, 2025.0
    s = logistic(years, L, k, t0)

    fig = plt.figure(figsize=(19.2, 10.8), dpi=100)  # 1920x1080
    ax = fig.add_subplot(111)

    ax.plot(years, s * 100)
    ax.set_xlim(2010, 2040)
    ax.set_ylim(0, 100)
    ax.set_xlabel("Year")
    ax.set_ylabel("Share of new sales (%)")
    ax.set_title("Technology adoption often follows an S-curve (logistic growth)")

    ax.axvline(t0, linestyle="--")
    ax.axhline(50, linestyle="--")
    ax.text(2012, 12, "Slow start", fontsize=16)
    ax.text(t0 + 0.5, 55, "Inflection\n(50%)", fontsize=16)
    ax.text(2032, 92, "Saturation", fontsize=16)

    # 20/80 markers
    t20 = inv_logistic_year(0.2, L, k, t0)
    t80 = inv_logistic_year(0.8, L, k, t0)
    ax.axvline(t20, linestyle=":")
    ax.axvline(t80, linestyle=":")
    ax.text(t20 + 0.2, 22, "20%", fontsize=14)
    ax.text(t80 + 0.2, 82, "80%", fontsize=14)

    out_path = out_dir / "A_scurve_cartoon.png"
    savefig(fig, out_path)
    return out_path


# -----------------------------
# Plot B
# -----------------------------
def plot_B(out_dir: Path) -> Path:
    fig = plt.figure(figsize=(16, 9), dpi=100)  # ~1600x900
    ax = fig.add_subplot(111)
    ax.axis("off")

    eq = r"$s(t)=\frac{L}{1+e^{-k\,(t-t_0)}}$"
    ax.text(0.5, 0.60, eq, ha="center", va="center", fontsize=48)

    ax.text(0.15, 0.30, r"$L$ = max share (asymptote)", fontsize=22, ha="left")
    ax.text(0.15, 0.22, r"$k$ = growth speed", fontsize=22, ha="left")
    ax.text(0.15, 0.14, r"$t_0$ = inflection year (50\%)", fontsize=22, ha="left")

    ax.text(0.60, 0.30, r"$L$ = Sättigungswert", fontsize=22, ha="left")
    ax.text(0.60, 0.22, r"$k$ = Wachstumsgeschwindigkeit", fontsize=22, ha="left")
    ax.text(0.60, 0.14, r"$t_0$ = Wendepunkt (50\%)", fontsize=22, ha="left")

    out_path = out_dir / "B_equation_card.png"
    savefig(fig, out_path)
    return out_path


# -----------------------------
# Plot C-left
# -----------------------------
def plot_C_left(out_dir: Path) -> Path:
    countries = ["Country A", "Country B"]
    ev_counts = [1_200_000, 120_000]  # illustrative absolute counts

    fig = plt.figure(figsize=(10, 10), dpi=120)
    ax = fig.add_subplot(111)

    ax.bar(countries, ev_counts)
    ax.set_title("Absolute EV registrations (counts can mislead)")
    ax.set_ylabel("EV registrations per year")
    ax.set_ylim(0, 1_300_000)

    for i, v in enumerate(ev_counts):
        ax.text(i, v * 1.01, f"{v/1e6:.2f} M", ha="center", va="bottom", fontsize=12)

    out_path = out_dir / "C_shares_vs_counts_left.png"
    savefig(fig, out_path)
    return out_path


# -----------------------------
# Plot D / E (single-country fits with bootstrap band)
# -----------------------------
def plot_country_fit(df: pd.DataFrame, entity: str, out_path: Path,
                     xlim=(2005, 2035),
                     bounds=((0.5, 0.0, 2000.0), (1.05, 5.0, 2040.0)),
                     p0=(1.0, 0.4, 2025.0),
                     n_boot=500,
                     seed=1) -> Path:
    g = df[df["Entity"] == entity].dropna(subset=["Year", "share"]).sort_values("Year")
    if g.empty:
        raise ValueError(f"No data for entity: {entity}")

    years = g["Year"].values.astype(float)
    shares = g["share"].values.astype(float)

    # Time grid fixed to match Norway scale
    t_grid = np.linspace(xlim[0], xlim[1], 500)

    lower68, upper68, (L_hat, k_hat, t0_hat) = bootstrap_band(
        years, shares, t_grid,
        bounds=bounds, p0=p0,
        n_boot=n_boot, seed=seed
    )

    fit_curve = logistic(t_grid, L_hat, k_hat, t0_hat)

    # Milestones from point estimate (may be outside xlim)
    t20 = inv_logistic_year(0.2, L_hat, k_hat, t0_hat)
    t50 = inv_logistic_year(0.5, L_hat, k_hat, t0_hat)
    t80 = inv_logistic_year(0.8, L_hat, k_hat, t0_hat)

    fig = plt.figure(figsize=(19.2, 10.8), dpi=100)
    ax = fig.add_subplot(111)

    ax.fill_between(t_grid, lower68 * 100, upper68 * 100, alpha=0.25,
                    label="68% uncertainty band (bootstrap)")
    ax.plot(t_grid, fit_curve * 100, linewidth=3, label="Logistic fit (median)")
    ax.scatter(years, shares * 100, s=90, zorder=3, label="Observed data (OWID)")

    for t, lab in [(t20, "20%"), (t50, "50% (t₀)"), (t80, "80%")]:
        if np.isfinite(t) and xlim[0] <= t <= xlim[1]:
            ax.axvline(t, linestyle="--", linewidth=2)
            ax.text(t + 0.15, 4, lab, rotation=90, fontsize=14, va="bottom")

    ax.set_xlim(*xlim)
    ax.set_ylim(0, 100)
    ax.set_xlabel("Year")
    ax.set_ylabel("EV share of new car sales (%)")
    ax.set_title(f"{entity} — EV share of new car sales\n"
                 f"Logistic fit with 68% bootstrap uncertainty (OWID)\n"
                 f"Time scale {xlim[0]}–{xlim[1]}")

    summary = (
        f"Point estimate\n"
        f"L = {L_hat:.2f}\n"
        f"k = {k_hat:.2f} /year\n"
        f"t₀ (50%) = {t50:.1f}\n"
        f"80% ≈ {t80:.1f}"
    )
    ax.text(xlim[0] + 0.5, 86, summary, fontsize=14,
            bbox=dict(boxstyle="round,pad=0.4", alpha=0.2))

    ax.legend(loc="lower right")

    savefig(fig, out_path)
    return out_path


def plot_D_norway(df: pd.DataFrame, out_dir: Path) -> Path:
    return plot_country_fit(
        df, "Norway", out_dir / "D_norway_fit.png",
        xlim=(2005, 2035),
        p0=(1.0, 0.5, 2018.0),
        seed=42
    )


def plot_E_china_2035(df: pd.DataFrame, out_dir: Path) -> Path:
    return plot_country_fit(
        df, "China", out_dir / "E_china_fit_2035.png",
        xlim=(2005, 2035),
        p0=(1.0, 0.4, 2025.0),
        seed=1
    )


# -----------------------------
# Plot J: Phase diagram
# -----------------------------
def plot_J_phase(df: pd.DataFrame, out_dir: Path, min_points: int = 8) -> Path:
    bounds = ((0.5, 0.0, 1990.0), (1.05, 5.0, 2050.0))
    rows = []

    for ent, g in df.groupby("Entity"):
        g = g.dropna(subset=["Year", "share"]).sort_values("Year")
        if len(g) < min_points:
            continue

        years = g["Year"].values.astype(float)
        shares = g["share"].values.astype(float)

        p0 = (1.0, 0.4, float(np.median(years)))
        try:
            L_hat, k_hat, t0_hat = fit_logistic(years, shares, bounds=bounds, p0=p0)
            rows.append({
                "Entity": ent,
                "LatestYear": int(years[-1]),
                "LatestShare": float(shares[-1]),
                "L": L_hat, "k": k_hat, "t0": t0_hat,
                "N": len(g),
            })
        except Exception:
            pass

    res = pd.DataFrame(rows)
    res = res[(res["k"] > 0) & (res["k"] < 5) & (res["LatestShare"].between(0, 1.05))].copy()

    # Labels (only if present)
    label_candidates = [
        "Norway", "China", "United States", "Germany", "United Kingdom", "France",
        "Netherlands", "Sweden", "Denmark", "India", "Japan", "Australia", "Canada",
        "Brazil", "Iceland"
    ]
    labels = [c for c in label_candidates if c in set(res["Entity"])]

    fig = plt.figure(figsize=(19.2, 10.8), dpi=100)
    ax = fig.add_subplot(111)

    x = res["LatestShare"] * 100
    y = res["k"]
    ax.scatter(x, y, s=80, alpha=0.85)

    ax.axvline(20, linestyle="--", linewidth=2)
    k_med = float(np.median(y))
    ax.axhline(k_med, linestyle="--", linewidth=2)

    ax.set_xlabel("Latest EV share of new car sales (%)")
    ax.set_ylabel("Growth rate k (per year) from logistic fit")
    ax.set_title("Phase diagram: adoption level vs growth speed (OWID EV sales share)")

    for ent in labels:
        r = res[res["Entity"] == ent].iloc[0]
        ax.text(r["LatestShare"] * 100 + 0.8, r["k"] + 0.02, ent, fontsize=14)

    ax.text(1, k_med + 0.05, "faster-than-median k", fontsize=12, va="bottom")
    ax.text(1, k_med - 0.05, "slower-than-median k", fontsize=12, va="top")
    ax.text(20.5, ax.get_ylim()[1] * 0.95, "20% threshold", fontsize=12, rotation=90, va="top")

    out_path = out_dir / "J_phase_k_vs_share.png"
    savefig(fig, out_path)
    return out_path


# -----------------------------
# CLI
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="electric-car-sales-share.csv", help="Path to OWID CSV")
    ap.add_argument("--out", default="fig", help="Output directory for PNGs")
    ap.add_argument("--all", action="store_true", help="Generate all plots")
    ap.add_argument("--A", action="store_true")
    ap.add_argument("--B", action="store_true")
    ap.add_argument("--Cleft", action="store_true")
    ap.add_argument("--D", action="store_true")
    ap.add_argument("--E", action="store_true")
    ap.add_argument("--J", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out)

    if args.A or args.B or args.Cleft or args.D or args.E or args.J or args.all:
        pass
    else:
        args.all = True  # default behavior

    if args.A or args.all:
        print(plot_A(out_dir))
    if args.B or args.all:
        print(plot_B(out_dir))
    if args.Cleft or args.all:
        print(plot_C_left(out_dir))

    df = load_owid_share(args.csv)

    if args.D or args.all:
        print(plot_D_norway(df, out_dir))
    if args.E or args.all:
        print(plot_E_china_2035(df, out_dir))
    if args.J or args.all:
        print(plot_J_phase(df, out_dir))


if __name__ == "__main__":
    main()

