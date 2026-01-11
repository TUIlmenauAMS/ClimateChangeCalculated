#!/usr/bin/env python3
from __future__ import annotations
"""
pip3 install numpy pandas matplotlib scipy
python3 make_L_policy_overlay_germany_top5_vertical_bootstrap.py \
  --ev_csv electric-car-sales-share.csv \
  --policy_csv policy_events.csv \
  --top_supportive 5 \
  --n_boot 800 \
  --seed 0
"""

"""
Germany policy overlay + logistic S-curve fit with 68% bootstrap uncertainty band.

What it does:
- Loads OWID EV sales share data (Germany) from electric-car-sales-share.csv
- Fits logistic curve: share(t) = L / (1 + exp(-k*(t-t0)))
- Computes 68% uncertainty band via bootstrap resampling (16–84 percentiles)
- Plots:
  - Observed EV share (dots)
  - Median fit (line)
  - 68% band (shaded)
  - All policy years as faint vertical lines
  - Labels: top-N supportive policies + ALWAYS include harmful policies (Option 2)

Outputs:
- L_policy_overlay_germany_top5_vertical_bootstrap.png
- L_policy_overlay_germany_top5_selected_bootstrap.csv
"""


import argparse
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


# bounds (same spirit as earlier work, but on percent scale: L in [50..105])
BOUNDS_PERCENT = ([50.0, 0.0, 1990.0], [105.0, 5.0, 2050.0])


def fit_logistic_percent(years: np.ndarray, shares_percent: np.ndarray, p0: list[float]) -> np.ndarray:
    params, _ = curve_fit(
        logistic,
        years,
        shares_percent,
        p0=p0,
        bounds=BOUNDS_PERCENT,
        maxfev=40000,
    )
    return params


def bootstrap_band(
    years: np.ndarray,
    shares_percent: np.ndarray,
    t_grid: np.ndarray,
    n_boot: int = 800,
    seed: int = 0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
      median_curve, lo68_curve, hi68_curve for logistic fits on t_grid
    computed via bootstrap resampling of the (year, share) pairs.
    """
    rng = np.random.default_rng(seed)

    # initial fit (also serves as warm-start)
    p0 = [100.0, 0.30, float(np.median(years))]
    p_hat = fit_logistic_percent(years, shares_percent, p0=p0)

    curves = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(years), len(years))
        yb = years[idx]
        sb = shares_percent[idx]
        try:
            pb = fit_logistic_percent(yb, sb, p0=list(p_hat))
            curves.append(logistic(t_grid, *pb))
        except Exception:
            pass

    if len(curves) < 30:
        # Fallback: no reliable band; return just point estimate as all three
        med = logistic(t_grid, *p_hat)
        return med, med, med

    curves = np.asarray(curves)
    lo = np.percentile(curves, 16, axis=0)
    med = np.percentile(curves, 50, axis=0)
    hi = np.percentile(curves, 84, axis=0)
    return med, lo, hi


# -----------------------------
# Data loading
# -----------------------------
def load_ev_data(ev_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(ev_csv)
    col = "Share of new cars that are electric"
    if col not in df.columns:
        raise ValueError(f"Missing column '{col}' in {ev_csv}. Found: {list(df.columns)}")

    s = df[col].astype(float)
    # Convert % to fraction if needed
    if s.max(skipna=True) > 1.5:
        s = s / 100.0
    df = df.copy()
    df["share"] = s
    return df


def get_country_series(df: pd.DataFrame, entity: str) -> tuple[np.ndarray, np.ndarray]:
    g = (
        df[df["Entity"] == entity]
        .dropna(subset=["Year", "share"])
        .sort_values("Year")
        .copy()
    )
    if len(g) < 6:
        raise ValueError(f"Not enough data points for {entity} (need >= 6, got {len(g)})")
    years = g["Year"].astype(float).to_numpy()
    shares_percent = (g["share"].astype(float).to_numpy()) * 100.0
    return years, shares_percent


def load_policy_events(policy_csv: Path, country_names: list[str]) -> pd.DataFrame:
    pol = pd.read_csv(policy_csv)
    required = {"Country", "Year", "Policy_Title"}
    missing = required - set(pol.columns)
    if missing:
        raise ValueError(f"policy_events.csv missing columns: {missing}. Found: {list(pol.columns)}")

    pol = pol[pol["Country"].isin(country_names)].copy()
    if pol.empty:
        raise ValueError(f"No policy events found for {country_names}")

    pol["Year"] = pol["Year"].astype(float)
    return pol


# -----------------------------
# Policy selection (Option 2)
# -----------------------------
def score_policy_title(title: str) -> int:
    KEYWORDS = {
        "bonus": 6,
        "umweltbonus": 7,
        "subsid": 5,
        "incent": 4,
        "tax": 4,
        "credit": 4,
        "rebate": 4,
        "mandate": 4,
        "target": 4,
        "ban": 5,
        "co2": 3,
        "fleet": 3,
        "charging": 3,
        "infrastructure": 3,
    }
    t = str(title).lower()
    return sum(w for k, w in KEYWORDS.items() if k in t)


def select_policies_option2(pol: pd.DataFrame, top_n_supportive: int) -> pd.DataFrame:
    """
    Option 2:
      - ALWAYS include harmful policies (keyword-based)
      - plus top N supportive by importance
    """
    # Identify harmful / rollback policies by title keywords (tune as needed)
    harmful = pol[
        pol["Policy_Title"].str.contains(
            r"rebate reduction|bonus cut|subsidy cut|cut|phase[- ]out|abolish|expire|end of",
            case=False,
            na=False,
            regex=True,
        )
    ].copy()

    # Score all policies
    pol = pol.copy()
    pol["importance"] = pol["Policy_Title"].apply(score_policy_title)
    if "Status" in pol.columns:
        pol["importance"] += (pol["Status"].astype(str).str.lower() == "implemented").astype(int)

    # Supportive candidates = everything not already in harmful set (by Year+Title)
    harmful_keys = set(zip(harmful["Year"], harmful["Policy_Title"]))
    supportive_candidates = pol[
        ~pol.apply(lambda r: (r["Year"], r["Policy_Title"]) in harmful_keys, axis=1)
    ].copy()

    supportive_top = (
        supportive_candidates
        .sort_values(["importance", "Year"], ascending=[False, True])
        .head(top_n_supportive)
        .copy()
    )

    chosen = (
        pd.concat([supportive_top, harmful], ignore_index=True)
        .drop_duplicates(subset=["Year", "Policy_Title"])
        .sort_values("Year")
        .reset_index(drop=True)
    )
    return chosen


# -----------------------------
# Plot
# -----------------------------
def make_plot(
    ev_csv: Path,
    policy_csv: Path,
    out_png: Path,
    out_selected_csv: Path,
    top_n_supportive: int = 5,
    n_boot: int = 800,
    seed: int = 0,
    x_min: float = 2005,
    x_max: float = 2035,
) -> None:
    ev = load_ev_data(ev_csv)
    years, shares_percent = get_country_series(ev, "Germany")

    pol = load_policy_events(
        policy_csv,
        country_names=["Germany", "Deutschland", "Federal Republic of Germany", "DE"],
    )

    chosen = select_policies_option2(pol, top_n_supportive=top_n_supportive)

    # Fit + band
    t_grid = np.linspace(x_min, x_max, 450)
    med, lo, hi = bootstrap_band(years, shares_percent, t_grid, n_boot=n_boot, seed=seed)

    # Plot: slide-friendly 1920x1080
    fig = plt.figure(figsize=(19.2, 10.8), dpi=100)
    ax = fig.add_subplot(111)

    # Uncertainty band + median fit
    ax.fill_between(t_grid, lo, hi, alpha=0.18, label="68% uncertainty (bootstrap)")
    ax.plot(t_grid, med, linewidth=3, label="S-curve fit (median)")

    # Observations
    ax.scatter(years, shares_percent, s=90, zorder=3, label="Observed EV share")

    # Faint vertical lines for all Germany policy years
    for y in sorted(pol["Year"].unique()):
        ax.axvline(float(y), linestyle=":", linewidth=1.2, alpha=0.3)

    # Label chosen policies (vertically staggered, no x-jitter)
    y_positions = np.linspace(22, 86, len(chosen))
    for (_, r), ylab in zip(chosen.iterrows(), y_positions):
        title = str(r["Policy_Title"])
        # color harmful-looking titles red (same keyword rule)
        is_harmful = bool(
            pd.Series([title]).str.contains(
                r"rebate reduction|bonus cut|subsidy cut|cut|phase[- ]out|abolish|expire|end of",
                case=False,
                na=False,
                regex=True,
            ).iloc[0]
        )
        color = "red" if is_harmful else "green"

        ax.annotate(
            title,
            xy=(float(r["Year"]), 4),
            xytext=(float(r["Year"]) + 0.25, float(ylab)),
            fontsize=13,
            ha="left",
            va="center",
            color=color,
            arrowprops=dict(arrowstyle="-", linewidth=1.4, color=color),
        )

    ax.text(
        x_min + 0.5, 92,
        "Green = supportive policies (top-N)\nRed = harmful / rollback policies (always included)",
        fontsize=13,
        bbox=dict(boxstyle="round,pad=0.4", alpha=0.18),
    )

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(0, 100)
    ax.set_xlabel("Year")
    ax.set_ylabel("EV share of new car sales (%)")
    ax.set_title("Policy events vs EV adoption (Germany)\nS-curve fit with 68% bootstrap uncertainty")

    ax.legend(loc="lower right", framealpha=0.9)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, bbox_inches="tight", transparent=True)
    plt.close(fig)

    # Save selection for transparency
    keep_cols = [c for c in ["Country", "Year", "Policy_Title", "Status"] if c in chosen.columns]
    if "importance" in chosen.columns:
        keep_cols.append("importance")
    chosen[keep_cols].to_csv(out_selected_csv, index=False)

    print(f"Saved: {out_png}")
    print(f"Saved: {out_selected_csv}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ev_csv", default="electric-car-sales-share.csv")
    ap.add_argument("--policy_csv", default="policy_events.csv")
    ap.add_argument("--out_png", default="L_policy_overlay_germany_top5_vertical_bootstrap.png")
    ap.add_argument("--out_selected_csv", default="L_policy_overlay_germany_top5_selected_bootstrap.csv")
    ap.add_argument("--top_supportive", type=int, default=5)
    ap.add_argument("--n_boot", type=int, default=800)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    make_plot(
        ev_csv=Path(args.ev_csv),
        policy_csv=Path(args.policy_csv),
        out_png=Path(args.out_png),
        out_selected_csv=Path(args.out_selected_csv),
        top_n_supportive=args.top_supportive,
        n_boot=args.n_boot,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()

