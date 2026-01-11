#!/usr/bin/env python3
from __future__ import annotations
"""
pip3 install numpy pandas matplotlib scipy
python3 make_L_policy_overlay_germany_top5_vertical.py \
  --ev_csv electric-car-sales-share.csv \
  --policy_csv policy_events.csv \
  --out L_policy_overlay_germany_top5_vertical.png

"""
"""
Reproduce: L_policy_overlay_germany_top5_vertical.png

Inputs:
- electric-car-sales-share.csv (OWID)
- policy_events.csv (your table)

Output:
- L_policy_overlay_germany_top5_vertical.png
- (optional) L_policy_overlay_germany_top5_selected.csv

Notes:
- Fits a logistic S-curve to Germany's EV share of new sales (in %).
- Draws faint vertical lines for all policy years (Germany).
- Labels only the TOP 5 policies, vertically staggered (no horizontal overlap).
"""


import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


# -----------------------------
# Helpers
# -----------------------------
def logistic(t: np.ndarray, L: float, k: float, t0: float) -> np.ndarray:
    return L / (1.0 + np.exp(-k * (t - t0)))


def load_ev_data(ev_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(ev_csv)
    col = "Share of new cars that are electric"
    if col not in df.columns:
        raise ValueError(f"Missing column '{col}'. Found: {list(df.columns)}")

    s = df[col].astype(float)
    # Convert % -> fraction if needed
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
    if g.empty:
        raise ValueError(f"No EV share data found for Entity='{entity}'")

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
        raise ValueError(f"No policy events found for countries={country_names}")

    # Ensure Year numeric
    pol["Year"] = pol["Year"].astype(float)
    return pol


def score_policy_title(title: str) -> int:
    """
    Transparent heuristic scoring: prioritizes subsidies/bonuses/targets/mandates etc.
    Adjust keywords/weights if you want different "top 5" selection.
    """
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


# -----------------------------
# Main plot
# -----------------------------
def make_plot(
    ev_csv: Path,
    policy_csv: Path,
    out_path: Path,
    top_n: int = 5,
    x_min: float = 2005,
    x_max: float = 2035,
) -> None:
    ev = load_ev_data(ev_csv)
    years, shares = get_country_series(ev, "Germany")

    # Policy table: accept common naming variants
    pol = load_policy_events(
        policy_csv,
        country_names=["Germany", "Deutschland", "Federal Republic of Germany", "DE"],
    )

    # Importance scoring
    pol["importance"] = pol["Policy_Title"].apply(score_policy_title)

    if "Status" in pol.columns:
        pol["importance"] += (pol["Status"].astype(str).str.lower() == "implemented").astype(int)

    pol_sorted = pol.sort_values(["importance", "Year"], ascending=[False, True]).reset_index(drop=True)
    top = pol_sorted.head(top_n).sort_values("Year").reset_index(drop=True)

    # Logistic fit (percent scale)
    p0 = [100.0, 0.35, float(np.median(years))]
    params, _ = curve_fit(logistic, years, shares, p0=p0, maxfev=30000)

    t_grid = np.linspace(x_min, x_max, 400)
    fit = logistic(t_grid, *params)

    # Plot: 1920x1080 slide-friendly
    fig = plt.figure(figsize=(19.2, 10.8), dpi=100)
    ax = fig.add_subplot(111)

    ax.plot(t_grid, fit, linewidth=3, label="S-curve fit")
    ax.scatter(years, shares, s=90, zorder=3, label="Observed EV share")

    # Faint policy-year lines for all events
    for y in sorted(pol["Year"].unique()):
        ax.axvline(float(y), linestyle=":", linewidth=1.2, alpha=0.3)

    # Vertically staggered labels (no x-jitter)
    y_positions = np.linspace(22, 86, len(top))
    for (_, r), ylab in zip(top.iterrows(), y_positions):
        ax.annotate(
            r["Policy_Title"],
            xy=(float(r["Year"]), 4),
            xytext=(float(r["Year"]) + 0.25, float(ylab)),
            fontsize=13,
            ha="left",
            va="center",
            color="green",
            arrowprops=dict(arrowstyle="-", linewidth=1.4, color="green"),
        )

    n_other = max(0, len(pol_sorted) - len(top))
    ax.text(
        x_min + 0.5,
        92,
        f"Top {len(top)} policies labeled\n(+{n_other} other events as faint lines)",
        fontsize=13,
        bbox=dict(boxstyle="round,pad=0.4", alpha=0.18),
    )

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(0, 100)
    ax.set_xlabel("Year")
    ax.set_ylabel("EV share of new car sales (%)")
    ax.set_title("Policy events vs EV adoption (Germany)")

    ax.legend(loc="lower right", framealpha=0.9)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", transparent=True)
    plt.close(fig)

    # Optional: save the selected “top” policies for transparency
    selected_csv = out_path.with_suffix("").with_name("L_policy_overlay_germany_top5_selected.csv")
    keep_cols = [c for c in ["Country", "Year", "Policy_Title", "Status", "Source", "importance"] if c in top.columns]
    top[keep_cols].to_csv(selected_csv, index=False)
    print(f"Saved: {out_path}")
    print(f"Saved: {selected_csv}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ev_csv", default="electric-car-sales-share.csv", help="OWID EV share CSV")
    ap.add_argument("--policy_csv", default="policy_events.csv", help="Policy events CSV")
    ap.add_argument("--out", default="L_policy_overlay_germany_top5_vertical.png", help="Output PNG path")
    ap.add_argument("--top_n", type=int, default=5, help="How many policies to label")
    args = ap.parse_args()

    make_plot(
        ev_csv=Path(args.ev_csv),
        policy_csv=Path(args.policy_csv),
        out_path=Path(args.out),
        top_n=args.top_n,
    )


if __name__ == "__main__":
    main()

