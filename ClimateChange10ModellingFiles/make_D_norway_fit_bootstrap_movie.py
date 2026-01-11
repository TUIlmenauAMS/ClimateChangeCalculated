import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from scipy.optimize import curve_fit

FADE_FRAMES = 15   # number of frames used for fade-in
MAX_ALPHA = 0.5   # final opacity of each bootstrap curve


# ---------------- Load data ----------------
ev = pd.read_csv("electric-car-sales-share.csv")
col = "Share of new cars that are electric"

s = ev[col].astype(float)
if s.max(skipna=True) > 1.5:
    ev["share"] = s / 100.0
else:
    ev["share"] = s

g = ev[ev["Entity"] == "Norway"].dropna(subset=["Year", "share"]).sort_values("Year")
years = g["Year"].values.astype(float)
shares = g["share"].values.astype(float)

# ---------------- Logistic model ----------------
def logistic(t, L, k, t0):
    return L / (1.0 + np.exp(-k * (t - t0)))

bounds = ([0.5, 0.0, 1990.0], [1.05, 5.0, 2050.0])
p0 = [1.0, 0.5, float(np.median(years))]

params, _ = curve_fit(
    logistic, years, shares, p0=p0, bounds=bounds, maxfev=30000
)

t_grid = np.linspace(2005, 2035, 400)
median_curve = logistic(t_grid, *params)

# ---------------- Bootstrap curves ----------------
rng = np.random.default_rng(0)
N_BOOT = 100
curves = []

for _ in range(N_BOOT):
    idx = rng.integers(0, len(years), len(years))
    try:
        pb, _ = curve_fit(
            logistic,
            years[idx],
            shares[idx],
            p0=list(params),
            bounds=bounds,
            maxfev=20000,
        )
        curves.append(logistic(t_grid, *pb))
    except RuntimeError:
        pass

curves = np.array(curves)

lo = np.percentile(curves, 16, axis=0)
hi = np.percentile(curves, 84, axis=0)

# ---------------- Animation ----------------
fig, ax = plt.subplots(figsize=(10, 5))

ax.set_xlim(2005, 2035)
ax.set_ylim(0, 1.05)
ax.set_xlabel("Year")
ax.set_ylabel("EV share of new car sales")
ax.set_title("Norway EV adoption: bootstrap S-curves")

ax.fill_between(t_grid, lo, hi, alpha=0.25, label="68% uncertainty band")
ax.plot(t_grid, median_curve, linewidth=3, label="Median S-curve")
ax.scatter(years, shares, s=35, zorder=3, label="Observed data")

line, = ax.plot([], [], lw=1.5, alpha=0.5)
ax.legend(loc="lower right")

def init():
    line.set_data([], [])
    return (line,)

"""
def update(i):
    line.set_data(t_grid, curves[i])
    return (line,)
"""
    
def update(i):
    y = curves[i]
    line.set_data(t_grid, y)

    # Fade-in logic
    fade_progress = min(1.0, i / FADE_FRAMES)
    alpha = fade_progress * MAX_ALPHA
    line.set_alpha(alpha)

    return (line,)


anim = FuncAnimation(
    fig,
    update,
    frames=len(curves),
    init_func=init,
    interval=100,
    blit=True,
)

# ---- Choose ONE output format ----

# GIF (recommended, simplest)
anim.save("D_norway_fit_bootstrap_movie.gif", writer=PillowWriter(fps=8)) #8

# OR MP4 (requires ffmpeg installed)
# anim.save("D_norway_fit_bootstrap.mp4", fps=20)

plt.show()

