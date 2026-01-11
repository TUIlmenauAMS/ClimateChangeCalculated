#!/usr/bin/env python3
"""
Climate Change Assistant: Can you now make a python program, which 
generates a map of the shares of EV sales for 2024, based on the csv file?
Gerald Schuller, December 2025
"""
"""
pip3 install numpy pandas matplotlib geopandas shapely fiona
python3 make_K_world_map_share_2024.py
"""

"""
World map of EV share of new car sales for a given year (default: 2024).

Input:
- electric-car-sales-share.csv (OWID)

Output:
- K_world_map_share_2024.png

Notes:
- Uses Natural Earth (low-res) country boundaries
- Colors countries by EV share (% of new car sales)
- Grey = no data for that year
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

import geopandas as gpd
import fiona
from shapely.geometry import shape


# -----------------------------
# Configuration
# -----------------------------
YEAR = 2024
CSV_FILE = "electric-car-sales-share.csv"
OUT_PNG = "K_world_map_share_2024.png"

FIGSIZE = (38.4, 21.6)   # 4K, 16:9
DPI = 100


# -----------------------------
# Load OWID data
# -----------------------------
df = pd.read_csv(CSV_FILE)

COL = "Share of new cars that are electric"
if COL not in df.columns:
    raise ValueError(f"Column '{COL}' not found in CSV.")

# Convert to fraction if necessary
s = df[COL].astype(float)
if s.max(skipna=True) > 1.5:
    df["share"] = s / 100.0
else:
    df["share"] = s

# Select target year
dfy = df[df["Year"] == YEAR].copy()

# Drop aggregates (OWID_* codes)
dfy = dfy.dropna(subset=["Code", "share"])
dfy = dfy[~dfy["Code"].astype(str).str.startswith("OWID_")]
dfy = dfy[dfy["Code"].astype(str).str.len() == 3]
#print("dfy=", dfy)

# ISO3 -> EV share (%)
share_map = dict(zip(dfy["Code"], dfy["share"] * 100.0))

print(f"Countries with EV share data in {YEAR}: {len(share_map)}")

# -----------------------------
# Load Natural Earth geometries
# -----------------------------
WORLD_SHP = "data/natural_earth/ne_110m_admin_0_countries.shp"

if not Path(WORLD_SHP).exists():
    raise FileNotFoundError(
        f"Natural Earth shapefile not found at {WORLD_SHP}\n"
        "Download from https://www.naturalearthdata.com/downloads/110m-cultural-vectors/"
    )

patches_no_data = []
patches_data = []
values = []

with fiona.open(WORLD_SHP) as src:
    for feat in src:
        #print("feat[properties]=", feat["properties"])
        #iso3 = feat["properties"].get("ISO_A3")
        #iso3 = feat["properties"].get("ADM0_A3")
        iso3 = (
            feat["properties"].get("ADM0_A3")
            or feat["properties"].get("ISO_A3")
            or feat["properties"].get("GU_A3")
            or None
        )
        
        geom = shape(feat["geometry"])
        val = share_map.get(iso3, None)
        print("iso3", iso3,"val=", val)

        def add_poly(coords, target):
            target.append(Polygon(coords, closed=True))

        if geom.geom_type == "Polygon":
            coords = np.asarray(geom.exterior.coords)
            (patches_data if val is not None else patches_no_data).append(
                Polygon(coords, closed=True)
            )
            if val is not None:
                values.append(val)

        elif geom.geom_type == "MultiPolygon":
            for poly in geom.geoms:
                coords = np.asarray(poly.exterior.coords)
                (patches_data if val is not None else patches_no_data).append(
                    Polygon(coords, closed=True)
                )
                if val is not None:
                    values.append(val)


# -----------------------------
# Plot
# -----------------------------
fig = plt.figure(figsize=FIGSIZE, dpi=DPI)
ax = fig.add_subplot(111)
ax.set_axis_off()

# Countries without data
pc_no = PatchCollection(
    patches_no_data,
    edgecolor="white",
    linewidths=0.25,
    facecolor="#dddddd"
)
ax.add_collection(pc_no)

# Countries with data
if values:
    vmax = max(100.0, float(np.nanmax(values)))
    norm = Normalize(vmin=0.0, vmax=vmax)
    sm = ScalarMappable(norm=norm)

    colors = [sm.to_rgba(v) for v in values]
    pc = PatchCollection(
        patches_data,
        edgecolor="white",
        linewidths=0.25
    )
    pc.set_facecolor(colors)
    ax.add_collection(pc)

    cbar = fig.colorbar(sm, ax=ax, fraction=0.02, pad=0.01)
    cbar.set_label("EV share of new car sales (%)")

ax.autoscale_view()

ax.set_title(
    f"World map: EV share of new car sales ({YEAR}, OWID)",
    pad=14
)

ax.text(
    0.01, 0.02,
    f"Countries with data: {len(share_map)} • Grey = no data",
    transform=ax.transAxes,
    fontsize=16,
    bbox=dict(boxstyle="round,pad=0.35", alpha=0.25),
)

# Save
out_path = Path(OUT_PNG)
fig.savefig(out_path, bbox_inches="tight", transparent=True)
plt.close(fig)

print(f"Saved: {out_path.resolve()}")

