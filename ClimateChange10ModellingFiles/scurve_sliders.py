"""
Climate Change Assistant: Can you make a python program for plotting a variable S-Curve, with sliders for the inflection point t0,  the growth speed k, and the saturation L?

Gerald Schuller, December 2025.

pip3 install streamlit matplotlib numpy
streamlit run scurve_sliders.py
"""


import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

def logistic(t, L, k, t0):
    return L / (1 + np.exp(-k * (t - t0)))

st.set_page_config(page_title="S-curve sliders", layout="centered")

st.title("Interactive S-curve (logistic adoption model)")
st.write("Adjust **t0** (inflection year), **k** (growth speed), and **L** (saturation).")

t_min, t_max = 2000, 2040
t = np.linspace(t_min, t_max, 800)

t0 = st.slider("t0 (inflection year)", float(t_min), float(t_max), 2025.0, 0.1)
k  = st.slider("k (growth speed)", 0.01, 2.0, 0.4, 0.01)
L  = st.slider("L (saturation, %)", 10.0, 110.0, 100.0, 1.0)
show_inflection = st.checkbox("Show inflection marker (t0, L/2)", value=True)

y = logistic(t, L, k, t0)

fig = plt.figure(figsize=(10, 5))
ax = fig.add_subplot(111)
ax.plot(t, y, linewidth=3)
ax.set_xlim(t_min, t_max)
ax.set_ylim(0, max(110, L))
ax.set_xlabel("Year (t)")
ax.set_ylabel("Adoption (%)")
ax.set_title("S-curve: L / (1 + exp(-k(t - t0)))")
ax.grid(True, linestyle=":", alpha=0.6)

if show_inflection:
    ax.axvline(t0, linestyle="--", linewidth=2)
    ax.scatter([t0], [L/2], s=80)
    ax.text(t0 + 0.2, L/2, "inflection (50%)", va="center")

st.pyplot(fig)

