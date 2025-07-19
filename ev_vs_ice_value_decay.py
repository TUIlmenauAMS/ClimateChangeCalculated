"""
Chatbot: Assume we have an electric car and an internal combustion engine car. Both cost 40000 euro new, and have a remaining value of 1000 eur at the end of their life. Both have an exponential decay of their value over their lifetime to the final remaining value. Assume the internal combustion car has a lifetime of 15 years, and the electric car a lifetime of 66 years. Can you make a plot of their respective value over their life, both in the same plot?
Gerald Schuller, July 2025
"""

import numpy as np
import matplotlib.pyplot as plt

# Initial and final values
initial_value = 40000
final_value = 1000

# Lifetimes
lifetime_ice = 15
lifetime_ev = 66

# Decay constants
def decay_constant(V0, Vt, t):
    return -np.log(Vt / V0) / t

k_ice = decay_constant(initial_value, final_value, lifetime_ice)
k_ev = decay_constant(initial_value, final_value, lifetime_ev)

# Time arrays
t_ice = np.linspace(0, lifetime_ice, 300)
t_ev = np.linspace(0, lifetime_ev, 300)

# Value over time
value_ice = initial_value * np.exp(-k_ice * t_ice)
value_ev = initial_value * np.exp(-k_ev * t_ev)

# Plot
plt.figure(figsize=(10, 6))
plt.plot(t_ice, value_ice, label='ICE Car (15 years)', color='red')
plt.plot(t_ev, value_ev, label='Electric Car (66 years)', color='green')
plt.xlabel('Years')
plt.ylabel('Value (€)')
plt.title('Vehicle Depreciation: ICE vs Electric Car')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

