# -*- coding: utf-8 -*-
"""
Created on Wed Aug 21 15:13:38 2024

@author: Douwe

Creates trace plots.
"""

import matplotlib.pyplot as plt
from diagnostics_d import get_chains

# %% Stretch
data_unflat = get_chains("Stretch", 4, 3, flat=False)
data_chain2 = data_unflat[:, 1, :]

# Create the trace plot
# Create the trace plot with one subplot for each variable
fig, ax = plt.subplots(figsize=(10, 6))

# Plot each variable's trace
for i in range(data_chain2.shape[1]):
    ax.plot(data_chain2[:, i], label=f"Variable {i+1}")

# Adding labels and legend
ax.set_xlabel("Step Number")
ax.set_ylabel("hydraulic conductivity (m/d)")
ax.legend(loc="upper right")

plt.tight_layout()
plt.show()


# %% DE
data_unflat = get_chains("DE", 4, 3, flat=False)
data_chain2 = data_unflat[:, 1, :]

# Create the trace plot
# Create the trace plot with one subplot for each variable
fig, ax = plt.subplots(figsize=(10, 6))

# Plot each variable's trace
for i in range(data_chain2.shape[1]):
    ax.plot(data_chain2[:, i], label=f"Variable {i+1}")

# Adding labels and legend
ax.set_xlabel("Step Number")
ax.set_ylabel("hydraulic conductivity (m/d)")
ax.legend(loc="upper right")

plt.tight_layout()
plt.show()


# %% DEsnooker
data_unflat = get_chains("DEsnooker", 4, 3, flat=False)
data_chain2 = data_unflat[:, 1, :]

# Create the trace plot
# Create the trace plot with one subplot for each variable
fig, ax = plt.subplots(figsize=(10, 6))

# Plot each variable's trace
for i in range(data_chain2.shape[1]):
    ax.plot(data_chain2[:, i], label=f"Variable {i+1}")

# Adding labels and legend
ax.set_xlabel("Step Number")
ax.set_ylabel("hydraulic conductivity (m/d)")
ax.legend(loc="upper right")

plt.tight_layout()
plt.show()
