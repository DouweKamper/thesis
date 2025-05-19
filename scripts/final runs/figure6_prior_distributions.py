# -*- coding: utf-8 -*-
"""
Created on Thu May  1 11:36:40 2025

@author: Douwe

Creates a figure that shows the wide and narrow prior for theta_2 in Model 2
"""

import matplotlib.pyplot as plt
import numpy as np
import os
from constants_d import UNIQUE
from main_module_d import trunc_norm

# %% select model, parameter and look up true value
MODEL_NR = 2
theta_index = 0
true_value = UNIQUE["HK"][MODEL_NR - 1][0]


# %% create figure with subplots
plt.rcParams.update(
    {
        "axes.labelsize": 33,  # Axis label font size
        "axes.labelpad": 10,  # padding between axis label and axis numbers
        "xtick.labelsize": 30,  # X-axis tick label font size
        "ytick.labelsize": 30,  # Y-axis tick label font size
        "xtick.major.size": 6,  # Length of major ticks on x-axis
        "ytick.major.size": 6,  # Length of major ticks on y-axis
        "xtick.minor.size": 6,  # Length of minor ticks on x-axis
        "ytick.minor.size": 6,  # Length of minor ticks on y-axis
        "xtick.major.width": 2.5,  # Width of major ticks on x-axis
        "ytick.minor.width": 2.5,  # Width of minor ticks on x-axis
        "ytick.major.width": 2.5,  # Width of major ticks on y-axis
        "ytick.minor.width": 2.5,  # Width of minor ticks on y-axis
    }
)

# Create a 2x4 subplot figure for the eight plots
fig, axes = plt.subplots(1, 2, figsize=(20, 10))

# Define the subplot labels and set a larger font size for all text elements
labels = ["A)", "B)"]
font_size = 33

# linewidth of plotted lines
linewidth = 2.5

# %% plot wide prior
prior_wide = trunc_norm(mean=0, std=0.5, lower=-1, upper=1)
x = np.linspace(-1, 1, 1000)  # generate x values
y_wide = prior_wide.pdf(x)  # generate y values


# primary axes
ax1 = axes[0]
ax1.set_xscale("log")  # Set logarithmic scale
ax1.set_xlim(1e-1, 1e3)  # Set start and stop values
ax1.set_xlabel("Hydraulic Conductivity (m/d)")
ax1.axvline(
    x=true_value,
    color="red",
    linestyle="--",
    label="True Value",
    linewidth=linewidth,
)
ax1.set_ylabel("Density")
ax1.text(
    0.05,
    0.9,
    labels[0],
    transform=ax1.transAxes,
    fontsize=font_size,
    fontweight="bold",
)
ax1.tick_params(axis="x", which="both", pad=10)

# Add secondary x-axis on top for first subfigure
ax2 = axes[0].twiny()  # Create secondary x-axis
ax2.plot(x, y_wide, linewidth=linewidth)
ax2.set_xlim(-1, 1)
ax2.set_ylim(0, 3.5)
ax2.set_xlabel("MCMC (-)")
ax2.set_ylabel("Density")
ax2.set_xticks([-1.0, -0.5, 0.0, 0.5, 1.0])
ax2.set_xticklabels([-1.0, -0.5, 0.0, 0.5, 1.0])

# %% plot narrow prior
prior_wide = trunc_norm(mean=0, std=0.125, lower=-1, upper=1)
x = np.linspace(-1, 1, 1000)  # generate x values
y_wide = prior_wide.pdf(x)  # generate y values


# primary axis
ax3 = axes[1]
ax3.set_xscale("log")  # Set logarithmic scale
ax3.set_xlim(1e-2, 1e2)  # Set start and stop values
ax3.set_xlabel("Hydraulic Conductivity (m/d)")
ax3.axvline(
    x=true_value,
    color="red",
    linestyle="--",
    label="True Value",
    linewidth=linewidth,
)
ax3.set_ylabel("Density")
ax3.text(
    0.05,
    0.9,
    labels[1],
    transform=ax3.transAxes,
    fontsize=font_size,
    fontweight="bold",
)
ax3.tick_params(axis="x", which="both", pad=10)

# Add secondary x-axis on top for first subfigure
ax4 = axes[1].twiny()  # Create secondary x-axis
ax4.plot(x, y_wide, linewidth=linewidth)
ax4.set_xlim(-1, 1)
ax4.set_ylim(0, 3.5)
ax4.set_xlabel("MCMC (-)")
ax4.set_ylabel("Density")
ax4.set_xticks([-1.0, -0.5, 0.0, 0.5, 1.0])
ax4.set_xticklabels([-1.0, -0.5, 0.0, 0.5, 1.0])

# %% make all axes thicker
for axis in [ax1, ax2, ax3, ax4]:
    for spine in axis.spines.values():
        spine.set_linewidth(1.2)


# %% Adjust layout and save
plt.tight_layout(rect=[0, 0, 1, 1.0])  # leave space for the title
output_dir = "."  # represents current directory
filename = "priors.png"
file_path = os.path.join(output_dir, filename)
plt.savefig(file_path, dpi=300, bbox_inches="tight")
plt.show()
