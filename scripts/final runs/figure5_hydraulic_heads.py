# -*- coding: utf-8 -*-
"""
Created on Tue Mar 18 11:54:18 2025

@author: Douwe

Creates subplots of the hydraulic heads for each model.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib

# Enable LaTeX rendering for text
matplotlib.rc("text", usetex=True)

from models_d import model_output
from constants_d import UNIVERSAL, UNIQUE

# %% Calculate hydraulic heads
heads1 = model_output(1)
heads2 = model_output(2)
heads3 = model_output(3)
heads4 = model_output(4)

# %% Define font sizes
label_fontsize = 30  # Axis labels
number_fontsize = 28  # Axis tick numbers
title_fontsize = 42  # Titles
legend_fontsize = 28  # Legend text

# %% Create figure and axes
fig, axs = plt.subplots(2, 2, figsize=(16, 12))  # 2x2 grid of plots
x = np.arange(UNIVERSAL["NCOL"])


# Function to plot models
def plot_model(ax, heads, model_name, nlay):
    for ilay in range(nlay):
        ax.plot(x, heads[ilay, 12, :], label=f"Layer {ilay+1}")

    ax.set_title(r"\textit{" + model_name + "}", fontsize=title_fontsize)
    ax.set_xlabel("Column (-)", fontsize=label_fontsize)
    ax.set_ylabel("Hydraulic Head (m)", fontsize=label_fontsize)
    ax.legend(loc="lower left", fontsize=legend_fontsize)

    ax.set_xlim(0, 24)
    ax.set_ylim(0, 10)

    # Set integer ticks on both axes
    ax.set_xticks(np.arange(0, 24, 1))  # Ticks at every integer
    ax.set_yticks(np.arange(0, 11, 1))  # Ticks at every integer

    # Show numbers only every 5 ticks
    ax.set_xticklabels(
        [str(i) if i % 5 == 0 else "" for i in range(24)],
        fontsize=number_fontsize,
    )
    ax.set_yticklabels(
        [str(i) if i % 5 == 0 else "" for i in range(11)],
        fontsize=number_fontsize,
    )

    # Increase line thickness for axes
    ax.spines["top"].set_linewidth(1.2)
    ax.spines["right"].set_linewidth(1.2)
    ax.spines["bottom"].set_linewidth(1.2)
    ax.spines["left"].set_linewidth(1.2)

    ax.tick_params(axis="both", which="major", width=2.5, length=6)


# Apply function to each subplot
plot_model(axs[0, 0], heads1, "Model 1", UNIQUE["NLAY"][0])
plot_model(axs[0, 1], heads2, "Model 2", UNIQUE["NLAY"][1])
plot_model(axs[1, 0], heads3, "Model 3", UNIQUE["NLAY"][2])
plot_model(axs[1, 1], heads4, "Model 4", UNIQUE["NLAY"][3])

plt.tight_layout()
plt.savefig("heads_side.png", dpi=300, bbox_inches="tight")
plt.show()
