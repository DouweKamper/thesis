# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 11:56:47 2024

@author: Douwe

Creates a schematic of the Stretch move.
"""

import matplotlib.pyplot as plt
import numpy as np

# set the seed for numpy
rng = np.random.default_rng(1)
# participating chains
x = [0.25, 0.59]
y = [0.15, 0.62]

# not participating chains
xx = [0.25, 0.23, 0.35, 0.54, 0.59, 0.8, 0.87, 0.85]
yy = [0.15, 0.40, 0.74, 0.29, 0.62, 0.43, 0.17, 0.75]


# proposal
scalar = 1.2
x_proposal = x[0] + scalar * (x[1] - x[0])
y_proposal = y[0] + scalar * (y[1] - y[0])


plt.figure("stretch_move", figsize=(6, 3.78))
plt.xlim(0, 1)
plt.ylim(0.135, 0.765)  # plt.ylim(0.1, 0.8)
plt.scatter(xx, yy, color="grey")  # not participating chains
plt.scatter(x, y, color="black")  # participating chains
plt.scatter(x_proposal, y_proposal, color="black")  # proposal

# Add arrow between two points
plt.annotate(
    "",
    xy=(x_proposal, y_proposal),
    xytext=(x[0], y[0]),
    arrowprops=dict(facecolor="black", arrowstyle="->"),
)

# Add text at a specific coordinate
plt.text(
    x[0] + 0.02,
    y[0] - 0.02,
    r"$\theta_{k,t}$",
    fontsize=12,
    ha="left",
    va="center",
)  # try: ,fontname='Source Code Pro', fontweight='normal'
plt.text(
    x[1] + 0.02,
    y[1] - 0.02,
    r"$\theta_{j,t}$",
    fontsize=12,
    ha="left",
    va="center",
)
plt.text(
    x_proposal + 0.02,
    y_proposal - 0.02,
    r"$\theta_{k,t+1}$",
    fontsize=12,
    ha="left",
    va="center",
)


# remove axis
plt.axis("off")

# Save figure as EPS file
plt.savefig("stretch_move.eps", format="eps", bbox_inches="tight")
