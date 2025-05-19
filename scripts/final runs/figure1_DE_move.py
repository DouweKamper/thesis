# -*- coding: utf-8 -*-
"""
Created on Sun Mar 17 09:36:17 2024

@author: Douwe

Creates a schematic of the Differential Evolution (DE) move.
"""

import matplotlib.pyplot as plt

# participating chains DE
x = [0.25, 0.23, 0.35]
y = [0.15, 0.40, 0.74]

# not participating chains
xx = [0.25, 0.23, 0.35, 0.54, 0.59, 0.8, 0.87, 0.85, 0.20]
yy = [0.15, 0.40, 0.74, 0.29, 0.62, 0.43, 0.17, 0.75, 0.60]

# proposal DE
scalar = 1.2
x_proposal = x[0] + scalar * (x[2] - x[1])
y_proposal = y[0] + scalar * (y[2] - y[1])

# Create subplots
plt.figure("DE", figsize=(6, 3.78))

# Plot DE
plt.xlim(0, 1)  # plt.xlim(0, 1)
plt.ylim(0.135, 0.765)  # plt.ylim(0.1, 0.9)
plt.scatter(xx, yy, color="grey")  # not participating chains
plt.scatter(x, y, color="black")  # participating chains
plt.scatter(x_proposal, y_proposal, color="black")  # proposal
plt.annotate(
    "",
    xy=(x_proposal, y_proposal),
    xytext=(x[0], y[0]),
    arrowprops=dict(facecolor="black", arrowstyle="->"),
)
plt.annotate(
    "",
    xy=(x[2], y[2]),
    xytext=(x[1], y[1]),
    arrowprops=dict(facecolor="black", arrowstyle="->"),
)
plt.text(
    x[0] + 0.02,
    y[0] - 0.02,
    r"$\theta_{k,t}$",
    fontsize=12,
    ha="left",
    va="center",
)
plt.text(
    x[1] + 0.02,
    y[1] - 0.02,
    r"$\theta_{i,t}$",
    fontsize=12,
    ha="left",
    va="center",
)
plt.text(
    x[2] + 0.02,
    y[2] - 0.02,
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
plt.axis("off")


# save DE fig
plt.savefig("DE_move_2025.eps", format="eps", bbox_inches="tight")
plt.show()
plt.close()
