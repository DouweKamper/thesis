# -*- coding: utf-8 -*-
"""
Created on Tue Mar 18 09:39:49 2025

@author: Douwe

Creates a schematic of the snooker update.
"""

import matplotlib.pyplot as plt

# participating chains in snooker update
x = [0.20, 0.35, 0.59, 0.85]
y = [0.60, 0.74, 0.62, 0.75]

# not participating chains
xx = [0.25, 0.23, 0.35, 0.54, 0.59, 0.8, 0.87, 0.85, 0.20]
yy = [0.15, 0.40, 0.74, 0.29, 0.62, 0.43, 0.17, 0.75, 0.60]

# proposal snooker update
# dotted line
gradient = (y[3] - y[2]) / (x[3] - x[2])
x_min = 0
x_max = 1
y_min = y[2] - gradient * x[2]
y_max = y[2] + gradient * (x_max - x[2])
# orthogonal lines
gradient_orth = 1 / gradient
xi_orth = (y[0] + gradient_orth * x[0] - y_min) / (
    gradient_orth + gradient
)  # obtained by solving system of equations
xj_orth = (y[1] + gradient_orth * x[1] - y_min) / (
    gradient_orth + gradient
)  # similar to xi_orth
yi_orth = (
    y[0] + gradient_orth * x[0] - gradient_orth * xi_orth
)  # follows from solving the same system of equations
yj_orth = (
    y[1] + gradient_orth * x[1] - gradient_orth * xj_orth
)  # similar to yi_orth

scalar = 1.2
x_proposal = x[2] + scalar * (xj_orth - xi_orth)
y_proposal = y[2] + scalar * (yj_orth - yi_orth)

# Plot snooker update
plt.figure("snooker", figsize=(6, 3.78))
plt.xlim(0, 1)
plt.ylim(0.135, 0.765)  # plt.ylim(0.1, 0.9)
plt.scatter(xx, yy, color="grey")  # not participating chains
plt.scatter(x, y, color="black")  # participating chains
plt.scatter(x_proposal, y_proposal, color="black")  # proposal

plt.annotate(
    "",
    xy=(x_min + 0.1, y_min + 0.05),
    xytext=(xi_orth, yi_orth),
    arrowprops=dict(
        facecolor="black", arrowstyle="-", linestyle="dotted", linewidth=2
    ),
)  # start at 0.1
plt.annotate(
    "",
    xy=(xj_orth, yj_orth),
    xytext=(x[2], y[2]),
    arrowprops=dict(
        facecolor="black", arrowstyle="-", linestyle="dotted", linewidth=2
    ),
)
plt.annotate(
    "",
    xy=(x_proposal, y_proposal),
    xytext=(x_max, y_max),
    arrowprops=dict(
        facecolor="black", arrowstyle="-", linestyle="dotted", linewidth=2
    ),
)
plt.annotate(
    "",
    xy=(x[0], y[0]),
    xytext=(xi_orth, yi_orth),
    arrowprops=dict(
        facecolor="black", arrowstyle="-", linestyle="-", linewidth=2
    ),
)
plt.annotate(
    "",
    xy=(x[1], y[1]),
    xytext=(xj_orth, yj_orth),
    arrowprops=dict(
        facecolor="black", arrowstyle="-", linestyle="-", linewidth=2
    ),
)
plt.annotate(
    "",
    xy=(xj_orth, yj_orth),
    xytext=(xi_orth, yi_orth),
    arrowprops=dict(facecolor="black", arrowstyle="->"),
)
plt.annotate(
    "",
    xy=(x_proposal, y_proposal),
    xytext=(x[2], y[2]),
    arrowprops=dict(facecolor="black", arrowstyle="->"),
)

plt.text(
    x[0] + 0.02,
    y[0] - 0.02,
    r"$\theta_{i,t}$",
    fontsize=12,
    ha="left",
    va="center",
)
plt.text(
    x[1] + 0.02,
    y[1] - 0.02,
    r"$\theta_{j,t}$",
    fontsize=12,
    ha="left",
    va="center",
)
plt.text(
    x[2] + 0.02,
    y[2] - 0.02,
    r"$\theta_{k,t}$",
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
plt.text(
    x[3] + 0.025,
    y[3] - 0.015,
    r"$\theta_{h,t}$",  # r"$\theta_{l,t}$",
    fontsize=12,
    ha="left",
    va="center",
)
plt.axis("off")

# Save figure as EPS file
plt.savefig("snooker_move_2025.eps", format="eps", bbox_inches="tight")
plt.show()
plt.close()
