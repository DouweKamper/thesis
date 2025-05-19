# -*- coding: utf-8 -*-
"""
Created on Wed May  7 12:20:23 2025

@author: Douwe

Creates boxplots for the acceptance rate table in the report.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from diagnostics_d import get_chains, acceptance_fraction_single


# Create empty DataFrame
index = [
    "Stretch 1",
    "DE 1",
    "DEsnooker 1",
    "Stretch 2",
    "DE 2",
    "DEsnooker 2",
    "Stretch 3",
    "DE 3",
    "DEsnooker 3",
    "Stretch 4",
    "DE 4",
    "DEsnooker 4",
]
columns = [
    "1 priorbroad",
    "3 priorbroad",
    "5 priorbroad",
    "1 priornarrow",
    "3 priornarrow",
    "5 priornarrow",
]
df = pd.DataFrame(index=index, columns=columns)

# %% Fill DataFrame
for MODEL_NR in [1, 2, 3, 4]:
    for move in ["Stretch", "DE", "DEsnooker"]:
        for prior_name in ["priorbroad", "priornarrow"]:
            for nlocs in [1, 3, 5]:
                frac_list = []
                for ensemble in [1, 2, 3, 4, 5]:
                    chains = get_chains(
                        nlocs=nlocs,
                        move=move,
                        MODEL_NR=MODEL_NR,
                        ensemble=ensemble,
                        prior_name=prior_name,
                        flat=False,
                        burn_in=1000,  # could be less?
                        steps=2000,
                    )

                    for i in range(10):  # loop through all 10 chains
                        chain_i = chains[:, i, :]  # select chain i
                        frac_ind = acceptance_fraction_single(chain_i)
                        frac_list.append(frac_ind)
                frac_array = np.array(frac_list)
                df.at[
                    f"{move} {MODEL_NR}", f"{nlocs} {prior_name}"
                ] = frac_array


# %% create figures with 9 box plots each
def add_boxplot(axes, data):
    """
    Adds a customized boxplot to a subfigure

    Parameters:
        axes (axes._axes.Axes) = Axes object of matplotlib.axes._axes module
        data (array) = acceptance rates (float64) of 10 chains * 5 ensembles


    Returns:
        nothing
    """
    # Create boxplot
    axes.boxplot(
        x=data,
        vert=False,
        widths=0.8,
        showfliers=False,
        medianprops=dict(linestyle="none"),
        boxprops=dict(linewidth=3, facecolor="lightgrey"),  # Thicker box lines
        whiskerprops=dict(linewidth=3),  # Thicker whisker lines
        capprops=dict(linewidth=3),  # Thicker caps
        patch_artist=True,  # fill with color
    )

    # Set plot limits and styling
    axes.set_xlim(0.01, 0.828)
    axes.set_axis_off()  # Hide axes (spines, ticks, labels)
    axes.margins(x=0, y=0)  # Set zero margins within the axes


for MODEL_NR in [1, 2, 3, 4]:
    for prior_name in ["priorbroad", "priornarrow"]:
        fig, axes = plt.subplots(3, 3, figsize=(15.5, 3.15))
        for move in enumerate(["Stretch", "DE", "DEsnooker"]):
            for nlocs in enumerate([1, 3, 5]):
                axes_subplot = axes[move[0], nlocs[0]]
                data_subplot = df.at[
                    f"{move[1]} {MODEL_NR}", f"{nlocs[1]} {prior_name}"
                ]
                # optimal acceptance rates by Schmon and Gagnon (2022)
                add_boxplot(axes=axes_subplot, data=data_subplot)
        # add vertical line indicating optimal acceptance rates
        optimal_frac = [0.4400, 0.3500, 0.3130, 0.2839][MODEL_NR - 1]
        for col in range(3):
            # x_fig determines line position, where 0.827 is x-axis length and 0.002 is a correcting offset
            x_fig = (col / 3) + (optimal_frac / 0.827) / 3 - 0.002
            fig.add_artist(
                plt.Line2D(
                    [x_fig, x_fig],
                    [0.03, 0.97],
                    transform=fig.transFigure,
                    color="red",
                    linestyle=(0, (7, 7)),
                    linewidth=3.39,
                )
            )
        fig.subplots_adjust(
            left=0, right=1, top=1, bottom=0, wspace=0, hspace=0
        )  # remove padding
        filename = f"boxplots_model{MODEL_NR}_{prior_name}.png"
        file_path = os.path.join(".", filename)  # "." = current directory
        plt.savefig(file_path, dpi=300, bbox_inches="tight", pad_inches=0)
        plt.show()
