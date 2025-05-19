# -*- coding: utf-8 -*-
"""
Created on Sun Nov  3 19:44:58 2024

@author: Douwe

Creates pair plots for different models and different samplers:
    - fig  8 DE-SNK Model 2
    - fig  9 DE-SNK Model 3
    - fig 10 DE-SNK Model 4
    - fig D4 DE     Model 2
    - fig D5 DE     Model 3
    - fig D6 DE     Model 4
    - fig D7 AI     Model 2
    - fig D8 AI     Model 3
    - fig D9 AI     Model 4
"""
from diagnostics_d import get_chains
import numpy as np
import pandas as pd
import seaborn as sns
import os
import matplotlib.pyplot as plt

# set constants
prior_name = "priorbroad"
columns_all = [
    ["theta_1", "theta_2"],
    ["theta_1", "theta_2", "theta_3"],
    ["theta_1", "theta_2", "theta_3", "theta_4", "theta_5"],
]

for i, MODEL_NR in enumerate([2, 3, 4]):
    for move in ["Stretch", "DE", "DEsnooker"]:
        for ensemble in [1, 2, 3, 4, 5]:
            chains = get_chains(
                nlocs=1,
                move=move,
                MODEL_NR=MODEL_NR,
                ensemble=ensemble,
                prior_name=prior_name,
                flat=False,
                burn_in=1000,
                steps=2000,
            )
            if ensemble == 1:
                chains_all = chains
            else:
                chains_all = np.concatenate((chains_all, chains))

        n_steps, n_chains, n_dim = chains_all.shape
        # concatenate chains
        chains_flat = chains_all.reshape(n_chains * n_steps, n_dim)
        # convert array to DataFrame
        columns = columns_all[i]
        chains_df = pd.DataFrame(chains_flat, columns=columns)

        # Plot with seaborn
        plt.figure(figsize=(10, 6))
        g = sns.pairplot(chains_df, kind="kde")

        # Modify axis labels to include theta_1 to theta_5 with LaTeX formatting
        for ax in g.axes.flatten():
            # Set x-axis labels
            if ax.get_xlabel() == "theta_1":
                ax.set_xlabel(r"$\theta_1$")
            elif ax.get_xlabel() == "theta_2":
                ax.set_xlabel(r"$\theta_2$")
            elif ax.get_xlabel() == "theta_3":
                ax.set_xlabel(r"$\theta_3$")
            elif ax.get_xlabel() == "theta_4":
                ax.set_xlabel(r"$\theta_4$")
            elif ax.get_xlabel() == "theta_5":
                ax.set_xlabel(r"$\theta_5$")

            # Set y-axis labels
            if ax.get_ylabel() == "theta_1":
                ax.set_ylabel(r"$\theta_1$")
            elif ax.get_ylabel() == "theta_2":
                ax.set_ylabel(r"$\theta_2$")
            elif ax.get_ylabel() == "theta_3":
                ax.set_ylabel(r"$\theta_3$")
            elif ax.get_ylabel() == "theta_4":
                ax.set_ylabel(r"$\theta_4$")
            elif ax.get_ylabel() == "theta_5":
                ax.set_ylabel(r"$\theta_5$")

        # plt.legend(title="Chains")
        output_dir = "."  # represents current directory
        filename = f"kde_model{MODEL_NR}_{move}.png"
        file_path = os.path.join(output_dir, filename)
        plt.savefig(file_path, dpi=300, bbox_inches="tight")
        plt.show()
