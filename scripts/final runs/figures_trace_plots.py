# -*- coding: utf-8 -*-
"""
Created on Thu May 15 19:12:48 2025

@author: Douwe
"""

import numpy as np
import arviz as az
import matplotlib.pyplot as plt
from diagnostics_d import get_chains
from constants_d import UNIQUE

# select worst ensembles
worst_ensembles = {}
MODEL_NR = 1
D = UNIQUE["NDIM"][MODEL_NR - 1]  # nr of parameters
for move in ["Stretch", "DE", "DEsnooker"]:
    for prior_name in ["priorbroad", "priornarrow"]:
        nlocs = 5
        count_threshold = 0
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
            rhat_list = []
            for j in range(D):  # loop through parameters
                chain_theta = chains[:, :, j]
                chains_T = np.transpose(chain_theta)
                chains_az = az.convert_to_dataset(chains_T)
                rhat_az = az.rhat(chains_az, method="rank")
                rhat = rhat_az["x"].values  # rhat is a 0-dim array
                # multiply by 1 to append a float instead of an array
                rhat_list.append(rhat * 1)
            if ensemble == 1:
                bad_ensemble = (1, max(rhat_list), chains)
            elif max(rhat_list) > bad_ensemble[1]:
                bad_ensemble = (ensemble, max(rhat_list), chains)
            worst_ensembles[f"{move} {prior_name}"] = bad_ensemble


# creat trace plots
for key in worst_ensembles:
    # Create a 10x1 subplot figure for the ten plots
    fig, axes = plt.subplots(10, 1, figsize=(20, 24))
    chains = worst_ensembles[key][2]
    for i in range(10):
        chain_i = chains[:, i]
        step = np.array(range(1000))
        axes[i].plot(step, chain_i)
        axes[i].set_xlim(0, 1000)
        axes[i].set_ylim(-1, 1)
        axes[i].set_title(f"Chain {i+1}", fontsize=18)
        axes[i].set_yticks([-1, -0.5, 0, 0.5, 1])
        axes[i].tick_params(axis="both", labelsize=14)
        if i != 9:
            axes[i].set_xticks([])  # hide axis labes and ticks
        else:
            axes[i].set_xticks([0, 200, 400, 600, 800, 1000])
            axes[i].set_xlabel("step number", fontsize=18)
        # second y axis
        twin = axes[i].twinx()
        twin.tick_params(axis="both", labelsize=14)
        twin.tick_params(axis="y", which="minor", length=4, width=0.5)
        twin.set_yscale("log")
        twin.minorticks_on()
        if "priorbroad" in key:
            twin.set_yticks([0.1, 1, 10, 100, 1000])
        else:
            twin.set_yticks([0.01, 0.1, 1, 10, 100])
        # set axis labels
        if i == 4:
            axes[i].set_ylabel("MCMC (-)", fontsize=18)
            twin.set_ylabel("Hydraulic Conductivity (m/d)", fontsize=18)
    ensemble_nr = worst_ensembles[key][0]
    filename = f"trace_plots_ensemble{ensemble_nr}_{key}"
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.show()
