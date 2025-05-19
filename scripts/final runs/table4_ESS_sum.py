# -*- coding: utf-8 -*-
"""
Created on Sun May  4 12:24:22 2025

@author: Douwe

Calculates the Effective Sample Size (ESS) for all ensembles summed together. 
"""

import pandas as pd
import numpy as np
import arviz as az
import emcee
from diagnostics_d import get_chains

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
burn_in = 1000

for MODEL_NR in [1, 2, 3, 4]:
    for move in ["Stretch", "DE", "DEsnooker"]:
        for prior_name in ["priorbroad", "priornarrow"]:
            for nlocs in [1, 3, 5]:
                tau_list = []
                for ensemble in [1, 2, 3, 4, 5]:
                    chains = get_chains(
                        nlocs=nlocs,
                        move=move,
                        MODEL_NR=MODEL_NR,
                        ensemble=ensemble,
                        prior_name=prior_name,
                        flat=False,
                        burn_in=burn_in,  # could be less?
                        steps=2000,
                    )
                    for i in range(chains.shape[1]):
                        chain = chains[:, i, :]  # select a single chain
                        tau = emcee.autocorr.integrated_time(
                            chain, tol=1, quiet=True, has_walkers=False
                        )  # one tau value for each parameter
                        tau_mean = np.mean(tau)  # average over parameters
                        tau_list += [tau_mean]
                ess_list = []
                for tau_ind in tau_list:
                    ess_list += [burn_in / tau_ind]
                ess_sum = np.sum(ess_list)
                df.at[
                    f"{move} {MODEL_NR}", f"{nlocs} {prior_name}"
                ] = f"{ess_sum:.0f}"

# Calculate the mean column and append it to the DataFrame
mean_col = df.apply(lambda x: np.mean(x.astype(int)), axis=0)
df.loc["Average"] = mean_col

# Calculate the mean row and append it to the DataFrame
df["Row Average"] = df.iloc[:, :].astype(float).mean(axis=1)

# convert DataFrame to latex table
latex_table = df.to_latex(escape=False)  # allows parentheses in LaTeX
print(latex_table)
