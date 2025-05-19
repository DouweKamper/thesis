# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 15:54:28 2024

@author: Douwe

table with a count of how many ensembles have all Rhat below the threshold of 1.05
"""

import pandas as pd
import numpy as np
import arviz as az
from diagnostics_d import get_chains
from constants_d import UNIQUE

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

# Fill Dataframe
for MODEL_NR in [1, 2, 3, 4]:
    D = UNIQUE["NDIM"][MODEL_NR - 1]  # nr of parameters
    for move in ["Stretch", "DE", "DEsnooker"]:
        for prior_name in ["priorbroad", "priornarrow"]:
            for nlocs in [1, 3, 5]:
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
                    if max(rhat_list) < 1.05:
                        count_threshold += 1
                # nr of ensembles for which all chains and all theta: R < 1.05
                df.at[
                    f"{move} {MODEL_NR}", f"{nlocs} {prior_name}"
                ] = f"{count_threshold}"

# Calculate the mean column and append it to the DataFrame
mean_row = df.apply(lambda x: np.mean(x.astype(float)), axis=0)
df.loc["Col Average"] = mean_row

# Calculate the mean row and append it to the DataFrame
df["Row Average"] = df.iloc[:, :].astype(float).mean(axis=1)

# convert DataFrame to latex table
latex_table = df.to_latex(escape=False)  # allows parentheses in LaTeX
print(latex_table)
