# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 15:54:28 2024

@author: Douwe

Creates a table with the posterior medians (color coded) for all scenarios.
"""

import pandas as pd
import numpy as np
import math
from diagnostics_d import get_chains, rescale, float_to_rgba, rgba_to_hex
from constants_d import UNIQUE


# %% Create empty DataFrame
index = []
for i in range(4):
    ndim = UNIQUE["NDIM"][i]
    for theta in range(1, ndim + 1):
        for move in ["Stretch", "DE", "DEsnooker"]:
            index.append(f"Model {i+1} theta {theta} {move}")

columns = [
    "1 priorbroad",
    "3 priorbroad",
    "5 priorbroad",
    "1 priornarrow",
    "3 priornarrow",
    "5 priornarrow",
]

# DataFrame that contains median modflow parameter values after calibration
df = pd.DataFrame(index=index, columns=columns)
# Dataframe that contains absolute log differences
df_abslogdif = pd.DataFrame(index=index, columns=columns)
burn_in = 1000

# %% fill df with absolute log difference values
for MODEL_NR in [1, 2, 3, 4]:
    for move in ["Stretch", "DE", "DEsnooker"]:
        for prior_name in ["priorbroad", "priornarrow"]:
            for nlocs in [1, 3, 5]:
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
                    if ensemble == 1:
                        chains_all = chains
                    else:
                        chains_all = np.concatenate((chains_all, chains))
                ndim = chains.shape[-1]
                for theta in range(ndim):
                    chains_theta = chains_all[:, :, theta]
                    median_emcee = np.median(chains_theta)
                    theta_true = UNIQUE["HK"][MODEL_NR - 1][theta]
                    # median modflow
                    if prior_name == "priorbroad":
                        lithology = UNIQUE["LITHOLOGY"][MODEL_NR - 1]
                        if lithology[theta] == "clean sand":
                            median_modflow = 10 ** (2 * median_emcee + 1)
                        elif lithology[theta] == "silty sand":
                            median_modflow = 10 ** (2 * median_emcee + 0)
                        elif lithology[theta] == "silt, loess":
                            median_modflow = 10 ** (2 * median_emcee - 1)
                        else:
                            raise ValueError(
                                f"Unknown lithology '{lithology[theta]}' for model number {MODEL_NR}"
                            )
                    elif prior_name == "priornarrow":
                        order_of_magnitude = math.floor(math.log10(theta_true))
                        median_modflow = 10 ** (
                            2 * median_emcee + order_of_magnitude
                        )
                    logdif = math.log(theta_true / median_modflow)
                    abs_logdif = abs(logdif)
                    df_abslogdif.at[
                        f"Model {MODEL_NR} theta {theta+1} {move}",
                        f"{nlocs} {prior_name}",
                    ] = f"{abs_logdif}"
                    df.at[
                        f"Model {MODEL_NR} theta {theta+1} {move}",
                        f"{nlocs} {prior_name}",
                    ] = f"{median_modflow:.2g}"


# %% convert dataframe values to hex (HTML color code)
df_float = df_abslogdif.apply(
    pd.to_numeric, errors="coerce"
)  # convert to float
min_value = df_float.min().min()
max_value = df_float.max().max()
df_scaled = df_float.map(lambda x: rescale(x, max_value, min_value))
df_rgba = df_scaled.map(lambda x: float_to_rgba(x))
df_hex = df_rgba.map(lambda x: rgba_to_hex(x))

# %% add HTML color codes to df
# create new map function where I add abs_logdif and df_hex to the same cell as: abs_logdif (df_hex),
# to be used by chatgpt to convert to properly formatted latex table (using \cellcolor)
df_combined = df + " (" + df_hex + ")"

# %% convert df_hex to latex table
latex_table = df_combined.to_latex(escape=False)  # allows parentheses in LaTeX
print(latex_table)

df.style.background_gradient(axis=None)
