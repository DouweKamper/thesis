# -*- coding: utf-8 -*-
"""
Created on Sun Nov  3 19:44:58 2024

@author: Douwe

Creates a Kernel Density Estimate (KDE) plot showing 10 chains of ensemble 1 from DE-SNK after calibrating Model 1.
"""

from diagnostics_d import get_chains, emcee_to_modflow
import numpy as np
import pandas as pd
import seaborn as sns
import os
import matplotlib.pyplot as plt

ensemble = get_chains(
    nlocs=1,
    move="DEsnooker",
    MODEL_NR=1,
    ensemble=1,
    prior_name="priorbroad",
    flat=False,
    burn_in=1000,
    steps=2000,
)


# Convert to DataFrame
n_steps, n_chains, n_dim = ensemble.shape
df = pd.DataFrame(
    {
        "Hydraulic Conductivity (m/d)": ensemble.reshape(
            -1
        ),  # Flatten the array
        "chain": np.repeat(np.arange(n_chains), n_steps),
    }
)

df["Hydraulic Conductivity (m/d)"] = df["Hydraulic Conductivity (m/d)"].apply(
    lambda x: emcee_to_modflow(
        MODEL_NR=1, theta_emcee=x, theta_index=0, prior_name="priorbroad"
    )
)

# Plot with seaborn
plt.figure(figsize=(10, 6))
sns.kdeplot(
    data=df,
    x="Hydraulic Conductivity (m/d)",
    hue="chain",
    fill=True,
    log_scale=True,
    common_norm=False,
    palette="crest",
    alpha=0.5,
    linewidth=0,
)

plt.xlim(0.1, 1000)
output_dir = "."  # represents current directory
filename = "kde_model1_ensemble_variation.png"
file_path = os.path.join(output_dir, filename)
plt.savefig(file_path, dpi=300, bbox_inches="tight")
plt.show()
