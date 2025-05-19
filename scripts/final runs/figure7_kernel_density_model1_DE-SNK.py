"""
@author: Douwe

creates Kernel Density Estimate (KDE) plots of DE-SNK from calibrating Model 1 
"""


import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from diagnostics_d import get_chains, emcee_to_modflow
from constants_d import UNIQUE
from main_module_d import trunc_norm

# Set font sizes globally using rcParams
plt.rcParams.update(
    {
        "axes.labelsize": 14,  # Axis label font size
        "axes.titlesize": 18,  # Title font size
        "xtick.labelsize": 14,  # X-axis tick label font size
        "ytick.labelsize": 14,  # Y-axis tick label font size
    }
)

# Create a 2x4 subplot figure for the eight plots
fig, axes = plt.subplots(2, 4, figsize=(20, 12))

# Define the subplot labels and set a larger font size for all text elements
labels = ["A)", "B)", "C)", "D)", "E)", "F)", "G)", "H)"]
font_size = 19

# First row: Broad prior and its posteriors
prior_broad = trunc_norm(mean=0, std=0.5, lower=-1, upper=1)
sample_broad = prior_broad.rvs(size=100000)
sample_broad_df = pd.DataFrame({"Hydraulic Conductivity (m/d)": sample_broad})
MODEL_NR = 1
theta_index = 0

# Apply the transformation and plot broad prior
prior_name = "priorbroad"
df_broad = sample_broad_df.apply(
    lambda x: emcee_to_modflow(MODEL_NR, x, theta_index, prior_name)
)
sns.kdeplot(
    data=df_broad,
    x="Hydraulic Conductivity (m/d)",
    fill=True,
    log_scale=True,
    alpha=0.5,
    ax=axes[0, 0],
)

true_value = UNIQUE["HK"][MODEL_NR - 1][0]
axes[0, 0].axvline(
    x=true_value, color="red", linestyle="--", label="True Value"
)
axes[0, 0].set_title("Wide Prior")
axes[0, 0].set_xlim(0.1, 1000)
axes[0, 0].set_ylim(0, 4.0)
axes[0, 0].text(
    0.05,
    0.9,
    labels[0],
    transform=axes[0, 0].transAxes,
    fontsize=font_size,
    weight="bold",
)


# Plot posteriors for broad prior with different nlocs values
for i, nlocs in enumerate([1, 3, 5]):
    chains_all = []
    for ensemble in [1, 2, 3, 4, 5]:
        chains = get_chains(
            nlocs=nlocs,
            move="DEsnooker",
            MODEL_NR=MODEL_NR,
            ensemble=ensemble,
            prior_name=prior_name,
            flat=True,
            burn_in=1000,
            steps=2000,
        )
        if ensemble == 1:
            chains_all = chains
        else:
            chains_all = np.concatenate((chains_all, chains))

    df_emcee = pd.DataFrame({"Hydraulic Conductivity (m/d)": chains_all})
    df_posterior = df_emcee.apply(
        lambda x: emcee_to_modflow(MODEL_NR, x, theta_index, prior_name)
    )

    sns.kdeplot(
        data=df_posterior,
        x="Hydraulic Conductivity (m/d)",
        fill=True,
        log_scale=True,
        alpha=0.5,
        ax=axes[0, i + 1],
    )
    true_value = UNIQUE["HK"][MODEL_NR - 1][0]
    axes[0, i + 1].axvline(
        x=true_value, color="red", linestyle="--", label="True Value"
    )
    median_emcee = np.median(chains_all)
    median = emcee_to_modflow(
        MODEL_NR=1,
        theta_emcee=median_emcee,
        theta_index=0,
        prior_name="priorbroad",
    )
    axes[0, i + 1].axvline(
        x=median, color="orange", linestyle="--", label="Median"
    )
    axes[0, i + 1].set_title(f"Posterior (nlocs={nlocs})")
    axes[0, i + 1].set_xlim(0.1, 1000)
    axes[0, i + 1].set_ylim(0, 4.0)
    axes[0, i + 1].text(
        0.05,
        0.9,
        labels[i + 1],
        transform=axes[0, i + 1].transAxes,
        fontsize=font_size,
        fontweight="bold",
    )
    if nlocs == 5:
        axes[0, i + 1].legend(fontsize=13.8, handlelength=1.0)

# Second row: Narrow prior and its posteriors
prior_narrow = trunc_norm(mean=0, std=0.125, lower=-1, upper=1)
sample_narrow = prior_narrow.rvs(size=100000)
sample_narrow_df = pd.DataFrame(
    {"Hydraulic Conductivity (m/d)": sample_narrow}
)

# Apply the transformation and plot narrow prior
prior_name = "priornarrow"
df_narrow = sample_narrow_df.apply(
    lambda x: emcee_to_modflow(MODEL_NR, x, theta_index, prior_name)
)
sns.kdeplot(
    data=df_narrow,
    x="Hydraulic Conductivity (m/d)",
    fill=True,
    log_scale=True,
    alpha=0.5,
    ax=axes[1, 0],
)
true_value = UNIQUE["HK"][MODEL_NR - 1][0]
axes[1, 0].axvline(
    x=true_value, color="red", linestyle="--", label="True Value"
)
axes[1, 0].set_title("Narrow Prior")
axes[1, 0].set_xlim(0.1, 1000)
axes[1, 0].set_ylim(0, 4.0)
axes[1, 0].text(
    0.05,
    0.9,
    labels[4],
    transform=axes[1, 0].transAxes,
    fontsize=font_size,
    fontweight="bold",
)

# Plot posteriors for narrow prior with different nlocs values
for i, nlocs in enumerate([1, 3, 5]):
    chains_all = []
    for ensemble in [1, 2, 3, 4, 5]:
        chains = get_chains(
            nlocs=nlocs,
            move="DEsnooker",
            MODEL_NR=MODEL_NR,
            ensemble=ensemble,
            prior_name=prior_name,
            flat=True,
            burn_in=1000,
            steps=2000,
        )
        if ensemble == 1:
            chains_all = chains
        else:
            chains_all = np.concatenate((chains_all, chains))

    df_emcee = pd.DataFrame({"Hydraulic Conductivity (m/d)": chains_all})
    df_posterior = df_emcee.apply(
        lambda x: emcee_to_modflow(MODEL_NR, x, theta_index, prior_name)
    )

    sns.kdeplot(
        data=df_posterior,
        x="Hydraulic Conductivity (m/d)",
        fill=True,
        log_scale=True,
        alpha=0.5,
        ax=axes[1, i + 1],
    )
    true_value = UNIQUE["HK"][MODEL_NR - 1][0]
    axes[1, i + 1].axvline(
        x=true_value, color="red", linestyle="--", label="True Value"
    )
    median_emcee = np.median(chains_all)
    median = emcee_to_modflow(
        MODEL_NR=1,
        theta_emcee=median_emcee,
        theta_index=0,
        prior_name="priornarrow",
    )
    axes[1, i + 1].axvline(
        x=median, color="orange", linestyle="--", label="Median"
    )
    axes[1, i + 1].set_title(f"Posterior (nlocs={nlocs})")
    axes[1, i + 1].set_xlim(0.1, 1000)
    axes[1, i + 1].set_ylim(0, 4.0)
    axes[1, i + 1].text(
        0.05,
        0.9,
        labels[i + 5],
        transform=axes[1, i + 1].transAxes,
        fontsize=font_size,
        fontweight="bold",
    )

# Adjust layout and save
plt.tight_layout(rect=[0, 0, 1, 0.96])  # leave space for the title
output_dir = "."  # represents current directory
filename = "kde_model1_DEsnooker.png"
file_path = os.path.join(output_dir, filename)
plt.savefig(file_path, dpi=300, bbox_inches="tight")
plt.show()
