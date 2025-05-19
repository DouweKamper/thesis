# -*- coding: utf-8 -*-
"""
Created on Wed Aug 21 15:17:28 2024

@author: Douwe

module for diagnostics to analyze results from running MCMCM with emcee
"""

# external modules
import numpy as np
import emcee
import pandas as pd
import itertools
import math
from matplotlib.colors import LinearSegmentedColormap
from constants_d import UNIQUE


def emcee_to_modflow(MODEL_NR, theta_emcee, theta_index, prior_name):
    """
    Convert theta emcee to theta modflow.

    Parameters:
        MODEL_NR (int) = identifier for the MODFLOW model
        theta_emcee (float) = value of theta as used by emcee
        theta_index (int) = used for identifying the lithology
        prior_name (str) = the name of the prior

    Returns:
        theta_modflow (float) = transformed parameter (m/d) as used by modflow
    """
    theta_true = UNIQUE["HK"][MODEL_NR - 1][theta_index]
    # theta modflow
    if prior_name == "priorbroad":
        lithology = UNIQUE["LITHOLOGY"][MODEL_NR - 1]
        if lithology[theta_index] == "clean sand":
            theta_modflow = 10 ** (2 * theta_emcee + 1)
        elif lithology[theta_index] == "silty sand":
            theta_modflow = 10 ** (2 * theta_emcee + 0)
        elif lithology[theta_index] == "silt, loess":
            theta_modflow = 10 ** (2 * theta_emcee - 1)
        else:
            raise ValueError(
                f"Unknown lithology '{lithology[theta_index]}' for model number {MODEL_NR}"
            )
    elif prior_name == "priornarrow":
        order_of_magnitude = math.floor(math.log10(theta_true))
        theta_modflow = 10 ** (2 * theta_emcee + order_of_magnitude)
    return theta_modflow


def rescale(abs_logdif, max_value, min_value):
    """
    Rescales abs_logdif, so it ranges from 0 to 1.

    Parameters:
        abs_logdif (float) = absolute log difference between true modflow
                             parameter value and the one from calibration
        max_value (float) = largest abs_logdif in dataset
        min_value (float) = smalles abs_logdif in dataset

    Returns:
        abs_logdif_scaled (float)
    """
    abs_logdif_scaled = abs_logdif / (max_value - min_value)
    return abs_logdif_scaled


def float_to_rgba(x):
    """
    Converts floating point value to rgba array

    Parameters:
        x (float) = floating point value

    Returns:
        rgba (array of floats) = array containing (red, green, blue, alpha)
    """
    colors_d = ["green", "yellow", "red"]
    positions_d = [0, 0.5, 1]
    cmap_d = LinearSegmentedColormap.from_list(
        "colormap_d", list(zip(positions_d, colors_d))
    )

    rgba = cmap_d(x)
    return rgba


def rgba_to_hex(rgba):
    """
    Converts an RGBA color value to a hexadecimal color string, formatted for
    Overleaf compatibility.

    Parameters:
        rgba (tuple of float) = color value in RGBA format, with each component
                                in the range [0, 1].

    Returns:
        hex_color (str) = hexadecimal color string (e.g., "AA0044") without the
                          # prefix.
    """
    # Convert each RGB component to an integer (0-255)
    r, g, b = [int(255 * x) for x in rgba[:3]]
    # Format and return as hex
    hex_color = f"{r:02X}{g:02X}{b:02X}"
    return hex_color


def get_chains(
    nlocs,
    move,
    MODEL_NR,
    ensemble,
    prior_name,
    flat=False,
    burn_in=0,
    steps=1500,
):
    """
    retrieves data from a file selected using: nlocs, move, MODEL_NR and ensemble.
    Note the entire selected ensemble is retrieved.

    Parameters:
        nlocs (int) = number of locations (x,y) with observations
        move (string) = the utilized move
        MODEL_NR (int) = identifier for the MODFLOW model
        ensemble (int) = identifier for the ensemble
        prior_name (string) = name of prior, consistent with scipy.stats
        flat (bool) = optional, whether all chains are concatenated
        burn_in (int) = optional, if specified, only values post burn-in are
                        returned (so only the main sampling)
        steps (int) = optional, total chain length

    Returns:
        chains (array of float64) = chains, size: (n_steps, n_chains, n_dim)
    """
    filename = f"nlocs{nlocs}_{move}_ensemble{ensemble}_steps{steps}_model{MODEL_NR}_{prior_name}.npy"
    chains = np.load(filename)
    chains = chains[burn_in:, :, :]  # remove burn-in
    if flat is True:
        chains = chains.flatten()
    return chains


def acceptance_fraction(chains):
    """
    Calculates acceptance fraction of proposals for an ensemble, by averaging
    over all chains in the ensemble.

    Parameters:
        chains (array) = ensemble of MCMC chains, [step,chain,parameter]

    Returns:
        frac (float) = accepted fraction of moves
    """
    chains_theta1 = chains[:, :, 0]  # only keep the first parameter
    # accepted proposals = (nr of unique floats in ensemble) - (nr of chains)
    accepted_proposals = len(np.unique(chains_theta1)) - chains.shape[1]
    # frac = (accepted proposals) / ((nr of steps per chain) * (nr of chains))
    frac = accepted_proposals / chains_theta1.size
    return frac


def acceptance_fraction_single(chain):
    """
    Calculates acceptance fraction of proposals for a single chain.

    Parameters:
        chains (array) = MCMC chain, [step,parameter]

    Returns:
        frac (float) = accepted fraction of moves
    """
    # Check if chain is a 2D array
    if not isinstance(chain, np.ndarray):
        raise ValueError("Input 'chain' must be a Numpy Array")
    if not len(chain.shape) == 2:
        raise ValueError(
            "Input 'chain' must be a 2D array with shape [step, parameter]."
        )

    chain_theta1 = chain[:, 0]  # only keep the first parameter
    # accepted proposals = (nr of unique floats in ensemble) - (nr of chains)
    accepted_proposals = len(np.unique(chain_theta1)) - chain.shape[1]
    # frac = (accepted proposals) / ((nr of steps per chain) * (nr of chains))
    frac = accepted_proposals / chain_theta1.size
    return frac


def create_filename_inputs(moves, model_numbers, ensembles):
    """
    Creates a dataframe for filename inputs. These filenames contain MCMC
    ensembles carried out with package emcee.

    Parameters:
        moves (list of string) = the utilized move(s)
        model_numbers (list of int) = identifiers for MODFLOW models
        ensembles (list of int) = identifiers for ensembles

    Returns:
        df (DataFrame) = DataFrame of filename inputs
    """
    combinations = list(itertools.product(moves, model_numbers, ensembles))
    df = pd.DataFrame(combinations, columns=["Move", "MODEL_NR", "Ensemble"])
    return df


def get_tau(move, MODEL_NR, ensemble):
    """
    Retrieves integrated autocorrelation time (tau) from a file selected using:
    move, MODEL_NR and ensemble.

    Parameters:
        move (string) = the utilized move
        MODEL_NR (int) = identifier for the MODFLOW model
        ensemble (int) = identifier for the ensemble

    Returns:
        tau (np.array) = estimate of integrated autocorrelation time
    """
    filename = f"{move}_ensemble{ensemble}_steps1000_model{MODEL_NR}.h5"
    reader = emcee.backends.HDFBackend(filename, read_only=True)
    chains = reader.get_chain()  # (n_step, n_walker, n_param)
    tau = emcee.autocorr.integrated_time(chains, tol=1, quiet=True)
    return tau


def tau_mean(tau_list):
    """
    calculates the mean value of the integrated autocorrelation time (tau) for
    each parameter. when a nan value is encountered it is excluded from the
    calculation

    Parameters:
        tau_list (list) = a list of np.array's' consisting of tau values

    Returns:
        tau_mean (np.array) = array of tau values averaged over the ensembles
    """
    n_dim = len(tau_list[0])
    tau_mean = np.zeros(n_dim)
    for i in range(n_dim):
        tau_i_list = []
        for tau_ensemble in tau_list:
            tau_i = tau_ensemble[i]
            if not np.isnan(tau_i):
                tau_i_list.append(tau_i)
        tau_i_mean = sum(tau_i_list) / len(tau_i_list)
        tau_mean[i] = tau_i_mean
    return tau_mean
