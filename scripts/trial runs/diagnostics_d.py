# -*- coding: utf-8 -*-
"""
Created on Wed Aug 21 15:17:28 2024

@author: Douwe

module for diagnostics to analyze results from running MCMCM with emcee
"""

import numpy as np
import emcee
import pandas as pd
import itertools


def reader(move, MODEL_NR, ensemble):
    """
    retrieves data from a file selected using: move, MODEL_NR and ensemble.
    Note the entire selected ensemble is retrieved.

    Parameters:
        move (string) = the utilized move
        MODEL_NR (int) = identifier for the MODFLOW model
        ensemble (int) = identifier for the ensemble

    Returns:
        data (backends.hdf.HDFBackend) = info on emcee ensemble
    """
    filename = f"{move}_ensemble{ensemble}_steps1000_model{MODEL_NR}.h5"
    data = emcee.backends.HDFBackend(filename, read_only=True)
    return data


def get_chains(move, MODEL_NR, ensemble, flat=False):
    """
    retrieves chains from a file selected using: move, MODEL_NR and ensemble.
    Note that all chains from the selected ensemble are retrieved.

    Parameters:
        move (string) = the utilized move
        MODEL_NR (int) = identifier for the MODFLOW model
        ensemble (int) = identifier for the ensemble
        flat (bool) = optional, whether all chains are concatenated

    Returns:
        chains (array of float64) = chains, size: (n_steps, n_chains, n_dim)
    """
    data = reader(move, MODEL_NR, ensemble)
    chains = data.get_chain(flat=flat)
    return chains


def acceptance_fraction(move, MODEL_NR, ensemble, chain, parameter=1):
    """
    retrieves acceptance fraction of moves for a specific chain.

    Parameters:
        move (string) = the utilized move
        MODEL_NR (int) = identifier for the MODFLOW model
        ensemble (int) = identifier for the ensemble
        chain (int) = identifier for the chain
        parameter (int) = optional, identifier for the parameter

    Returns:
        frac (float) = accepted fraction of moves
    """
    chains = get_chains(move, MODEL_NR, ensemble)
    chain_parameter = chains[:, chain - 1, parameter - 1]
    unique = np.unique(chain_parameter)  # array of unique floats
    frac = len(unique) / len(chain_parameter)
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
