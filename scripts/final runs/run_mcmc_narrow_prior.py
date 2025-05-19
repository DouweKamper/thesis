# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 16:05:09 2024

@author: Douwe

run MCMC with an informative Gaussian prior and chain initialisation
"""

# external modules:
import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"  # to avoid interference with emcee-
os.environ["MKL_NUM_THREADS"] = "1"  # paralellization, need to define-
os.environ["BLIS_NUM_THREADS"] = "1"  # environmental variables before-
os.environ["NUMEXPR_NUM_THREADS"] = "1"  # importing numpy
import emcee
import scipy.stats

# modules created by Douwe:
from main_module_d import (
    MCMC,
    make_locs_xy,
    create_observations,
    create_noise,
    trunc_norm,
)


if __name__ == "__main__":
    # %% create obs and add noise
    create_observations()
    create_noise(std=0.1, numpy_seed=123)

    # %% function inputs for MCMC
    error_distribution = scipy.stats.norm(0, 0.1)  # likelihood
    prior = trunc_norm(mean=0, std=0.125, lower=-1, upper=1)
    prior_name = "priornarrow"
    nwalkers = 10
    ensembles = 5
    steps = 2000
    all_moves = [
        [emcee.moves.StretchMove(), "Stretch"],
        [emcee.moves.DEMove(), "DE"],
        [
            [(emcee.moves.DEMove(), 0.9), (emcee.moves.DESnookerMove(), 0.1)],
            "DEsnooker",
        ],
    ]  # first entry of each sublist is for emcee, the second for the file name
    k0_seed = 666  # for consistent chain initialisation across samplers

    # %% run MCMC with uninformative Gaussian prior and chain initialisation
    ALL_LOCATIONS = []
    for nlocs in (1, 3, 5):
        ALL_LOCATIONS.append(make_locs_xy(nlocs))

    for LOCATIONS in ALL_LOCATIONS:
        for moves in all_moves:
            MCMC(
                LOCATIONS,
                error_distribution,
                prior,
                prior_name,
                nwalkers,
                ensembles,
                moves,
                steps,
                k0_seed,
            )
