# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 16:05:09 2024

@author: Douwe

calibrates my 4 MODFLOW models with emcee, with several ensembels per model and
    utilizing different moves. 
    Also using parallel computing (on a personal computer)
"""

import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"  # to avoid interference with emcee-
os.environ["MKL_NUM_THREADS"] = "1"  # paralellization, need to define-
os.environ["BLIS_NUM_THREADS"] = "1"  # environmental variables before-
os.environ["NUMEXPR_NUM_THREADS"] = "1"  # importing numpy
import numpy as np
import emcee
import scipy.stats
from main_module_d import MCMC


if __name__ == "__main__":
    np.random.seed(123)  # set global seed for reproducability

    # %% function inputs for run_parallel
    LOCATIONS = ((0, 6, 2), (0, 7, 4), (0, 12, 20))
    error_distribution = scipy.stats.norm(0, 0.1)  # likelihood
    prior = scipy.stats.norm(10, 10)  # for now all priors are the same
    nwalkers = 10
    ensembles = 3
    burn_in = 500
    main_sampling = 1000
    all_moves = [
        [emcee.moves.StretchMove(), "Stretch"],
        [emcee.moves.DEMove(), "DE"],
        [
            [(emcee.moves.DEMove(), 0.8), (emcee.moves.DESnookerMove(), 0.2)],
            "DEsnooker",
        ],
    ]  # first entry of each sublist is for emcee, the second for the file name
    k0_seed = 666

    # %% run an ensemble for all models, for different moves
    for moves in all_moves:
        MCMC(
            LOCATIONS,
            error_distribution,
            prior,
            nwalkers,
            ensembles,
            moves,
            burn_in,
            main_sampling,
            k0_seed,
        )
