# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 11:33:32 2024

@author: Douwe

module containting functions and classes for implementing emcee with MODFLOW
"""

import numpy as np
import scipy.stats
import emcee
from multiprocessing import Pool, cpu_count
from models_d import model_output
from constants_d import UNIQUE


def create_observations():
    """
    Run all models to obtain synthetic observations of hydraulic heads and
    saves them as npy files
    """
    for MODEL_NR in UNIQUE["MODEL_NR"]:
        HEADS_ALL = model_output(MODEL_NR)  # run model
        np.save(f"HEADS_ALL{MODEL_NR}.npy", HEADS_ALL)


def observations(MODEL_NR, LOCATIONS):
    """
    Retrieves obervations for given MODEL_NR and LOCATIONS

    MODEL_NR = int, identifier for the MODFLOW model
    LOCATIONS = tuple, measurement locations
    """
    all_obs = np.load(f"HEADS_ALL{MODEL_NR}_NOISE.npy")
    obs = np.empty(len(LOCATIONS))
    for index, loc in enumerate(LOCATIONS):
        obs[index] = all_obs[loc]
    return obs


def create_noise_single(observation, noise_dist):
    """
    adds noise to a single observation
    where:
        observation = a single observation (float)
        noise_dist = distribution (scipy.stats object)
    """
    noise = noise_dist.rvs()
    return observation + noise


def create_noise(std):
    """
    Adds noise to the obervations of all models and saves this as npy files
    where:
        std = standard deviation of the measurements, measurement error (float)
    """
    for MODEL_NR in UNIQUE["MODEL_NR"]:
        model_output = np.load(f"HEADS_ALL{MODEL_NR}.npy")
        noise_dist = scipy.stats.norm(0, std)
        create_noise_vectorized = np.vectorize(
            create_noise_single
        )  # modifies function, so it can be applied to an array directly
        obs_with_noise = create_noise_vectorized(model_output, noise_dist)
        np.save(f"HEADS_ALL{MODEL_NR}_NOISE.npy", obs_with_noise)


class ModelStructure:
    def __init__(self, y, model_nr, locations, ndim):
        self.y = y  # observations
        self.model_nr = model_nr  # int
        self.locations = locations  # tuple
        self.ndim = ndim  # int


def create_model_structure(MODEL_NR, LOCATIONS):
    y = observations(MODEL_NR, LOCATIONS)
    ndim = UNIQUE["NDIM"][MODEL_NR - 1]
    model_structure = ModelStructure(y, MODEL_NR, LOCATIONS, ndim)
    return model_structure


def log_posterior(
    k, error_distribution, priors, MODEL  # gaussian likelihood and prior
):
    """
    Calculates the unnormalised log posterior density
    where:
        k = vector providing (current position) of chain, also called parameter values
        error_distribution = the distribution of the measurement error (likelihood)
        priors = a tuple of all priors
        MODEL = the MODEL of class ModelStructure
    """
    for k_ind in k:
        if k_ind <= 0:  # negative k are theoretically impossible
            return -np.inf
    h = model_output(MODEL.model_nr, MODEL.locations, K=k)
    if h is None:  # in this case Modflow did not run succesfully
        return -np.inf  # return -inf, because an AssertionError took place
    else:
        loglikelihood = np.sum(error_distribution.logpdf(h - MODEL.y))
        logprior = sum(map(lambda priors, k: priors.logpdf(k), priors, k))
        logposterior = loglikelihood + logprior
        return logposterior


def MCMC(
    LOCATIONS,
    error_distribution,
    prior,
    nwalkers,
    ensembles,
    moves,
    burn_in,
    main_sampling,
    k0_seed,
):
    """
    run an MCMC ensemble using python package emcee with multiprocessing
    where:
        LOCATIONS = a tuple of tuples providing measurement locations
        error_distribution = the distribution of the measurement error (likelihood)
        prior = the prior used for all priors (scipy.stats)
        nwalkers = nr of chains or walkers
        ensembles = nr of ensembles
        moves = which moves should be used to move the Markov chain
        burn_in = nr of burn in steps
        main_sampling = nr of main sampling steps
        k0_seed = seed initial position of walkers (int)
    """
    for MODEL_NR in UNIQUE["MODEL_NR"]:
        MODEL = create_model_structure(MODEL_NR, LOCATIONS)
        ndim = MODEL.ndim  # need to define something for this
        # set seed only for initialisation so the different moves have the same starting points
        rng = np.random.default_rng(k0_seed)
        ensembles_init = rng.uniform(
            0.0001, 20, size=(ensembles, nwalkers, ndim)
        )

        for i in range(ensembles):
            k0 = ensembles_init[i, :, :]  # initialisation of chains
            priors = ndim * (prior,)
            run_parallel(
                MODEL,
                priors,
                error_distribution,
                k0,
                nwalkers,
                moves,
                burn_in,
                main_sampling,
                i,
            )


def run_parallel(
    MODEL,
    priors,
    error_distribution,
    k0,
    nwalkers,
    moves,
    burn_in,
    main_sampling,
    i,
):
    """
    run a MCMC sampler using python package emcee with multiprocessing
    where:
        MODEL = the MODEL of class ModelStructure
        priors = a tuple of all priors
        error_distribution = the distribution of the measurement error (likelihood)
        k0 = the initial position of all walkers (np.ndarray)
        nwalkers = nr of chains or walkers
        ensembles = nr of ensembles
        moves = which moves should be used to move the Markov chain (emcee.moves)
        burn_in = nr of burn in steps
        main_sampling = nr of main sampling steps
        i = ensemble identifier (used for file naming)
    """
    ndim = MODEL.ndim
    # going multi
    with Pool(processes=cpu_count()) as pool:
        # set up backend
        filename = f"{moves[1]}_ensemble{i+1}_steps{main_sampling}_model{MODEL.model_nr}.h5"  # .h5 -> save as h5 file
        backend = emcee.backends.HDFBackend(filename)
        backend.reset(nwalkers, ndim)
        # initialize the sampler
        multi_sampler = emcee.EnsembleSampler(
            nwalkers,
            ndim,
            log_posterior,
            pool=pool,
            moves=moves[0],
            args=[error_distribution, priors, MODEL],
            backend=backend,
        )
        # Burn-in
        multi_state = multi_sampler.run_mcmc(
            k0, burn_in, progress=True
        )  # loc after burn-in steps
        multi_sampler.reset()

        # Main sampling
        multi_sampler.run_mcmc(multi_state, main_sampling, progress=True)
