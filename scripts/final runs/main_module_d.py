# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 11:33:32 2024

@author: Douwe

module containting functions and classes for implementing emcee with MODFLOW
"""
# external modules:
import numpy as np
import scipy.stats
import emcee
import random
import math
from multiprocessing import Pool

# modules created by Douwe:
from models_d import model_output
from constants_d import UNIQUE


def trunc_norm(mean, std, lower, upper):
    """
    creates a normalised truncated normal distribution, cut-off at lower & upper.
    After truncation the mean stays equal, but the standard deviation (std) actually
    becomes smaller as a consequence of truncation. This distribution is used as
    the prior distribution.

    Parameters:
        mean (float, int) = mean of distribution
        std (float, int) = standard deviation of untruncated distribution (trucated std is smaller)
        lower (float, int) = lower boundary of trucated normal distribution
        upper (float, int) = upper boundary of truncated normal distribution

    Returns:
        trunc_dist (scipy.stats object) = truncated normal distribution
    """
    a = (lower - mean) / std
    b = (upper - mean) / std
    trunc_dist = scipy.stats.truncnorm(a, b, loc=mean, scale=std)
    return trunc_dist


def create_observations():
    """
    Run all models to obtain synthetic observations (withouth noise) of hydraulic
    heads and saves them as npy files
    """
    for MODEL_NR in UNIQUE["MODEL_NR"]:
        HEADS_ALL = model_output(MODEL_NR)  # run model
        np.save(f"HEADS_ALL{MODEL_NR}.npy", HEADS_ALL)


def observations(MODEL_NR, LOCATIONS):
    """
    Retrieves obervations for given MODEL_NR and LOCATIONS

    Parameters:
        MODEL_NR (int) = identifier for the MODFLOW model
        LOCATIONS (tuple) = measurement locations

    Returns:
        obs (array of floats) = observations with noise at specified locations
    """
    all_obs = np.load(f"HEADS_ALL{MODEL_NR}_NOISE.npy")
    obs = np.empty(len(LOCATIONS))
    for index, loc in enumerate(LOCATIONS):
        obs[index] = all_obs[loc]
    return obs


def create_noise_single(observation, noise_dist):
    """
    Adds noise to a single observation

    Parameters:
        observation = a single observation (float)
        noise_dist = distribution (scipy.stats object)

    Returns:
        observation + noise (float) = observation with added noise
    """
    noise = noise_dist.rvs()
    return observation + noise


def create_noise(std, numpy_seed):
    """
    Adds noise to the obervations of all models and saves this as npy files

    Parameters:
        std (float) = standard deviation of the measurements, measurement error
        numpy_seed (int) = seed used by numpy for generating locations

    Returns:
        -
    """
    np.random.seed(numpy_seed)
    for MODEL_NR in UNIQUE["MODEL_NR"]:
        model_output = np.load(f"HEADS_ALL{MODEL_NR}.npy")
        noise_dist = scipy.stats.norm(0, std)
        create_noise_vectorized = np.vectorize(
            create_noise_single
        )  # modifies function, so it can be applied to an array directly
        obs_with_noise = create_noise_vectorized(model_output, noise_dist)
        np.save(f"HEADS_ALL{MODEL_NR}_NOISE.npy", obs_with_noise)


class ModelStructure:
    """
    Contains information on a MODFLOW6 model simulation
    """

    def __init__(self, y, model_nr, locations, ndim):
        self.y = y  # observations
        self.model_nr = model_nr  # int
        self.locations = locations  # list of tuples
        self.ndim = ndim  # int


def create_model_structure(MODEL_NR, LOCATIONS):
    """
    Creates an object of class ModelStructure

    Parameters:
        LOCATIONS (list of tuples) = list of (z,x,y) coordinates
        MODEL_NR (int) = model identifier

    Returns:
        model_structure (ModelStructure) = object of class ModelStructure
    """
    y = observations(MODEL_NR, LOCATIONS)
    ndim = UNIQUE["NDIM"][MODEL_NR - 1]
    model_structure = ModelStructure(y, MODEL_NR, LOCATIONS, ndim)
    return model_structure


def make_locs_xy(nlocs, random_seed=None):
    """
    stochastically chooses (x,y) coordinates for measurement locations

    Parameters:
        nlocs (int): number of measurement locations per layer
        random_seed (int): the seed to be used by the random module

    Returns:
        locations (list of tuples) = list of (x,y) coordinates
    """
    if random_seed is None:
        random.seed(nlocs)  # use the number of locations to generate seed.
    else:
        random.seed(random_seed)

    locations = []
    while len(locations) != nlocs:
        x = random.randint(2, 24)
        y = random.randint(2, 24)
        if (x, y) not in locations:
            locations.append((x, y))
    return locations


def make_locs_z(LOCATIONS, MODEL_NR):
    """
    adds z coordinates to (x,y) tuples for a specific model

    Parameters:
        LOCATIONS (list of tuples) = list of (x,y) coordinates
        MODEL_NR (int) = model identifier

    Returns:
        locations_3d (list of tuples) = list of (z,x,y) coordinates
        where z is the layer number and x,y are cell numbers in x and y direction
    """
    locations_3d = []
    ndim = UNIQUE["NDIM"][MODEL_NR - 1]
    for z in range(ndim):
        for xy in LOCATIONS:
            zxy = (z,) + xy
            locations_3d.append(zxy)
    return locations_3d


def scale_k(k, MODEL_NR, prior_name):
    """
    Scales k, from [-1,1] as it is used in emcee for numerical stability
    towards values representing hydraulic conductivity (m/d) to be used in
    the function model_output for MODFLOW simulations. A formula for scaling k
    is selected based on the selected prior.

    Parameters:
        k (tuple) = scaled hydraulic conductivity as used by emcee
        MODEL_NR (int) = the model number of the model calibrated with emcee
        prior_name (str) = name of the prior, used for transforming k

    Returns:
        k_new (tuple) = rescaled hydraulic conductivity (m/d) as used by MODFLOW
    """
    ndim = UNIQUE["NDIM"][MODEL_NR - 1]
    if prior_name == "priorbroad":
        lithology = UNIQUE["LITHOLOGY"][MODEL_NR - 1]
        k_new = ()
        for i in range(ndim):
            if lithology[i] == "clean sand":
                k_new += (10 ** (2 * k[i] + 1),)
            elif lithology[i] == "silty sand":
                k_new += (10 ** (2 * k[i] + 0),)
            elif lithology[i] == "silt, loess":
                k_new += (10 ** (2 * k[i] - 1),)
            else:
                raise ValueError(
                    f"Unknown lithology '{lithology[i]}' for model number {MODEL_NR}"
                )
        return k_new
    elif prior_name == "priornarrow":
        k_true_all = UNIQUE["HK"][MODEL_NR - 1]
        k_new = ()
        for i in range(ndim):
            k_true = k_true_all[i]
            k_emcee = k[i]
            order_of_magnitude = math.floor(math.log10(k_true))
            k_new += (10 ** (2 * k_emcee + order_of_magnitude),)
        return k_new
    else:
        raise ValueError(f"{prior_name} is an incorrect name")


def log_posterior(k, error_distribution, priors, prior_name, MODEL):
    """
    Calculates the unnormalised log posterior density

    Parameters:
        k = vector of current position of chain (provided by emcee.EnsembleSampler)
        error_distribution (scipy.stats object) = the distribution of the measurement error (likelihood)
        priors (tuple of scipy.stats objects) = all priors
        prior_name (str) = name of the prior, used for transforming k
        MODEL = the MODEL of class ModelStructure

    Returns:
        logposterior (float) = log posterior density
    """
    for k_ind in k:
        if k_ind < -1 or k_ind > 1:  # if outside prior range
            # implementation of parameter limits: https://emcee.readthedocs.io/en/stable/user/faq/
            return -np.inf
    K = scale_k(k, MODEL.model_nr, prior_name)  # rescale k for MODFLOW
    h = model_output(MODEL.model_nr, MODEL.locations, K=K)
    if h is None:  # in this case MODFLOW did not run succesfully
        return -np.inf
    else:
        loglikelihood = np.sum(error_distribution.logpdf(h - MODEL.y))
        logprior = sum(map(lambda priors, k: priors.logpdf(k), priors, k))
        logposterior = loglikelihood + logprior
        return logposterior


def run_parallel(
    MODEL,
    priors,
    prior_name,
    error_distribution,
    k0,
    nwalkers,
    moves,
    steps,
    i,
    nprocesses=5,
):
    """
    Run an MCMC ensemble using python package emcee with multiprocessing and
        save results as a npy file.

    Parameters:
        MODEL (ModelStructure) = the MODEL of class ModelStructure
        priors (tuple of scipy.stats objects) = all priors
        prior_name (str) = name of the prior, used for file naming and transforming k
        error_distribution (scipy.stats object) = the distribution of the measurement error (likelihood)
        k0 (array of floats) = the initial position of all walkers
        nwalkers (int) = nr of chains or walkers
        ensembles (int) = nr of ensembles
        moves (list containing an emcee.moves object and its name as a string) = moves to be used by the sampler
        steps (int) = nr of steps per chain, including burn-in
        i (int) = ensemble identifier to be used for file naming
        nprocesses (int) = number of processors to be used

    Returns:
        -
    """
    ndim = MODEL.ndim
    with Pool(processes=nprocesses) as pool:
        # initialize the sampler
        multi_sampler = emcee.EnsembleSampler(
            nwalkers,
            ndim,
            log_posterior,
            pool=pool,
            moves=moves[0],
            args=[error_distribution, priors, prior_name, MODEL],
        )
        # start sampling
        multi_sampler.run_mcmc(k0, steps, progress=True)
        # save chains as npy file
        nlocs = int(len(MODEL.locations) / ndim)  # nlocs is part of filename
        filename = f"nlocs{nlocs}_{moves[1]}_ensemble{i+1}_steps{steps}_model{MODEL.model_nr}_{prior_name}.npy"
        samples = multi_sampler.get_chain()
        np.save(filename, samples)


def MCMC(
    LOCATIONS,
    error_distribution,
    prior,
    prior_name,
    nwalkers,
    ensembles,
    moves,
    steps,
    k0_seed,
):
    """
    Run several MCMC ensembles, for all designed MODFLOW6 models.

    Parameters:
        LOCATIONS (list of tuples) = contains measurement locations, with (z,x,y) coordinates
        error_distribution (scipy.stats object) = distribution of the measurement error (likelihood)
        prior (scipy.stats object) = the prior (identical for all parameters)
        prior_name (str) = name of the prior, used for file naming and transforming k
        nwalkers (int) = nr of chains or walkers
        ensembles (int) = nr of ensembles
        moves (list containing an emcee.moves object and its name as a string) = moves to be used by the sampler
        steps (int) = nr of steps per chain, including burn-in
        k0_seed (int) = seed initial position of walkers

    Returns:
        -
    """
    for MODEL_NR in UNIQUE["MODEL_NR"]:
        locations_3d = make_locs_z(LOCATIONS, MODEL_NR)
        MODEL = create_model_structure(MODEL_NR, locations_3d)
        ndim = MODEL.ndim
        priors = ndim * (prior,)
        ensembles_init = prior.rvs(
            size=(ensembles, nwalkers, ndim), random_state=k0_seed
        )  # generate initial locations of samplers stochastically
        for i in range(ensembles):
            k0 = ensembles_init[i, :, :]  # initial position of chains
            run_parallel(
                MODEL,
                priors,
                prior_name,
                error_distribution,
                k0,
                nwalkers,
                moves,
                steps,
                i,
            )
