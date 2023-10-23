"""Theoretical model."""

import os
import itertools

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm

# =====================================================
# Function to set initial conditions
# =====================================================


def set_initial_conditions(J=20, K=15, M=625,
                           nprop=10, distrib_env="uniform",
                           seed_init=1234):
    """"Setting initial conditions.

    :param J: Number of species.
    :param K: Number of environmental dimensions.
    :param M: Number of sites.
    :param nprop: Number of initial propagules per species.
    :param distrib_env: Distribution for environmental
        variables. Either "normal" or "uniform".
    :param seed_init: Seed for initial conditions.

    :return: A dictionary with initial conditions.

    """

    # RNG
    rng_init = np.random.default_rng(seed_init)

    # Environment
    if distrib_env == "uniform":
        E = rng_init.uniform(0, 1, (K, M))
    elif distrib_env == "normal":
        E = rng_init.normal(0, 1, (K, M))
        # Normalize on [0, 1]
        E = (E - np.min(E)) / (np.max(E) - np.min(E))
    # Species optima
    Xopt = rng_init.uniform(0, 1, (J, K))

    # Initial occupancy
    Occ = np.zeros(M, dtype="int")-1
    initial_pos = rng_init.choice(
        np.where(Occ < 0)[0], size=J*nprop, replace=False)
    for jj in range(J):
        Occ[initial_pos[jj*nprop:(jj+1)*nprop]] = jj

    return {"Occ": Occ, "E": E, "Xopt": Xopt}


# =====================================================
# Function to compute propagule performance
# =====================================================


def calc_perf_propagules(E, Xopt, ndim, propagule_pos, propagule_sp,
                         mu_res, sig_res, mup, sigp,
                         sp_intercept=True, uIV=False,
                         rng_dyn=np.random.default_rng(1234)):
    """Computing propagule performances.

    The performance of a propagule depends on both the environment and
    the species optima. Residuals associated with the missing
    dimensions can be added to the performance computation. Per
    species residuals have a mean and a standard deviation. Either the
    mean or standard deviation of the species residuals or both can be
    taken into account to compute species and propagule performances.

    :param E: K x M np.array. Environment with K dimensions on M sites.
    :param Xopt: J x K np.array. Species optima for the K dimensions.
    :param ndim: Number of observed environmental dimensions.
    :param propagule_pos: Sites reached by propagules.
    :param propagule_sp: Species for each propagule.
    :param mu_res: Residual mean per sp. due to unobserved dimensions.
    :param sig_res: Residual std per sp. due to unobserved dimensions.
    :param mup: Mean performance of all species on all sites.
    :param sigp: Std of the performance of all species on all sites.
    :param sp_intercept: Add species intercept from missing dimensions.
    :param uIV: Add random IV from missing dimensions.
    :param rng_dyn: RNG for dynamics.

    :return: Propagule performances.

    """

    E_propagules = E[:ndim, propagule_pos]
    Xopt_propagules = Xopt[propagule_sp, :ndim]
    perf = np.sum((E_propagules - Xopt_propagules.T)**2, axis=0)**0.5
    if sp_intercept:
        perf += mu_res[propagule_sp]
    if uIV:
        perf += rng_dyn.normal(0, sig_res[propagule_sp], size=perf.shape)

    # Return opposite of distance (-)
    return -(perf-mup)/sigp


# =====================================================
# Function to simulate the community dynamics
# =====================================================


def simulate_dynamics(J=20, K=15, M=625, seed_init=1234,
                      death=0.01, nprop=10, fecund=0.5,
                      mortality="deterministic", fecundity="percapita",
                      distrib_env="uniform",
                      ndim=1, sp_intercept=False, uIV=False,
                      tmax=10000, seed_dyn=1234):
    """Simulating community dynamics.

    :param J: Number of species.
    :param K: Number of environmental dimensions.
    :param M: Number of sites.
    :param seed_init: Random seed for initial conditions.

    :param death: Fraction of dead individuals per turn.
    :param nprop:  Nb of propagules for fixed fecundity.
    :param fecund: Nb of propagules per capita (for
        abundance-dependent fecundity).

    :param mortality: Either "deterministic" or "random".
    :param fecundity: Either "fixed" or "percapita".
    :param distrib_env: Distribution for environmental
        variables. Either "normal" or "uniform".

    :param ndim: Number of observed dimensions (ndim <= K).
    :param sp_intercept: Add species intercept from missing dimensions.
    :param uIV: Add random IV from missing dimensions.

    :param tmax: Maximal number of iterations.
    :param seed_dyn: Random seed for dynamics.

    :return: Dataframe of results.

    """

    # RNG simu
    rng_dyn = np.random.default_rng(seed=seed_dyn)

    # Initial conditions
    init = set_initial_conditions(J, K, M, nprop,
                                  distrib_env, seed_init)
    Occ = np.copy(init["Occ"])
    E = init["E"]
    Xopt = init["Xopt"]

    # Performance of species on all sites
    perf_tot = np.sum((E[np.newaxis, ...] -
                       Xopt[..., np.newaxis])**2, axis=1)**0.5
    mup, sigp = np.mean(perf_tot), np.std(perf_tot)

    # Residual mean and sd per species due to missing dimensions
    # perf_res = np.dot(Xopt[:, ndim:], E[ndim:, :])
    E_dim = E[:ndim, :]
    Xopt_dim = Xopt[:, :ndim]
    perf_obs = np.sum((E_dim[np.newaxis, ...] -
                       Xopt_dim[..., np.newaxis])**2, axis=1)**0.5
    perf_res = perf_tot - perf_obs
    mu_res = np.mean(perf_res, axis=1)
    sig_res = np.std(perf_res, axis=1)

    # Initial performances
    Perf = calc_perf_propagules(E, Xopt, ndim,
                                np.arange(M), Occ,
                                mu_res, sig_res,
                                mup, sigp,
                                sp_intercept, uIV,
                                rng_dyn)
    Perf[Occ < 0] = np.nan
    table = []
    x = np.array([np.sum(Occ == j) for j in range(J)])
    p = x/np.sum(x)
    dic = {"iteration": 0,
           "diversity": np.sum(x > 0),
           "shannon": np.exp(-np.sum(p[p > 0]*np.log(p[p > 0]))),
           "occupation": np.mean(Occ >= 0),
           "mean_performance": np.nanmean(Perf)}
    table.append(dic)

    # Iterations
    for t in tqdm(range(tmax)):

        # Death
        occupied = np.where(Occ >= 0)[0].astype("int")
        worst = []
        if mortality == "deterministic":
            # The worst x% die
            worst = occupied[np.argsort(Perf[occupied])[:int(M*death)]]
        elif mortality == "random":
            # x% die at random (just a test, not in the paper)
            worst = rng_dyn.choice(occupied, size=int(M*death))
        Occ[worst] = -1
        Perf[worst] = np.nan

        # Fecundity
        propagule_sp = []
        if fecundity == "fixed":
            # Fixed fecundity
            propagule_sp = np.array([j for j in range(J)
                                     for prop in range(nprop)
                                     if (Occ == j).any()])
        elif fecundity == "percapita":
            # Abundance-dependent fecundity
            propagule_sp = np.array([j for j in range(J)
                                     for prop in
                                     range(int(np.sum(Occ == j)*fecund))])
        if len(propagule_sp) == 0:
            break
        propagule_pos = rng_dyn.choice(np.where(Occ < 0)[0],
                                       size=len(propagule_sp), replace=True)
        propagule_perf = calc_perf_propagules(E, Xopt, ndim,
                                              propagule_pos, propagule_sp,
                                              mu_res, sig_res,
                                              mup, sigp,
                                              sp_intercept, uIV,
                                              rng_dyn)
        for pos in np.unique(propagule_pos):
            candidates = np.where(propagule_pos == pos)[0]
            best = candidates[np.argmax(propagule_perf[candidates])]
            Occ[pos] = propagule_sp[best]
            Perf[pos] = propagule_perf[best]

        # Summary outcomes
        x = np.array([np.sum(Occ == j) for j in range(J)])
        p = x/np.sum(x)
        dic = {"iteration": t + 1,
               "diversity": np.sum(x > 0),
               "shannon": np.exp(-np.sum(p[p > 0]*np.log(p[p > 0]))),
               "occupation": np.mean(Occ >= 0),
               "mean_performance": np.nanmean(Perf)}
        table.append(dic)

    return {"dynamics": pd.DataFrame(table), "community": Occ}


# =====================================================
# Function to plot results
# =====================================================

def plot_results(dist, mort, fec):
    """Plotting results"""
    # Load results
    f_in = f"outputs/results_{dist}_{mort}_{fec}.csv"
    df_exp = pd.read_csv(f_in)

    # Boxplots
    fig = plt.figure(figsize=(15, 10), dpi=300)
    fig.suptitle((f"MODEL | Distribution: {dist}, "
                  f"Mortality: {mort}, Fecundity: {fec}"),
                 fontsize=16)

    axes = plt.subplot(221)
    axes.set_title("m1: perf=dist(opt)_ndim and {intercept=0, uIV=0}")
    dm1 = df_exp.loc[(df_exp["intercept"] == 0) & (df_exp["uIV"] == 0)]
    sns.boxplot(x="ndim", y="diversity", data=dm1, color="white")
    sns.stripplot(x="ndim", y="diversity", data=dm1, hue="seed_dyn")
    axes.set_ylim(0, 21)

    axes = plt.subplot(222)
    axes.set_title("m2: perf=dist(opt)_ndim and {intercept=0, uIV=1}")
    dm1 = df_exp.loc[(df_exp["intercept"] == 0) & (df_exp["uIV"] == 1)]
    sns.boxplot(x="ndim", y="diversity", data=dm1, color="white")
    sns.stripplot(x="ndim", y="diversity", data=dm1, hue="seed_dyn")
    axes.set_ylim(0, 21)

    axes = plt.subplot(223)
    axes.set_title("m3: perf=dist(opt)_ndim and {intercept=1, uIV=0}")
    dm1 = df_exp.loc[(df_exp["intercept"] == 1) & (df_exp["uIV"] == 0)]
    sns.boxplot(x="ndim", y="diversity", data=dm1, color="white")
    sns.stripplot(x="ndim", y="diversity", data=dm1, hue="seed_dyn")
    axes.set_ylim(0, 21)

    axes = plt.subplot(224)
    axes.set_title("m4: perf=dist(opt)_ndim and {intercept=1, uIV=1}")
    dm1 = df_exp.loc[(df_exp["intercept"] == 1) & (df_exp["uIV"] == 1)]
    sns.boxplot(x="ndim", y="diversity", data=dm1, color="white")
    sns.stripplot(x="ndim", y="diversity", data=dm1, hue="seed_dyn")
    axes.set_ylim(0, 21)

    fig_out = f"outputs/results_{dist}_{mort}_{fec}.png"
    fig.savefig(fig_out)


# =====================================================
# Experiment
# =====================================================

# Model type with variable species parameters: intercept and variance
sp_intercept = [False, True]
sp_variance = [False, True]
sp_int_var = list(itertools.product(sp_intercept, sp_variance))

# Model type with variable demographic processes
dist_env = ["uniform", "normal"]
mortality = ["deterministic", "random"]
fecundity = ["fixed", "percapita"]
dist_mort_fec = list(itertools.product(dist_env, mortality, fecundity))

# Loop on demographic processes
for (dist, mort, fec) in dist_mort_fec:

    # Output file
    f_out = f"outputs/results_{dist}_{mort}_{fec}.csv"

    # Run only if not output file
    if not os.path.isfile(f_out):

        # Dataframe to store results
        df = pd.DataFrame(columns=["seed_init", "seed_dyn", "mortality",
                                   "fecundity", "intercept", "uIV", "ndim",
                                   "iteration", "diversity", "shannon",
                                   "occupation", "mean_perf"])

        # Message
        print((f"MODEL | Distribution: {dist}, "
               f"Mortality: {mort}, Fecundity: {fec}"))
        # Selection of dimensions to avoid iterating over all values of K
        ndim_sel = [0, 1, 3, 6, 9, 12, 14, 15]
        for i in ndim_sel:

            # Number of repetitions
            N_REP_INIT = 2
            N_REP_DYN = 2

            # Iterations
            TMAX = 10000

            # Seeds
            rng_exp = np.random.default_rng(1234)
            S_init = rng_exp.integers(10000, size=N_REP_INIT)
            S_dyn = rng_exp.integers(10000, size=N_REP_DYN)

            print(f"-- Dimension: {i}")

            # Loop on repetitions
            for j in range(N_REP_INIT):
                for k in range(N_REP_DYN):
                    # Loop on models including or not species parameters
                    for (sp_int, sp_var) in sp_int_var:
                        r = simulate_dynamics(ndim=i, sp_intercept=sp_int,
                                              uIV=sp_var,
                                              tmax=TMAX,
                                              mortality=mort, fecundity=fec,
                                              distrib_env=dist,
                                              seed_init=S_init[j],
                                              seed_dyn=S_dyn[k])
                        rr = r["dynamics"].iloc[TMAX].tolist()
                        ll = [S_init[j], S_dyn[k], mort,
                              fec, sp_int, sp_var, i] + rr
                        df.loc[len(df)] = ll

            # Saving results
            df.to_csv(f_out, index=False)

    # Plotting results
    plot_results(dist, mort, fec)
