"""Theoretical model."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm


def set_initial_conditions(J=20, K=15, M=625,
                           nprop=10, seed_init=1234):
    """"Set initial conditions

    :param J: nb species.
    :param K: nb env dim.
    :param M: nb sites.
    :param nprop: nb of initial propagules per species.
    :param seed_init: seed for initila conditions.

    :return: A dictionary
    """

    # RNG
    rng_init = np.random.default_rng(seed_init)

    # Initial parameters
    E = rng_init.uniform(0, 1, (K, M))  # environment
    Xopt = rng_init.uniform(0, 1, (J, K))  # species optima

    # Initial occupancy
    Occ = np.zeros(M, dtype='int')-1
    initial_pos = rng_init.choice(
        np.where(Occ < 0)[0], size=J*nprop, replace=False)
    for j in range(J):
        Occ[initial_pos[j*nprop:(j+1)*nprop]] = j

    return {"Occ": Occ, "E": E, "Xopt": Xopt}


def calc_perf_propagules(E, Xopt, ndim, propagule_pos, propagule_sp,
                         mu_res, sig_res, mup, sigp,
                         sp_intercept=True, uIV=False,
                         rng_dyn=np.random.default_rng(1234)):
    """Computing propagule performances.

    The performance of a propagule depends on both the environment and
    the species optima. Residuals associated with the missing
    dimensions can be added to the performance computation. Per
    species residuals have a mean and standard deviation. Either the
    mean or standard deviation of the species residuals or both can be
    taken into account to compute species (and thus propagule)
    performances.

    :param E: K x M np.array. Environment with K dimensions on M sites.
    :param Xopt: J x K np.array. Species optima for the K dimensions.
    :param ndim: Number of observed environmental dimensions.
    :param propagule_pos: Sites reached by propagules.
    :param propagule_sp: Species for each propagule.
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

    return (perf-mup)/sigp


def simulate_dynamics(J=20, K=15, M=625, seed_init=1234,
                      death=0.01, nprop=10, fecund=0.5,
                      mortality="deterministic", fecundity="percapita",
                      ndim=1, sp_intercept=False, uIV=False,
                      tmax=10000, seed_dyn=1234):
    """Simulate dynamics

    :param J: nb species.
    :param K: nb env dim.
    :param M: nb sites.
    :param seed_init: Random seed for initial conditions.

    :param death: Fraction of dead individuals per turn.
    :param nprop:  Nb of propagules for fixed fecundity.
    :param fecund: Nb of propagules per capita (for
        abundance-dependent fecundity).

    :param mortality: Either "deterministic" or "random".
    :param percapita: Either "fixed" or "percapita".

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
    init = set_initial_conditions(J, K, M, nprop, seed_init)
    Occ = np.copy(init["Occ"])
    E = init["E"]
    Xopt = init["Xopt"]
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
    dic = {'diversity': np.sum(x > 0),
           'shannon': np.exp(-np.sum(p[p > 0]*np.log(p[p > 0]))),
           'occupation': np.mean(Occ >= 0),
           'mean_performance': np.nanmean(Perf)}
    table.append(dic)

    for t in tqdm(range(tmax)):

        # Death
        occupied = np.where(Occ >= 0)[0].astype('int')
        worst = []
        if mortality == 'deterministic':
            # The worst x% die
            worst = occupied[np.argsort(Perf[occupied])[:int(M*death)]]
        elif mortality == 'random':
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
        dic = {'diversity': np.sum(x > 0),
               'shannon': np.exp(-np.sum(p[p > 0]*np.log(p[p > 0]))),
               'occupation': np.mean(Occ >= 0),
               'mean_performance': np.nanmean(Perf)}
        table.append(dic)

    return {"dynamics": pd.DataFrame(table), "community": Occ}


# perf = 0 --> neutral: all species coexist
r = simulate_dynamics(ndim=0, sp_intercept=False, uIV=False)
r["dynamics"]

# perf = mu --> 1 species dominates
r = simulate_dynamics(ndim=0, sp_intercept=True, uIV=False)
r["dynamics"]

# perf = mu + delta --> a higher number of species coexist (5)
r = simulate_dynamics(ndim=0, sp_intercept=True, uIV=True)
r["dynamics"]

# perf = delta, delta ~ N(0, \sigma^2) --> neutral
r = simulate_dynamics(ndim=0, sp_intercept=False, uIV=True)
r["dynamics"]

r = simulate_dynamics(ndim=1, sp_intercept=True, uIV=False)
r["dynamics"]

r = simulate_dynamics(ndim=15, sp_intercept=False, uIV=False)
r["dynamics"]

r = simulate_dynamics(ndim=1, sp_intercept=False, uIV=False)
r["dynamics"]


########################
# RESULTS
results = r["dynamics"]
plt.figure()
plt.subplot(221), plt.title('Occupation')
plt.plot(results['occupation'])
plt.subplot(222), plt.title('Diversity')
plt.plot(results['diversity'], label='number')
plt.plot(results['shannon'], label='shannon')
plt.legend()
plt.subplot(223), plt.title('Performance')
plt.plot(results['mean_performance'])
plt.subplot(224), plt.title('Final landscape occupation')
x = r["community"].copy().astype('float')
x[x < 0] = np.nan
plt.colorbar(plt.imshow(x.reshape((25, 25)), cmap='jet'))
plt.show()
