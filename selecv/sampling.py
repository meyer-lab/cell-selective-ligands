"""
Sample from population to calculation population specificities.
"""

import numpy as np
from scipy.stats import multivariate_normal

from .model import polyfc, polyc


nsample = 200


def sampleSpec(L0, KxStar, f, RtotMeans, RtotCovs, LigC, Kav):
    """
    Sample the specificity between populations.
    RtotMeans: Tuple of receptor population expression means.
    """
    assert len(RtotMeans) == len(RtotCovs)

    quantsNum = samplePop(L0, KxStar, f, RtotMeans[0], RtotCovs[0], LigC, Kav)
    quants = np.zeros([len(RtotMeans) - 1, len(quantsNum)])

    for ii in range(0, len(RtotCovs) - 1):
        quants[ii, :] = quantsNum / samplePop(L0, KxStar, f, RtotMeans[ii + 1], RtotCovs[ii + 1], LigC, Kav)

    quants = np.min(quants, axis=0)

    return np.quantile(quants, [0.1, 0.5, 0.9])


def samplePop(L0, KxStar, f, RtotMeans, RtotCovs, LigC, Kav, quantity=0):
    """
    Sample the binding for one population.
    Note that receptor expression is given in log10 units
    Quantity: (0) Lbound, (1) Rbound
    """
    quants = np.empty(nsample)
    pop = np.power(10.0, multivariate_normal.rvs(mean=RtotMeans, cov=RtotCovs, size=nsample))

    for ii in range(nsample):
        quants[ii] = polyfc(L0, KxStar, f, pop[ii, :], LigC, Kav)[quantity]

    return quants


def sampleSpecC(L0, KxStar, RtotMeans, RtotCovs, LigCplx, Ctheta, Kav):
    """
    Sample the specificity between populations.
    RtotMeans: Tuple of receptor population expression means.
    """

    quantsNum = samplePopC(L0, KxStar, RtotMeans[0], RtotCovs[0], LigCplx, Ctheta, Kav)
    quants = np.zeros([len(RtotMeans) - 1, len(quantsNum)])

    for ii in range(0, len(RtotCovs) - 1):
        quants[ii, :] = quantsNum / samplePopC(L0, KxStar, RtotMeans[ii + 1], RtotCovs[ii + 1], LigCplx, Ctheta, Kav)

    quants = np.min(quants, axis=0)

    return np.quantile(quants, [0.1, 0.5, 0.9])


def samplePopC(L0, KxStar, RtotMeans, RtotCovs, LigCplx, Ctheta, Kav, quantity=0):
    """
    Sample the binding for one population.
    Note that receptor expression is given in log10 units
    Quantity: (0) Lbound, (1) Rbound
    """
    quants = np.empty(nsample)
    pop = np.power(10.0, multivariate_normal.rvs(mean=RtotMeans, cov=RtotCovs, size=nsample))

    for ii in range(nsample):
        quants[ii] = polyc(L0, KxStar, pop[ii, :], LigCplx, Ctheta, Kav)[quantity]

    return quants

