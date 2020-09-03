"""
Sample from population to calculation population specificities.
"""

import numpy as np
from scipy.stats import multivariate_normal

from .model import polyfc, polyc


nsample = 200


cellPopulations = {
    r"$R_1^{lo}R_2^{lo}$": [2, 2, 0.5, 0.25, 45],
    r"$R_1^{med}R_2^{lo}$": [3, 2, 0.5, 0.25, 0],
    r"$R_1^{hi}R_2^{lo}$": [4, 2, 0.5, 0.25, 0],
    r"$R_1^{lo}R_2^{hi}$": [2, 4, 0.3, 0.6, 0],
    r"$R_1^{med}R_2^{hi}$": [3.0, 3.9, 0.5, 0.25, 45],
    r"$R_1^{hi}R_2^{med}$": [3.9, 3.0, 0.5, 0.25, 45],
    r"$R_1^{hi}R_2^{hi}$": [4, 4, 0.5, 0.25, 45],
    r"$R_1^{med}R_2^{med}$": [3.1, 3.1, 0.25, 1, 45],
}

def sigmapts(name, h=None):
    if h is None:
        h = np.sqrt(3)
    l = cellPopulations[name]
    x = np.array([l[0], l[1]])
    rot = np.array([[np.cos(np.deg2rad(l[4])), -np.sin(np.deg2rad(l[4]))], [np.sin(np.deg2rad(l[4])), np.cos(np.deg2rad(l[4]))]])
    srlamb = np.diag([l[2], l[3]]) #np.diag(np.sqrt([l[2], l[3]]))
    srcov = rot @ srlamb @ np.transpose(rot)
    return np.power(10, [x, x+h*srcov[:,0], x-h*srcov[:,0], x+h*srcov[:,1], x-h*srcov[:,1]])


def sigmaPop(name, L0, KxStar, f, LigC, Kav, quantity=0, h=None):
    return [polyfc(L0, KxStar, f, Rtot, LigC, Kav)[quantity] for Rtot in sigmapts(name, h = h)]

def sigmaPopC(name, L0, KxStar, Cplx, Ctheta, Kav, quantity=0, h=None):
    return [polyc(L0, KxStar, Rtot, Cplx, Ctheta, Kav)[quantity] for Rtot in sigmapts(name, h = h)]


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
