import numpy as np
from scipy.stats import multivariate_normal

from .model import polyfc


def sampleSpec(L0, KxStar, f, RtotMeans, RtotCovs, IgGC, Kav, nsample=100):
    """
    Sample the specificity between two populations.
    RtotMeans: Tuple of receptor population expression means.
    """
    assert len(RtotMeans) == 2
    assert len(RtotCovs) == 2

    quants = samplePop(L0, KxStar, f, RtotMeans[0], RtotCovs[0], IgGC, Kav, nsample)
    quants /= samplePop(L0, KxStar, f, RtotMeans[1], RtotCovs[1], IgGC, Kav, nsample)

    return np.mean(quants), np.std(quants)


def samplePop(L0, KxStar, f, RtotMeans, RtotCovs, IgGC, Kav, nsample=100):
    """
    Sample the binding for one population.
    Note that receptor expression is given in log10 units
    """
    quants = np.empty(nsample)
    pop = np.power(10.0, multivariate_normal.rvs(mean=RtotMeans, cov=RtotCovs, size=nsample))

    for ii in range(nsample):
        quants[ii] = polyfc(L0, KxStar, f, pop[ii, :], IgGC, Kav)

    return quants
