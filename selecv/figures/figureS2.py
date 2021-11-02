"""
Figure S2. Plotting Csizmar reimplementation
"""

import numpy as np
from .figureCommon import subplotLabel, getSetup
from ..csizmar import fit_slope, discrim2, fitfunc, xeno


def makeFigure():
    # Get list of axis objects
    ax, f = getSetup((12, 3), (1, 4))
    subplotLabel(ax)
    fsize = 9
    fit = fitfunc()
    FitKX, FitSlopeC5, FitSlopeB22, Kav, Fitabund = np.exp(fit[0]), fit[1], fit[2], [[np.exp(fit[3])], [np.exp(fit[4])], [0]], np.exp(fit[5])

    fit_slope(ax[0], FitKX, FitSlopeC5, FitSlopeB22, Kav, Fitabund)
    discrim2(ax[1], FitKX, FitSlopeC5, FitSlopeB22, Kav)
    xeno(ax[2], FitKX, Kav)

    # Fit without affinity
    fitnA = fitfunc(fitAff=False)
    FitKXnA, FitSlopeC5nA, FitSlopeB22nA, KavnA, FitabundnA = np.exp(fitnA[0]), fitnA[1], fitnA[2], [[np.exp(fitnA[3])], [np.exp(fitnA[4])], [0]], np.exp(fitnA[5])

    fit_slope(ax[3], FitKXnA, FitSlopeC5nA, FitSlopeB22nA, KavnA, FitabundnA)

    for subax in ax:
        subax.set_xticklabels(subax.get_xticklabels(), fontsize=fsize)
        subax.set_yticklabels(subax.get_yticklabels(), fontsize=fsize)
        subax.set_xlabel(subax.get_xlabel(), fontsize=fsize)
        subax.set_ylabel(subax.get_ylabel(), fontsize=fsize)
        subax.set_title(subax.get_title(), fontsize=fsize)

    return f
