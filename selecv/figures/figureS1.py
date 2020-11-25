"""
Figure S1. Plotting Csizmar reimplementation
"""

import numpy as np
from .figureCommon import subplotLabel, getSetup
from ..csizmar import fit_slope, discrim2, fitfunc, xeno


def makeFigure():
    """ Make figure 4. """
    # Get list of axis objects
    ax, f = getSetup((9, 3), (1, 3))
    subplotLabel(ax)
    fit = fitfunc()
    FitKX, FitSlopeC5, FitSlopeB22, Kav, Fitabund = np.exp(fit[0]), fit[1], fit[2], [[np.exp(fit[3])], [np.exp(fit[4])], [0]], np.exp(fit[5])

    # FitKX = 1.1140699176510234e-12
    # FitSlopeC5 = 0.05086117915555084
    # FitSlopeB22 = 0.032171363604266194
    # Kav = [[578466.7589450455], [308358.5526742298], [0]]
    # Fitabund = 3805378.019178565

    fit_slope(ax[0], FitKX, FitSlopeC5, FitSlopeB22, Kav, Fitabund)
    discrim2(ax[1], FitKX, FitSlopeC5, FitSlopeB22, Kav)
    xeno(ax[2], FitKX, Kav)

    return f
