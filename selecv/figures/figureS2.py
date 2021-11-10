"""
Figure S2. Plotting Csizmar reimplementation
"""

import numpy as np
from .figureCommon import subplotLabel, getSetup
from ..csizmar import fit_slope, discrim2, fitfunc, xeno


def makeFigure():
    # Get list of axis objects
    ax, f = getSetup((9, 3), (1, 3))
    subplotLabel(ax)
    fit = fitfunc()
    FitKX, FitSlopeC5, FitSlopeB22, Fitabund, valencies = np.exp(fit[0]), fit[1], fit[2], np.exp(fit[3]), fit[4:8]
    fit_slope(ax[0], FitKX, FitSlopeC5, FitSlopeB22, Fitabund, valencies)
    discrim2(ax[1], FitKX, FitSlopeC5, FitSlopeB22, valencies[2:4])
    xeno(ax[2], FitKX, [valencies[3], valencies[2], valencies[3], valencies[2]])

    return f
