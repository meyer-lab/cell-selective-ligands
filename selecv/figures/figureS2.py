"""
Figure S2. Plotting Csizmar reimplementation
"""

import numpy as np
from .figureCommon import subplotLabel, setFontSize, getSetup
from ..csizmar import fit_slope, discrim2, fitfunc, xeno


def makeFigure():
    # Get list of axis objects
    ax, f = getSetup((12, 3), (1, 4))
    subplotLabel(ax)
    fit = fitfunc()
    FitKX, FitSlopeC5, FitSlopeB22, Kav, Fitabund, valencies = np.exp(fit[0]), fit[1], fit[2], [[np.exp(fit[3])], [np.exp(fit[4])], [0]], np.exp(fit[5]), fit[6:10]
    fit_slope(ax[0], FitKX, FitSlopeC5, FitSlopeB22, Kav, Fitabund, valencies)
    discrim2(ax[1], FitKX, FitSlopeC5, FitSlopeB22, Kav, valencies[2:4])
    xeno(ax[2], FitKX, Kav, [valencies[3], valencies[2], valencies[3], valencies[2]])

    # Fit without affinity
    fitnA = fitfunc(fitAff=False)
    FitKXnA, FitSlopeC5nA, FitSlopeB22nA, KavnA, FitabundnA, valenciesnA = np.exp(fitnA[0]), fitnA[1], fitnA[2], [[np.exp(fitnA[3])], [np.exp(fitnA[4])], [0]], np.exp(fitnA[5]), fitnA[6:10]

    fit_slope(ax[3], FitKXnA, FitSlopeC5nA, FitSlopeB22nA, KavnA, FitabundnA, valenciesnA)

    #setFontSize(ax, 9)
    return f
