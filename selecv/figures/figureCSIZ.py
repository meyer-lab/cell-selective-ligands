"""
Figure 6. PolyC vs PolyFc.
"""

import numpy as np
from .figureCommon import subplotLabel, getSetup, PlotCellPops
from ..imports import getPopDict
from ..sampling import sampleSpec, sampleSpecC
from ..csizmar import model_predict, fit_slope, discrim, discrim2, fitfunc, xeno

ligConc = np.array([10e-10])
# 10**-12.734, KxStar_C5 = 10**-14.693, KxStar = 10**-14.714
KxStarP = KxStar = 10**-12.734


def makeFigure():
    """ Make figure 4. """
    # Get list of axis objects
    ax, f = getSetup((7, 6), (2, 3))
    subplotLabel(ax)
    fit = fitfunc()
    FitKX, FitSlopeC5, FitSlopeB22, Kav, Fitabund = np.exp(fit[0]), fit[1], fit[2], [[np.exp(fit[3])], [np.exp(fit[4])], [0]], np.exp(fit[5])

    fit_slope(ax[0], FitKX, FitSlopeC5, FitSlopeB22, Kav, Fitabund)
    discrim2(ax[1], FitKX, FitSlopeC5, FitSlopeB22, Kav)
    xeno(ax[2], FitKX, Kav)

    return f