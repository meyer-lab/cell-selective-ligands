"""
Figure 6. PolyC vs PolyFc.
"""

import numpy as np
from .figureCommon import subplotLabel, getSetup
from ..csizmar import fit_slope, discrim2, fitfunc, xeno


def makeFigure():
    """ Make figure 4. """
    # Get list of axis objects
    ax, f = getSetup((6, 3), (1, 2))
    subplotLabel(ax)
    fit = fitfunc()
    FitKX, FitSlopeC5, FitSlopeB22, Kav, Fitabund = np.exp(fit[0]), fit[1], fit[2], [[np.exp(fit[3])], [np.exp(fit[4])], [0]], np.exp(fit[5])
    valencies = np.array([fit[6], fit[7], fit[8], fit[9]])

    fit_slope(ax[0], FitKX, FitSlopeC5, FitSlopeB22, Kav, Fitabund)
    discrim2(ax[1], FitKX, FitSlopeC5, FitSlopeB22, Kav)

    return f