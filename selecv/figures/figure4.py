"""
Figure 4. Mixtures for enhanced targeting.
"""

import numpy as np
from .figureCommon import subplotLabel, getSetup, PlotCellPops, popCompare
from ..imports import getPopDict

ligConc = np.array([10e-9])
KxStarP = 10e-11
val = 1.0


def makeFigure():
    """ Make figure 4. """
    # Get list of axis objects
    ax, f = getSetup((7, 4), (2, 3))

    subplotLabel(ax)

    affinities = np.array([[10e9, 10e1], [10e1, 10e9]])
    _, populationsdf = getPopDict()
    PlotCellPops(ax[0], populationsdf)
    popCompare(ax[1], ["Pop3", "Pop2"], populationsdf, "Mix", Kav=np.array([[10e5, 10e4], [10e4, 10e5]]))
    popCompare(ax[2], ["Pop7", "Pop3", "Pop4"], populationsdf, "Mix", Kav=np.array([[10e5, 10e4], [10e4, 10e5]]))
    popCompare(ax[3], ["Pop8", "Pop5", "Pop6"], populationsdf, "Mix", Kav=np.array([[10e5, 10e4], [10e4, 10e5]]))
    popCompare(ax[4], ["Pop5", "Pop3", "Pop4"], populationsdf, "Mix", Kav=np.array([[10e5, 10e4], [10e4, 10e5]]))
    popCompare(ax[5], ["Pop6", "Pop3", "Pop4"], populationsdf, "Mix", Kav=np.array([[10e5, 10e4], [10e4, 10e5]]))

    return f
