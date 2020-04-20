"""
Figure 4. Mixtures for enhanced targeting.
"""

import numpy as np
from .figureCommon import subplotLabel, getSetup, PlotCellPops, popCompare
from ..imports import getPopDict

ligConc = 1.0e-9
KxStarP = 1.0e-11
val = 1.0


def makeFigure():
    """ Make figure 4. """
    # Get list of axis objects
    ax, f = getSetup((7, 4), (2, 3))

    subplotLabel(ax)

    affinities = np.array([[10e8, 10e1], [10e1, 10e8]])
    _, populationsdf = getPopDict()
    PlotCellPops(ax[0], populationsdf)
    popCompare(ax[1], ["Pop3", "Pop2"], populationsdf, "Mix", Kav=np.array([[10e8, 10e1], [10e1, 10e8]]))
    popCompare(ax[2], ["Pop5", "Pop3", "Pop4"], populationsdf, "Mix", Kav=np.array([[10e8, 10e1], [10e1, 10e8]]))
    popCompare(ax[3], ["Pop6", "Pop3", "Pop4"], populationsdf, "Mix", Kav=np.array([[10e8, 10e1], [10e1, 10e8]]))
    popCompare(ax[4], ["Pop7", "Pop3", "Pop4"], populationsdf, "Mix", Kav=np.array([[10e8, 10e1], [10e1, 10e8]]))
    popCompare(ax[5], ["Pop8", "Pop5", "Pop6"], populationsdf, "Mix", Kav=np.array([[10e8, 10e1], [10e1, 10e8]]))

    return f
