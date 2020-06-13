"""
Figure 4. Mixtures for enhanced targeting.
"""

import numpy as np
from .figureCommon import subplotLabel, getSetup, PlotCellPops, popCompare, mixture
from ..imports import getPopDict

ligConc = np.array([10e-9])
KxStarP = 10e-11
val = 1.0


def makeFigure():
    """ Make figure 4. """
    # Get list of axis objects
    ax, f = getSetup((7, 8), (4, 3))
    mixture(f, ax, 1e-9, 10 ** -10, ff=1, vmin=-2, vmax=3.5)
    subplotLabel(ax, [0] + list(range(5, 10)))

    affinities = np.array([[10e8, 10e1], [10e1, 10e8]])
    _, populationsdf = getPopDict()
    ax[10].axis("off")
    ax[11].axis("off")

    popCompare(ax[5], ["Pop3", "Pop2"], populationsdf, "Mix", Kav=np.array([[10e5, 10e4], [10e4, 10e5]]))
    popCompare(ax[6], ["Pop7", "Pop3", "Pop4"], populationsdf, "Mix", Kav=np.array([[10e5, 10e4], [10e4, 10e5]]))
    popCompare(ax[7], ["Pop8", "Pop5", "Pop6"], populationsdf, "Mix", Kav=np.array([[10e5, 10e4], [10e4, 10e5]]))
    popCompare(ax[8], ["Pop5", "Pop3", "Pop4"], populationsdf, "Mix", Kav=np.array([[10e5, 10e4], [10e4, 10e5]]))
    popCompare(ax[9], ["Pop6", "Pop3", "Pop4"], populationsdf, "Mix", Kav=np.array([[10e5, 10e4], [10e4, 10e5]]))

    return f
