"""
Figure 2. Explore selectivity vs. affinity.
"""

import numpy as np
import seaborn as sns
import pandas as pds
from .figureCommon import subplotLabel, getSetup, PlotCellPops, popCompare, affinity
from ..imports import getPopDict

ligConc = np.array([10e-9])
KxStarP = 10e-11
val = 1.0


def makeFigure():
    """ Make figure 2. """
    # Get list of axis objects
    ax, f = getSetup((12, 14), (5, 3))
    _, populationsdf = getPopDict()
    subplotLabel(ax, [0] + list(range(9, 13))
    ax[13].axis("off")
    ax[14].axis("off")
    affinity(f, ax[0:9], 1e-9, 10 ** -10, [1.0], ff=1, vmin=-1, vmax=5.5)

    popCompare(ax[9], ["Pop3", "Pop2"], populationsdf, "Aff", Kav=[4, 6])
    popCompare(ax[10], ["Pop5", "Pop6"], populationsdf, "Aff", Kav=[4, 6])
    popCompare(ax[11], ["Pop7", "Pop8"], populationsdf, "Aff", Kav=[4, 6])
    popCompare(ax[12], ["Pop3", "Pop4"], populationsdf, "Aff", Kav=[4, 6])



    return f
