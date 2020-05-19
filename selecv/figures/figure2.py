"""
Figure 2. Explore selectivity vs. affinity.
"""

import numpy as np
import seaborn as sns
import pandas as pds
from .figureCommon import subplotLabel, getSetup, PlotCellPops, popCompare
from ..imports import getPopDict

ligConc = np.array([10e-9])
KxStarP = 10e-11
val = 1.0


def makeFigure():
    """ Make figure 2. """
    # Get list of axis objects
    ax, f = getSetup((7, 4), (2, 3))
    _, populationsdf = getPopDict()
    PlotCellPops(ax[0], populationsdf)
    ax[5].axis("off")

    popCompare(ax[1], ["Pop3", "Pop2"], populationsdf, "Aff", Kav=[4, 6])
    popCompare(ax[2], ["Pop5", "Pop6"], populationsdf, "Aff", Kav=[4, 6])
    popCompare(ax[3], ["Pop7", "Pop8"], populationsdf, "Aff", Kav=[4, 6])
    popCompare(ax[4], ["Pop3", "Pop4"], populationsdf, "Aff", Kav=[4, 6])
    # popCompare(ax[5], ["Pop7", "Pop8"], populationsdf, "Aff", Kav=[4, 6])

    subplotLabel(ax)

    return f
