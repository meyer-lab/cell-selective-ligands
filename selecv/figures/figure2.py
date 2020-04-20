"""
Figure 2. Explore selectivity vs. affinity.
"""

import numpy as np
import seaborn as sns
import pandas as pds
from .figureCommon import subplotLabel, getSetup, PlotCellPops, popCompare
from ..imports import getPopDict

ligConc = 1.0e-9
KxStarP = 1.0e-11
val = 1.0


def makeFigure():
    """ Make figure 2. """
    # Get list of axis objects
    ax, f = getSetup((7, 4), (2, 3))
    _, populationsdf = getPopDict()
    PlotCellPops(ax[0], populationsdf)

    popCompare(ax[1], ["Pop3", "Pop2"], populationsdf, "Aff", Kav=[4, 9])
    popCompare(ax[2], ["Pop5", "Pop3"], populationsdf, "Aff", Kav=[4, 9])
    popCompare(ax[3], ["Pop7", "Pop4"], populationsdf, "Aff", Kav=[4, 9])
    popCompare(ax[4], ["Pop5", "Pop6"], populationsdf, "Aff", Kav=[4, 9])
    popCompare(ax[5], ["Pop7", "Pop8"], populationsdf, "Aff", Kav=[4, 9])

    subplotLabel(ax)

    return f
