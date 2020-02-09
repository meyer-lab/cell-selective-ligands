"""
Figure 2. Explore selectivity vs. affinity.
"""
from .figureCommon import subplotLabel, getSetup


def makeFigure():
    """ Make figure 2. """
    # Get list of axis objects
    ax, f = getSetup((7, 6), (3, 4))

    subplotLabel(ax[0], 'A')

    return f
