"""
Figure 4.
"""
from .figureCommon import subplotLabel, getSetup


def makeFigure():
    """ Make figure 4. """
    # Get list of axis objects
    ax, f = getSetup((7, 6), (3, 4))

    subplotLabel(ax)

    return f
