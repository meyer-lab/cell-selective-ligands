"""
Figure 5.
"""
from .figureCommon import subplotLabel, getSetup


def makeFigure():
    """ Make figure 5. """
    # Get list of axis objects
    ax, f = getSetup((7, 6), (3, 4))

    subplotLabel(ax)

    return f