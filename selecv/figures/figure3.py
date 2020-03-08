"""
Figure 3.
"""
from .figureCommon import subplotLabel, getSetup


def makeFigure():
    """ Make figure 3. """
    # Get list of axis objects
    ax, f = getSetup((7, 6), (3, 4))

    subplotLabel(ax)

    return f
