"""
Figure 1. Introduce the model system.
"""
from .figureCommon import subplotLabel, getSetup


def makeFigure():
    """ Make figure 1. """
    # Get list of axis objects
    ax, f = getSetup((7, 6), (3, 4))

    subplotLabel(ax[0], 'A')

    return f
