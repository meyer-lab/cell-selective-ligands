"""
Figure 6.
"""
import numpy as np
import pandas as pd
import seaborn as sns
from .figureCommon import subplotLabel, getSetup
from .figure6 import optimizeDesign


def makeFigure():
    """ Make figure 6. """
    # Get list of axis objects
    ax, f = getSetup((16, 8), (3, 6))
    subplotLabel(ax)

    optimizeDesign(ax[0:6], [r"$R_1^{lo}R_2^{hi}$"], vrange=(0, 3), recFactor=100)
    optimizeDesign(ax[6:12], [r"$R_1^{hi}R_2^{hi}$"], vrange=(0, 1.5), recFactor=100)
    optimizeDesign(ax[12:18], [r"$R_1^{med}R_2^{med}$"], vrange=(0, 10), recFactor=100)

    return f
