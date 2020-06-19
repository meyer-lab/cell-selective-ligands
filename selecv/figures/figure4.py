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
    ax, f = getSetup((12, 14), (4, 3))
    mixture(f, ax, 1e-9, 10 ** -10, ff=1, vmin=-2, vmax=3.5)
    subplotLabel(ax, [0] + list(range(5, 10)))

    affinities = np.array([[10e8, 10e1], [10e1, 10e8]])
    _, populationsdf = getPopDict()
    ax[10].axis("off")
    ax[11].axis("off")

    popCompare(ax[5], [r"$R_1^{hi}R_2^{lo}$", r"$R_1^{med}R_2^{lo}$"], populationsdf, "Mix", Kav=np.array([[10e5, 10e4], [10e4, 10e5]]))
    popCompare(ax[6], [r"$R_1^{hi}R_2^{hi}$", r"$R_1^{hi}R_2^{lo}$", r"$R_1^{lo}R_2^{hi}$"], populationsdf, "Mix", Kav=np.array([[10e5, 10e4], [10e4, 10e5]]))
    popCompare(ax[7], [r"$R_1^{med}R_2^{med}$", r"$R_1^{med}R_2^{hi}$", r"$R_1^{hi}R_2^{med}$"], populationsdf, "Mix", Kav=np.array([[10e5, 10e4], [10e4, 10e5]]))
    popCompare(ax[8], [r"$R_1^{med}R_2^{hi}$", r"$R_1^{hi}R_2^{lo}$", r"$R_1^{lo}R_2^{hi}$"], populationsdf, "Mix", Kav=np.array([[10e5, 10e4], [10e4, 10e5]]))
    popCompare(ax[9], [r"$R_1^{hi}R_2^{med}$", r"$R_1^{hi}R_2^{lo}$", r"$R_1^{lo}R_2^{hi}$"], populationsdf, "Mix", Kav=np.array([[10e5, 10e4], [10e4, 10e5]]))

    return f
