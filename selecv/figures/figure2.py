"""
Figure 2. Explore selectivity vs. affinity.
"""

import numpy as np
from .figureCommon import subplotLabel, getSetup, popCompare, heatmap
from ..imports import getPopDict

ligConc = np.array([10e-9])
KxStarP = 10e-11
val = 1.0


def makeFigure():
    """ Make figure 2. """
    # Get list of axis objects
    ax, f = getSetup((12, 14), (5, 3))
    _, populationsdf = getPopDict()
    subplotLabel(ax, [0] + list(range(9, 13)))
    ax[13].axis("off")
    ax[14].axis("off")
    affinity(f, ax[0:9], 1e-9, 10 ** -10, [1.0], ff=1, vmin=-1, vmax=5.5)

    popCompare(ax[9], [r"$R_1^{hi}R_2^{lo}$", r"$R_1^{med}R_2^{lo}$"], populationsdf, "Aff", Kav=[5, 7])
    popCompare(ax[10], [r"$R_1^{med}R_2^{hi}$", r"$R_1^{hi}R_2^{med}$"], populationsdf, "Aff", Kav=[5, 7])
    popCompare(ax[11], [r"$R_1^{hi}R_2^{hi}$", r"$R_1^{med}R_2^{med}$"], populationsdf, "Aff", Kav=[5, 7])
    popCompare(ax[12], [r"$R_1^{hi}R_2^{lo}$", r"$R_1^{lo}R_2^{hi}$"], populationsdf, "Aff", Kav=[5, 7])

    return f



def affinity(fig, axs, L0, KxStar, Comp, ff=None, Cplx=None, vmin=-2, vmax=4):
    nAffPts = 3
    affRange = (5., 7.)
    affScan = np.logspace(affRange[0], affRange[1], nAffPts)
    for i1, aff1 in enumerate(affScan):
        for i2, aff2 in enumerate(np.flip(affScan)):
            cbar = False
            if i2 * nAffPts + i1 in [2, 5, 8]:
                cbar = True
            heatmap(axs[i2 * nAffPts + i1], L0, KxStar, [[aff1, aff2]], Comp, f=ff, Cplx=Cplx, vrange=(vmin, vmax), cbar=cbar)
            axs[i2 * nAffPts + i1].set(xlabel="Receptor 1 Abundance (#/cell)", ylabel='Receptor 2 Abundance (#/cell)')
            plt.plot([3.3, 3.7], [2, 2], color="w", marker=2)
            plt.text(3.5, 2.1, "b", size='large', color='white', weight='semibold', horizontalalignment='center', verticalalignment='center')
            plt.plot([3.3, 3.7], [3.6, 3.2], color="w", marker=2)
            plt.text(3.4, 3.63, "c", size='large', color='white', weight='semibold', horizontalalignment='center', verticalalignment='center')
            plt.plot([3.3, 3.8], [3.2, 3.7], color="w", marker=2)
            plt.text(3.7, 3.85, "d", size='large', color='white', weight='semibold', horizontalalignment='center', verticalalignment='center')
            plt.plot([2, 3.7], [3.5, 2.2], color="w", marker=2)
            plt.text(2.3, 3.5, "e", size='large', color='white', weight='semibold', horizontalalignment='center', verticalalignment='center')
            axs[i2 * nAffPts + i1].set_title("$K_1$ = {:.1e}".format(aff1) +  "M$^{-1}$ $K_2$ = " + "{:.1e}".format(aff2) +  "M$^{-1}$")
    return fig

