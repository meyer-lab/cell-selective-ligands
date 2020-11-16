"""
Figure 2. Explore selectivity vs. affinity.
"""

import numpy as np
from matplotlib import pyplot as plt
from .figureCommon import subplotLabel, getSetup, popCompare, heatmap
from .figure1 import demoPopulations

ligConc = np.array([1e-9])
KxStarP = 1e-10
val = 1.0


def makeFigure():
    """ Make figure 2. """
    # Get list of axis objects
    ax, f = getSetup((12, 14), (5, 3))
    subplotLabel(ax, [0] + list(range(10, 14)))
    ax[14].axis("off")
    affinity(f, ax[0:9], 1e-9, 1e-10, [1.0], ff=1, vmin=-1, vmax=5.5)

    showPopulations(ax[9])
    popCompare(ax[10], [r"$R_1^{hi}R_2^{lo}$", r"$R_1^{lo}R_2^{hi}$"], "Aff", Kav=[5, 7])
    popCompare(ax[11], [r"$R_1^{med}R_2^{hi}$", r"$R_1^{hi}R_2^{med}$"], "Aff", Kav=[5, 7])
    popCompare(ax[12], [r"$R_1^{hi}R_2^{lo}$", r"$R_1^{med}R_2^{lo}$"], "Aff", Kav=[5, 7])
    popCompare(ax[13], [r"$R_1^{hi}R_2^{hi}$", r"$R_1^{med}R_2^{med}$"], "Aff", Kav=[5, 7])

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
            axs[i2 * nAffPts + i1].set_title(r"$K_{d1}$" + " = {:d}".format(int(1e9 * (1 / aff1))) + " nM, " + r"$K_{d2}$" + " = {:d}".format(int(1e9 * (1 / aff2))) + " nM")
    return fig


def showPopulations(ax):
    demoPopulations(ax)
    ax.plot([10**2, 10**3.7], [10**3.5, 10**2.2], color="w", marker=2)
    ax.text(10**2.3, 10**3.5, "b", size='large', color='white', weight='semibold', horizontalalignment='center',
             verticalalignment='center')
    ax.plot([10**3.3, 10**3.7], [10**3.6, 10**3.2], color="w", marker=2)
    ax.text(10**3.4, 10**3.63, "c", size='large', color='white', weight='semibold', horizontalalignment='center',
             verticalalignment='center')
    ax.plot([10**3.3, 10**3.7], [10**2, 10**2], color="w", marker=2)
    ax.text(10**3.5, 10**2.1, "d", size='large', color='white', weight='semibold', horizontalalignment='center',
             verticalalignment='center')
    ax.plot([10**3.3, 10**3.8], [10**3.2, 10**3.7], color="w", marker=2)
    ax.text(10**3.7, 10**3.85, "e", size='large', color='white', weight='semibold', horizontalalignment='center',
             verticalalignment='center')
    ax.set_title("Population pairs to be compared in b, c, d, e")
