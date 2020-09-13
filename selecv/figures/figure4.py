"""
Figure 4. Mixtures for enhanced targeting.
"""

import numpy as np
from .figureCommon import subplotLabel, getSetup, popCompare, heatmap
from ..imports import getPopDict

ligConc = np.array([1e-8])
KxStarP = 1e-10
val = 1.0


def makeFigure():
    """ Make figure 4. """
    # Get list of axis objects
    ax, f = getSetup((12, 12), (3, 3))
    mixture(f, ax, 1e-9, 10 ** -10, ff=1, vmin=-2, vmax=3.5)
    subplotLabel(ax, [0] + list(range(5, 9)))

    _, populationsdf = getPopDict()

    popCompare(ax[5], [r"$R_1^{hi}R_2^{lo}$", r"$R_1^{med}R_2^{lo}$"], populationsdf, "Mix", Kav=np.array([[1e6, 1e5], [1e5, 1e6]]))
    popCompare(ax[6], [r"$R_1^{hi}R_2^{hi}$", r"$R_1^{hi}R_2^{lo}$", r"$R_1^{lo}R_2^{hi}$"], populationsdf, "Mix", Kav=np.array([[1e6, 1e5], [1e5, 1e6]]))
    popCompare(ax[7], [r"$R_1^{med}R_2^{med}$", r"$R_1^{med}R_2^{hi}$", r"$R_1^{hi}R_2^{med}$"], populationsdf, "Mix", Kav=np.array([[1e6, 1e5], [1e5, 1e6]]))
    popCompare(ax[8], [r"$R_1^{med}R_2^{hi}$", r"$R_1^{hi}R_2^{lo}$", r"$R_1^{lo}R_2^{hi}$"], populationsdf, "Mix", Kav=np.array([[1e6, 1e5], [1e5, 1e6]]))

    return f


def mixture(fig, axs, L0, KxStar, Kav=[[1e6, 1e5], [1e5, 1e6]], ff=5, vmin=-2, vmax=4):
    comps = [0.0, 0.2, 0.5, 0.8, 1.0]

    for i, comp in enumerate(comps):
        cbar = False
        if i in [2, 4]:
            cbar = True
        heatmap(axs[i], L0, KxStar, Kav, [comp, 1 - comp], f=ff, Cplx=None, vrange=(vmin, vmax), cbar=cbar)
        axs[i].set(xlabel="Receptor 1 Abundance (#/cell)", ylabel='Receptor 2 Abundance (#/cell)')
        axs[i].set_title("Ligand 1 in Mixture = {}%".format(comp * 100))

    return fig