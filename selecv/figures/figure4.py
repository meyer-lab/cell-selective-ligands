"""
Figure 4. Mixtures for enhanced targeting.
"""

import numpy as np
from .figureCommon import subplotLabel, getSetup, popCompare, heatmap

ligConc = np.array([1e-8])
KxStarP = 1e-10
val = 1.0


def makeFigure():
    """ Make figure 4. """
    # Get list of axis objects
    ax, f = getSetup((8, 6), (2, 3))
    mixture(f, ax, 1e-9, 10 ** -10, ff=1, vmin=-2, vmax=3.5)
    subplotLabel(ax, [0, 3, 4])
    fsize = 10

    popCompare(ax[3], [r"$R_1^{hi}R_2^{lo}$", r"$R_1^{med}R_2^{lo}$"], "Mix", Kav=np.array([[1e6, 1e5], [1e5, 1e6]]))
    popCompare(ax[4], [r"$R_1^{hi}R_2^{hi}$", r"$R_1^{hi}R_2^{lo}$", r"$R_1^{lo}R_2^{hi}$"], "Mix", Kav=np.array([[1e6, 1e5], [1e5, 1e6]]))
    ax[5].axis("off")

    for subax in ax:
        subax.set_xticklabels(subax.get_xticklabels(), fontsize=fsize)
        subax.set_yticklabels(subax.get_yticklabels(), fontsize=fsize)
        subax.set_xlabel(subax.get_xlabel(), fontsize=fsize)
        subax.set_ylabel(subax.get_ylabel(), fontsize=fsize)
        subax.set_title(subax.get_title(), fontsize=fsize)

    return f


def mixture(fig, axs, L0, KxStar, Kav=[[1e6, 1e5], [1e5, 1e6]], ff=5, vmin=-2, vmax=4):
    comps = [0.0, 0.5, 1.0]

    for i, comp in enumerate(comps):
        cbar = False
        if i in [2, 4]:
            cbar = True
        heatmap(axs[i], L0, KxStar, Kav, [comp, 1 - comp], f=ff, Cplx=None, vrange=(vmin, vmax), cbar=cbar, layover=1)
        axs[i].set(xlabel="Receptor 1 Abundance (#/cell)", ylabel='Receptor 2 Abundance (#/cell)')
        axs[i].set_title("Ligand 1 = {}%, Ligand 2 = {}%".format(comp * 100, 100 - comp * 100))

    return fig
