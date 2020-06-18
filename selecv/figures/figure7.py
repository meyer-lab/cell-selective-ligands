import numpy as np
from .figureCommon import subplotLabel, getSetup
from ..model import *
from matplotlib import gridspec, pyplot as plt
import seaborn as sns
import matplotlib.cm as cm


def makeFigure():
    """ Make figure 7. """
    ax, f = getSetup((7, 6), (3, 3))
    subplotLabel(ax)

    #bispecific_ratio(ax[0])
    #bispecific_heatmap(ax[1])
    bi_affinity(f, ax[0:9])

    return f


def bisp_ratio(L0, KxStar, Rtot, Kav, Lbound = True):
    """ Lbound of bispecific to monovalent ligand ratio """
    Lbound1, Rbound1 = polyc(L0, KxStar, Rtot, np.array([[1, 0]]), [1], Kav)
    Lbound2, Rbound2 = polyc(L0, KxStar, Rtot, np.array([[1, 1]]), [1], Kav)
    if Lbound:
        return Lbound2[0] / Lbound1[0]
    else:
        return np.sum(Rbound2) / np.sum(Rbound1)


def bispecific_ratio(ax):
    L0 = 1e-9
    Rtot = [1e4, 1e2]
    Kav = np.array([[1e6, 5e4], [5e4, 1e6]])

    x = np.logspace(-15, -8, num=16)
    y = [bisp_ratio(L0, KxStar, Rtot, Kav) for KxStar in x]

    sns.lineplot(ax=ax, x=x, y=y)
    ax.set(xscale="log", ylim=(0, 2), title="Bispecific to monospecific Lbound ratio", xlabel="$K_x^*$", ylabel="ratio")


def bispecific_heatmap(ax, Kav, KxStar = 1e-12, vmin=0, vmax=3, Lbound = True):
    L0 = 1e-9
    nAbdPts = 70
    abundScan = np.logspace(1.5, 4.5, nAbdPts)
    func = np.vectorize(lambda r1, r2: bisp_ratio(L0, KxStar, [r1, r2], Kav, Lbound = Lbound))
    X, Y = np.meshgrid(abundScan, abundScan)
    logZ = np.log(func(X, Y))

    #contours = ax.contour(X, Y, logZ, levels=np.arange(-10, 20, 0.1), colors="black", linewidths=0.2)
    ax.pcolor(X, Y, logZ, cmap="RdGy_r", vmin=vmin, vmax=vmax)
    ax.set_xscale("log")
    ax.set_yscale("log")
    #plt.clabel(contours, inline=True, fontsize=3)

    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    cbar = ax.figure.colorbar(cm.ScalarMappable(norm=norm, cmap="RdGy_r"), ax=ax)
    cbar.set_label("Log Ligand Bound")


def bi_affinity(fig, axs, vmin=0, vmax=3):
    fig.suptitle("Bispecific to monovalent Lbound ratio")

    nAffPts = 3
    affRange = (5., 7.)
    affScan = np.logspace(affRange[0], affRange[1], nAffPts)

    for i1, aff1 in enumerate(affScan):
        for i2, aff2 in enumerate(np.flip(affScan)):
            Kav = np.array([[aff1, 1e5], [1e5, aff2]])
            bispecific_heatmap(axs[i2 * nAffPts + i1], Kav, KxStar=1e-12, vmin=vmin, vmax=vmax)
            axs[i2 * nAffPts + i1].set_title("$K_1$ = {:.1e} $K_2$ = {:.1e}".format(aff1, aff2))
    return fig










