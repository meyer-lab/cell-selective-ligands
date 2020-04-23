import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

from ..model import *
from .figureCommon import getSetup, subplotLabel


cellPopulations = {
    "1": [2, 2, 0.5, 0.25, 45],
    "2": [3, 2, 0.5, 0.25, 0],
    "3": [4, 2, 0.5, 0.25, 0],
    "4": [2, 4, 0.3, 0.6, 0],
    "5": [3.2, 3.7, 0.5, 0.25, 45],
    "6": [3.7, 3.2, 0.5, 0.25, 45],
    "7": [4, 4, 0.5, 0.25, 45],
    "8": [3.2, 3.2, 0.25, 1, 45],
}

abundRange = (1.5, 4.5)


def overlapCellPopulation(ax, scale, data=cellPopulations):
    ax_new = ax.twinx().twiny()
    ax_new.set_xscale("linear")
    ax_new.set_yscale("linear")
    ax_new.set_xticks([])
    ax_new.set_yticks([])
    ax_new.set_xlim(scale)
    ax_new.set_ylim(scale)
    for label, item in data.items():
        ax_new.add_patch(Ellipse(xy=(item[0], item[1]),
                                 width=item[2],
                                 height=item[3],
                                 angle=item[4],
                                 facecolor="blue",
                                 fill=True,
                                 alpha=0.5,
                                 linewidth=1))
        ax_new.text(item[0], item[1], label,
                    horizontalalignment='center',
                    verticalalignment='center',
                    fontweight='bold',
                    color='white')


def abundHeatMap(ax, abundRange, L0, KxStar, Kav, Comp, f=None, Cplx=None, vmin=-2, vmax=4):
    assert bool(f is None) != bool(Cplx is None)
    nAbdPts = 70
    abundScan = np.logspace(abundRange[0], abundRange[1], nAbdPts)

    if f is None:
        func = np.vectorize(lambda abund1, abund2: polyc(L0, KxStar, [abund1, abund2], Cplx, Comp, Kav)[0])
    else:
        func = np.vectorize(lambda abund1, abund2: polyfc(L0, KxStar, f, [abund1, abund2], Comp, Kav)[0])

    X, Y = np.meshgrid(abundScan, abundScan)
    logZ = np.log(func(X, Y))

    contours = ax.contour(X, Y, logZ, levels=np.arange(-10, 20, 0.1), colors='black', linewidths=0.2)
    ax.set_xscale("log")
    ax.set_yscale("log")
    plt.clabel(contours, inline=True, fontsize=3)
    ax.pcolor(X, Y, logZ, cmap='RdGy_r', vmin=vmin, vmax=vmax)
    overlapCellPopulation(ax, abundRange)


def affinity(L0, KxStar, Comp, ff=None, Cplx=None, offdiag=1e5, vmin=-2, vmax=4):
    nAffPts = 3
    axs, fig = getSetup((8, 8), (nAffPts, nAffPts))

    fig.suptitle("Lbound when $L_0$={}, $f$={}, $K_x^*$={:.2e}, $LigC$={}".format(L0, ff, KxStar, Comp))

    affRange = (5., 7.)
    affScan = np.logspace(affRange[0], affRange[1], nAffPts)

    for i1, aff1 in enumerate(affScan):
        for i2, aff2 in enumerate(np.flip(affScan)):
            abundHeatMap(axs[i2 * nAffPts + i1], abundRange,
                         L0, KxStar, [[aff1, aff2]], Comp, f=ff, Cplx=Cplx,
                         vmin=vmin, vmax=vmax)
            axs[i2 * nAffPts + i1].set_title("$K_1$ = {:.1e} $K_2$ = {:.1e}".format(aff1, aff2))
    return fig


def valency(L0, KxStar, Comp, Kav=[[1e6, 1e5], [1e5, 1e6]], Cplx=None, vmin=-2, vmax=4):
    ffs = [1, 2, 4, 8, 16]
    axs, fig = getSetup((3 * len(ffs) - 1, 3), (1, len(ffs)))

    fig.suptitle("Lbound when $L_0$={}, $Kav$={}, $K_x^*$={:.2e}, $LigC$={}".format(L0, Kav, KxStar, Comp))

    for i, v in enumerate(ffs):
        abundHeatMap(axs[i], abundRange, L0, KxStar, Kav, Comp, f=v, Cplx=Cplx, vmin=vmin, vmax=vmax)
        axs[i].set_title("$f$ = {}".format(v))

    return fig


def mixture(L0, KxStar, Kav=[[1e6, 1e5], [1e5, 1e6]], ff=5, vmin=-2, vmax=4):
    comps = [0.0, 0.2, 0.5, 0.8, 1.0]
    axs, fig = getSetup((3 * len(comps) - 1, 3), (1, len(comps)))

    fig.suptitle("Lbound when $L_0$={}, $Kav$={}, $f$={}, $K_x^*$={:.2e}".format(L0, Kav, ff, KxStar))

    for i, comp in enumerate(comps):
        abundHeatMap(axs[i], abundRange, L0, KxStar, Kav, [comp, 1 - comp], f=ff, Cplx=None, vmin=vmin, vmax=vmax)
        axs[i].set_title("$LigC$ = [{}, {}]".format(comp, 1 - comp))

    return fig


def complex(L0, KxStar, Kav=[[1e6, 1e5], [1e5, 1e6]], vmin=-2, vmax=4):
    cplx = [0, 2, 4]
    axs, fig = getSetup((8, 8), (len(cplx), len(cplx)))

    fig.suptitle("Lbound when $L_0$={}, $Kav$={}, $K_x^*$={:.2e}".format(L0, Kav, KxStar))

    for i1, cplx1 in enumerate(cplx):
        for i2, cplx2 in enumerate(np.flip(cplx)):
            abundHeatMap(axs[i2 * len(cplx) + i1], abundRange, L0, KxStar, Kav, [1],
                         Cplx=[[cplx1, cplx2]], vmin=vmin, vmax=vmax)
            axs[i2 * len(cplx) + i1].set_title("$Cplx$ = [{}, {}]".format(cplx1, cplx2))
    return fig
