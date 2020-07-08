import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import matplotlib.cm as cm
from .figureCommon import getSetup
from ..model import polyc


def makeFigure(KxStar=1e-12):
    ax, f = getSetup((14, 9), (3, 4))

    L0 = 1e-9
    Comp = [0.5, 0.5]
    Kav = [[1e7, 1e5], [1e5, 1e6]]

    competeHeatmap(ax[0], L0, KxStar, [[1, 1], [1, 0]], Comp, Kav, title="[1, 0] ratio", ratio=True)
    competeHeatmap(ax[1], L0, KxStar, [[1, 1], [0, 1]], Comp, Kav, title="[0, 1] ratio", ratio=True)
    competeHeatmap(ax[2], L0, KxStar, [[1, 1], [2, 0]], Comp, Kav, title="[2, 0] ratio", ratio=True)
    competeHeatmap(ax[3], L0, KxStar, [[1, 1], [0, 2]], Comp, Kav, title="[0, 2] ratio", ratio=True)
    competeHeatmap(ax[4], L0, KxStar, [[1, 1], [2, 1]], Comp, Kav, title="[2, 1] ratio", ratio=True)
    competeHeatmap(ax[5], L0, KxStar, [[1, 1], [1, 2]], Comp, Kav, title="[1, 2] ratio", ratio=True)
    competeHeatmap(ax[6], L0, KxStar, [[1, 1], [1, 0]], Comp, Kav, title="[1, 0] Lbound", ratio=False)
    competeHeatmap(ax[7], L0, KxStar, [[1, 1], [0, 1]], Comp, Kav, title="[0, 1] Lbound", ratio=False)
    competeHeatmap(ax[8], L0, KxStar, [[1, 1], [2, 0]], Comp, Kav, title="[2, 0] Lbound", ratio=False)
    competeHeatmap(ax[9], L0, KxStar, [[1, 1], [0, 2]], Comp, Kav, title="[0, 2] Lbound", ratio=False)
    competeHeatmap(ax[10], L0, KxStar, [[1, 1], [2, 1]], Comp, Kav, title="[2, 1] Lbound", ratio=False)
    competeHeatmap(ax[11], L0, KxStar, [[1, 1], [1, 2]], Comp, Kav, title="[1, 2] Lbound", ratio=False)

    f.suptitle("$K_x^* = ${}".format(KxStar))
    return f


def Cplx1to2Ratio(L0, KxStar, Rtot, Cplx, Ctheta, Kav, ratio=True):
    """ Always LigCplx 1 / LigCplx 0 """
    Lbound, Rbound = polyc(L0, KxStar, Rtot, Cplx, Ctheta, Kav)
    if ratio:
        return Lbound[1] / Lbound[0]
    else:
        return Lbound[1]


def competeHeatmap(ax, L0, KxStar, Cplx, Comp, Kav, vrange=(-3, 3), cbar=True, layover=False, title="", ratio=True):
    abundRange = 2, 6
    nAbdPts = 70
    abundScan = np.logspace(abundRange[0], abundRange[1], nAbdPts)

    func = np.vectorize(lambda abund1, abund2: Cplx1to2Ratio(L0, KxStar, [abund1, abund2], Cplx, Comp, Kav, ratio=ratio))
    X, Y = np.meshgrid(abundScan, abundScan)
    logZ = np.log(func(X, Y))

    contours = ax.contour(X, Y, logZ, levels=np.arange(-10, 20, 0.4), colors="black", linewidths=0.5)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title(title)
    plt.clabel(contours, inline=True, fontsize=5)
    ax.pcolor(X, Y, logZ, cmap='summer', vmin=vrange[0], vmax=vrange[1])
    norm = plt.Normalize(vmin=vrange[0], vmax=vrange[1])
    if cbar:
        cbar = ax.figure.colorbar(cm.ScalarMappable(norm=norm, cmap='summer'), ax=ax)
        if ratio:
            cbar.set_label("Log Cplx2 to Cplx1 ratio")
        else:
            cbar.set_label("Log Cplx2 Lbound")
    if layover:
        overlapCellPopulation(ax, abundRange)
