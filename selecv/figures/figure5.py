"""
Figure 5. Heterovalent bispecific
"""

import seaborn as sns
import pandas as pds
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from .figureCommon import getSetup, subplotLabel, heatmap, cellPopulations, overlapCellPopulation
from ..model import polyc, polyfc


pairs = [(r"$R_1^{hi}R_2^{lo}$", r"$R_1^{med}R_2^{lo}$"), (r"$R_1^{hi}R_2^{hi}$", r"$R_1^{med}R_2^{med}$"),
         (r"$R_1^{med}R_2^{hi}$", r"$R_1^{hi}R_2^{med}$"), (r"$R_1^{hi}R_2^{lo}$", r"$R_1^{lo}R_2^{hi}$")]


def makeFigure():
    ax, f = getSetup((10, 12), (4, 3))
    subplotLabel(ax, list(range(0, 9)))

    L0 = 1e-8
    Kav = [[1e7, 1e5], [1e5, 1e6]]

    for i, KxStar in enumerate([1e-10, 1e-12, 1e-14]):
        cbar = True if i==2 else False
        heatmap(ax[i], L0, KxStar, Kav, [1.0], Cplx=[[1, 1]], vrange=(-4, 7), fully=True,
                title="[1, 1] log fully bound with Kx*={}".format(KxStar), cbar=cbar)
        ax[i].set(xlabel="Receptor 1 Abundance (#/cell)", ylabel='Receptor 2 Abundance (#/cell)')
        heatmap(ax[i + 3], L0, KxStar, Kav, [0.5, 0.5], Cplx=[[2, 0], [0, 2]], vrange=(-4, 7), fully=True,
                title="Mixture of bivalent log Lbound with Kx*={}".format(KxStar), cbar=cbar)
        ax[i+3].set(xlabel="Receptor 1 Abundance (#/cell)", ylabel='Receptor 2 Abundance (#/cell)')
        normHeatmap(ax[i + 6], L0, KxStar, Kav, vrange=(-14, 0),
                    title="Tethered normalized by untethered with Kx*={}".format(KxStar), cbar=cbar)
        ax[i+6].set(xlabel="Receptor 1 Abundance (#/cell)", ylabel='Receptor 2 Abundance (#/cell)')


    KxStarFully(ax[9], L0, Kav, fully=False)
    KxStarFully(ax[10], L0, Kav, fully=True)
    KxStarFully(ax[11], L0, Kav, ylim=(-8, 9),  tetherComp=True)

    """for i, s in enumerate([[1, 1], [2, 0], [0, 2]]):
        heatmap(ax[i], L0 * 0.5, KxStar, Kav, [1.0], Cplx=[s], vrange=(-7, 3), title="{} log fully bound".format(s))
        ax[i].set(xlabel="Receptor 1 Abundance (#/cell)", ylabel='Receptor 2 Abundance (#/cell)')

    for i, pair in enumerate(pairs):
        affHeatMap(ax[i + 3], pair[0], pair[1], L0, KxStar, Cbar=True)

    composition(ax[7], pairs, L0, KxStar, Kav, [[1, 1], [2, 0]])
    composition(ax[8], pairs, L0, KxStar, Kav, [[2, 0], [1, 1]])
    composition(ax[9], pairs, L0, KxStar, Kav, [[1, 1], [2, 0]], all=True)"""

    return f

def tetheredYN(L0, KxStar, Rtot, Kav, fully=True):
    if fully:
        return polyc(L0, KxStar, Rtot, [[1, 1]], [1.0], Kav)[2][0] / \
               np.sum(polyc(L0, KxStar, Rtot, [[2, 0], [0, 2]], [0.5, 0.5], Kav)[2])
               #polyfc(L0*2, KxStar, 1, Rtot, [0.5, 0.5], Kav)[0]
    else:
        return polyc(L0, KxStar, Rtot, [[1, 1]], [1.0], Kav)[0][0] / \
               np.sum(polyc(L0, KxStar, Rtot, [[2, 0], [0, 2]], [0.5, 0.5], Kav)[2])
               #polyfc(L0*2, KxStar, 1, Rtot, [0.5, 0.5], Kav)[0]


def normHeatmap(ax, L0, KxStar, Kav, vrange=(-4, 2), title="", cbar=False, fully=True, layover=True):
    nAbdPts = 70
    abundRange = (1.5, 4.5)
    abundScan = np.logspace(abundRange[0], abundRange[1], nAbdPts)

    func = np.vectorize(lambda abund1, abund2: tetheredYN(L0, KxStar, [abund1, abund2], Kav, fully=fully))
    X, Y = np.meshgrid(abundScan, abundScan)
    logZ = np.log(func(X, Y))

    contours = ax.contour(X, Y, logZ, levels=np.arange(-20, 20, 0.5), colors="black", linewidths=0.5)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title(title)
    plt.clabel(contours, inline=True, fontsize=6)
    ax.pcolor(X, Y, logZ, cmap='RdYlGn', vmin=vrange[0], vmax=vrange[1])
    norm = plt.Normalize(vmin=vrange[0], vmax=vrange[1])
    if cbar:
        cbar = ax.figure.colorbar(cm.ScalarMappable(norm=norm, cmap='RdYlGn'), ax=ax)
        cbar.set_label("Log tethered advantage")
    if layover:
        overlapCellPopulation(ax, abundRange)


def selectivity(pop1name, pop2name, L0, KxStar, Cplx, Ctheta, Kav, fully=True, untethered=False):
    """ Always calculate the full binding of the 1st kind of complex """
    pop1 = cellPopulations[pop1name][0], cellPopulations[pop1name][1]
    pop2 = cellPopulations[pop2name][0], cellPopulations[pop2name][1]
    if untethered:
        return polyfc(L0, KxStar, 1, np.power(10, pop1), [0.5, 0.5], Kav)[0] \
               / polyfc(L0, KxStar, 1, np.power(10, pop2), [0.5, 0.5], Kav)[0]
    if fully:
        return np.sum(polyc(L0, KxStar, np.power(10, pop1), Cplx, Ctheta, Kav)[2]) \
            / np.sum(polyc(L0, KxStar, np.power(10, pop2), Cplx, Ctheta, Kav)[2])
    else:
        return np.sum(polyc(L0, KxStar, np.power(10, pop1), Cplx, Ctheta, Kav)[0]) \
               / np.sum(polyc(L0, KxStar, np.power(10, pop2), Cplx, Ctheta, Kav)[0])


def affHeatMap(ax, pop1name, pop2name, L0, KxStar, Cbar=True):
    npoints = 50
    affRange = (4, 8)
    ticks = np.full([npoints], None)
    affScan = np.logspace(affRange[0], affRange[1], npoints)
    ticks[0], ticks[-1] = "1e" + str(affRange[0]), "1e" + str(affRange[1])

    sampMeans = np.zeros(npoints)
    ratioDF = pds.DataFrame(columns=affScan, index=affScan)

    for ii, aff1 in enumerate(affScan):
        for jj, aff2 in enumerate(np.flip(affScan)):
            sampMeans[jj] = selectivity(pop1name, pop2name, L0, KxStar, [[1, 1]], [1], [[aff1, 1e4], [1e4, aff2]])
            sampMeans[jj] = np.log(sampMeans[jj])
        ratioDF[ratioDF.columns[ii]] = sampMeans

    maxind = np.argmax(ratioDF.to_numpy())
    maxx, maxy = maxind // npoints, maxind % npoints
    maxval = ratioDF.to_numpy()[maxx, maxy]
    ax.scatter(np.array([maxy]), np.array([maxx]), s=320, marker='*', color='green', zorder=3)

    sns.heatmap(ratioDF, ax=ax, xticklabels=ticks, yticklabels=np.flip(ticks), cbar=Cbar, annot=False)
    ax.set(xlabel="Rec 1 Affinity ($K_a$, in M$^{-1}$)", ylabel="Rec 2 Affinity ($K_a$, in M$^{-1}$)")
    ax.set_title(pop1name + " to " + pop2name + " bispecific log binding ratio", fontsize=8)


def KxStarFully(ax, L0, Kav, ylim=(-7, 5), fully=False, tetherComp=False):
    nPoints = 50
    Kxaxis = np.logspace(-15, -7, nPoints)

    colors = ["royalblue", "orange", "limegreen", "orangered"]
    sHolder = np.zeros((nPoints))
    for i, pair in enumerate(pairs):
        for j, KxStar in enumerate(Kxaxis):
            if tetherComp:
                sHolder[j] = selectivity(pair[0], pair[1], L0, KxStar, [[1, 1]], [1], Kav, fully=True, untethered=False) \
                             / selectivity(pair[0], pair[1], L0, KxStar, [[2, 0], [0, 2]], [0.5, 0.5], Kav, fully=True)
            else:
                sHolder[j] = np.log(selectivity(pair[0], pair[1], L0, KxStar, [[1, 1]], [1], Kav, fully=fully, untethered=False))
        ax.plot(Kxaxis, sHolder, color=colors[i], label=pair[0] + " to " + pair[1], linestyle="-")

    ax.set(xlim=(1e-15, 1e-7), ylim=ylim,
           xlabel="Kx*")
    ax.set_xscale('log')
    if tetherComp:
        ax.set_ylabel("Ratio of selectivity")
        ax.set_title("Ratio of tethered selectivity to untethered selectivity")
    else:
        ax.set_ylabel("Log selectivity of [1, 1]")
        if fully:
            ax.set_title("Log selectivity varies with Kx* for Lfbnd")
        else:
            ax.set_title("Log selectivity varies with Kx* for Lbound")
    ax.legend(loc='lower right', fancybox=True, framealpha=1)


"""def composition(ax, pairs, L0, KxStar, Kav, Cplx, ylim=(-5, 17), all=False):
    nPoints = 20
    xaxis = np.linspace(0.01, 1, nPoints)

    colors = ["royalblue", "orange", "limegreen", "orangered"]
    sHolder = np.zeros((nPoints))
    for i, pair in enumerate(pairs):
        for j, Ctheta0 in enumerate(xaxis):
            sHolder[j] = np.log(selectivity(pair[0], pair[1], L0, KxStar, Cplx, [Ctheta0, 1 - Ctheta0], Kav, all=all))
        ax.plot(xaxis, sHolder, color=colors[i], label=pair[0] + " to " + pair[1], linestyle="-")

    if all:
        ax.set(xlim=(0, 1), ylim=ylim, xlabel="Composition of " + str(Cplx[0]),
               ylabel="Log selectivity of all complexes")
        ax.set_title("Composition of " + str(Cplx[0]) + " vs selectivity of all")
    else:
        ax.set(xlim=(0, 1), ylim=ylim, xlabel="Composition of " + str(Cplx[0]),
               ylabel="Log selectivity of " + str(Cplx[0]))
        ax.set_title("Composition vs selectivity of " + str(Cplx[0]))
    ax.legend(loc='upper left', fancybox=True, framealpha=1)"""
