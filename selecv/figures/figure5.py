"""
Figure 5. Heterovalent bispecific
"""

import seaborn as sns
import pandas as pds
import numpy as np
import matplotlib.pyplot as plt
from .figureCommon import getSetup, subplotLabel, heatmap, cellPopulations
from ..model import polyc, polyfc


pairs = [(r"$R_1^{hi}R_2^{lo}$", r"$R_1^{med}R_2^{lo}$"), (r"$R_1^{hi}R_2^{hi}$", r"$R_1^{med}R_2^{med}$"),
             (r"$R_1^{med}R_2^{hi}$", r"$R_1^{hi}R_2^{med}$"), (r"$R_1^{hi}R_2^{lo}$", r"$R_1^{lo}R_2^{hi}$")]

def makeFigure():
    ax, f = getSetup((11, 11), (3, 3))
    subplotLabel(ax, list(range(0, 9)))

    L0 = 1e-8
    KxStar = 1e-12
    Kav = [[1e7, 1e5], [1e5, 1e6]]

    for i, KxStar in enumerate([1e-12, 1e-10, 1e-14]):
        heatmap(ax[i], L0, KxStar, Kav, [1.0], Cplx=[[1, 1]], vrange=(-4, 7), fully=True, title="[1, 1] log fully bound with Kx*={}".format(KxStar))
        ax[0].set(xlabel="Receptor 1 Abundance (#/cell)", ylabel='Receptor 2 Abundance (#/cell)')
        heatmap(ax[i+3], L0, KxStar, Kav, [1.0], Cplx=[[1, 1]], vrange=(-4, 7), fully=False, title="[1, 1] log Lbound with Kx*={}".format(KxStar))
        ax[0].set(xlabel="Receptor 1 Abundance (#/cell)", ylabel='Receptor 2 Abundance (#/cell)')

    KxStarFully(ax[6], L0, Kav, fully=False)
    KxStarFully(ax[7], L0, Kav, fully=True)

    """for i, s in enumerate([[1, 1], [2, 0], [0, 2]]):
        heatmap(ax[i], L0 * 0.5, KxStar, Kav, [1.0], Cplx=[s], vrange=(-7, 3), title="{} log fully bound".format(s))
        ax[i].set(xlabel="Receptor 1 Abundance (#/cell)", ylabel='Receptor 2 Abundance (#/cell)')

    for i, pair in enumerate(pairs):
        affHeatMap(ax[i + 3], pair[0], pair[1], L0, KxStar, Cbar=True)

    composition(ax[7], pairs, L0, KxStar, Kav, [[1, 1], [2, 0]])
    composition(ax[8], pairs, L0, KxStar, Kav, [[2, 0], [1, 1]])
    composition(ax[9], pairs, L0, KxStar, Kav, [[1, 1], [2, 0]], all=True)"""

    return f


def selectivity(pop1name, pop2name, L0, KxStar, Cplx, Ctheta, Kav, all=False, fully=True):
    """ Always calculate the full binding of the 1st kind of complex """
    pop1 = cellPopulations[pop1name][0], cellPopulations[pop1name][1]
    pop2 = cellPopulations[pop2name][0], cellPopulations[pop2name][1]
    if all:
        return np.sum(polyc(L0, KxStar, np.power(10, pop1), Cplx, Ctheta, Kav)[2]) \
            / np.sum(polyc(L0, KxStar, np.power(10, pop2), Cplx, Ctheta, Kav)[2])
    else:
        if fully:
            return polyc(L0, KxStar, np.power(10, pop1), Cplx, Ctheta, Kav)[2][0] \
                / polyc(L0, KxStar, np.power(10, pop2), Cplx, Ctheta, Kav)[2][0]
        else:
            return polyc(L0, KxStar, np.power(10, pop1), Cplx, Ctheta, Kav)[0][0] \
                   / polyc(L0, KxStar, np.power(10, pop2), Cplx, Ctheta, Kav)[0][0]


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


def KxStarFully(ax, L0, Kav, ylim=(-7, 5), fully=False):
    nPoints = 50
    Kxaxis = np.logspace(-15, -7, nPoints)

    colors = ["royalblue", "orange", "limegreen", "orangered"]
    sHolder = np.zeros((nPoints))
    for i, pair in enumerate(pairs):
        for j, KxStar in enumerate(Kxaxis):
            sHolder[j] = np.log(selectivity(pair[0], pair[1], L0, KxStar, [[1, 1]], [1], Kav, fully=fully))
        ax.plot(Kxaxis, sHolder, color=colors[i], label=pair[0] + " to " + pair[1], linestyle="-")

    ax.set(xlim=(1e-15, 1e-7), ylim=ylim,
           xlabel="Kx*",
           ylabel="Log selectivity of [1, 1]")
    ax.set_xscale('log')
    if fully:
        ax.set_title("Log selectivity varies with Kx* for Lfbnd")
    else:
        ax.set_title("Log selectivity varies with Kx* for Lbound")
    ax.legend(loc='lower right', fancybox=True, framealpha=1)


def composition(ax, pairs, L0, KxStar, Kav, Cplx, ylim=(-5, 17), all=False):
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
    ax.legend(loc='upper left', fancybox=True, framealpha=1)
