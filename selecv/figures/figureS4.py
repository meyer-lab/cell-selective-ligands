"""
Figure S4.
"""
import numpy as np
import pandas as pd
import seaborn as sns
import copy
from matplotlib import pyplot as plt
import matplotlib.cm as cm
from scipy.optimize import minimize
from .figureCommon import subplotLabel, getSetup, cellPopulations, overlapCellPopulation
from .figure6 import genOnevsAll
from ..model import polyc


def makeFigure():
    """ Make figure S4. """
    # Get list of axis objects
    ax, f = getSetup((18, 24), (4, 3))
    subplotLabel(ax)
    affDLsub = np.array([0, 10])

    fDLsub = 1
    optParams, DLaffs = optimizeDesignDL(ax[0], [r"$R_1^{lo}R_2^{hi}$"], fDLsub, affDLsub)
    modPop = modifyCellPops(cellPopulations, optParams, DLaffs, fDLsub)
    heatmapDL(ax[6], np.exp(optParams[0]), np.exp(optParams[1]), np.array([[DLaffs[0], DLaffs[1]], [np.exp(optParams[4]), np.exp(optParams[5])]]),
              [0.5, 0.5], modPop, Cplx=np.array([[fDLsub, 0], [0, optParams[2]]]), vrange=(-2, 4), title="", cbar=False)
    optParams, DLaffs = optimizeDesignDL(ax[1], [r"$R_1^{hi}R_2^{hi}$"], fDLsub, affDLsub)
    modPop = modifyCellPops(cellPopulations, optParams, DLaffs, fDLsub)
    heatmapDL(ax[7], np.exp(optParams[0]), np.exp(optParams[1]), np.array([[DLaffs[0], DLaffs[1]], [np.exp(optParams[4]), np.exp(optParams[5])]]),
              [0.5, 0.5], modPop, Cplx=np.array([[fDLsub, 0], [0, optParams[2]]]), vrange=(-2, 4), title="", cbar=False)

    optParams, DLaffs = optimizeDesignDL(ax[2], [r"$R_1^{med}R_2^{med}$"], fDLsub, affDLsub)
    modPop = modifyCellPops(cellPopulations, optParams, DLaffs, fDLsub)
    heatmapDL(ax[8], np.exp(optParams[0]), np.exp(optParams[1]), np.array([[DLaffs[0], DLaffs[1]], [np.exp(optParams[4]), np.exp(optParams[5])]]),
              [0.5, 0.5], modPop, Cplx=np.array([[fDLsub, 0], [0, optParams[2]]]), vrange=(-2, 4), title="", cbar=False)

    fDLsub = 4
    optParams, DLaffs = optimizeDesignDL(ax[3], [r"$R_1^{lo}R_2^{hi}$"], fDLsub, affDLsub)
    modPop = modifyCellPops(cellPopulations, optParams, DLaffs, fDLsub)
    heatmapDL(ax[9], np.exp(optParams[0]), np.exp(optParams[1]), np.array([[DLaffs[0], DLaffs[1]], [np.exp(optParams[4]), np.exp(optParams[5])]]),
              [0.5, 0.5], modPop, Cplx=np.array([[fDLsub, 0], [0, optParams[2]]]), vrange=(-2, 4), title="", cbar=False)

    optParams, DLaffs = optimizeDesignDL(ax[4], [r"$R_1^{hi}R_2^{hi}$"], fDLsub, affDLsub)
    modPop = modifyCellPops(cellPopulations, optParams, DLaffs, fDLsub)
    heatmapDL(ax[10], np.exp(optParams[0]), np.exp(optParams[1]), np.array([[DLaffs[0], DLaffs[1]], [np.exp(optParams[4]), np.exp(optParams[5])]]),
              [0.5, 0.5], modPop, Cplx=np.array([[fDLsub, 0], [0, optParams[2]]]), vrange=(-2, 4), title="", cbar=False)

    optParams, DLaffs = optimizeDesignDL(ax[5], [r"$R_1^{med}R_2^{med}$"], fDLsub, affDLsub)
    modPop = modifyCellPops(cellPopulations, optParams, DLaffs, fDLsub)
    heatmapDL(ax[11], np.exp(optParams[0]), np.exp(optParams[1]), np.array([[DLaffs[0], DLaffs[1]], [np.exp(optParams[4]), np.exp(optParams[5])]]),
              [0.5, 0.5], modPop, Cplx=np.array([[fDLsub, 0], [0, optParams[2]]]), vrange=(-2, 4), title="", cbar=False)

    return f


def minSelecFuncDL(x, tMeans, offTMeans, fDL, affDL):
    "Provides the function to be minimized to get optimal selectivity with addition of dead ligand"
    offTargetBound = 0
    targetBound = polyc(np.exp(x[0]), np.exp(x[1]), [10**tMeans[0][0], 10**tMeans[0][1]], [[fDL, 0], [0, x[2]]], [0.5, 0.5], np.array([[affDL[0], affDL[1]], [np.exp(x[4]), np.exp(x[5])]]))[0][1]

    for means in offTMeans:
        offTargetBound += polyc(np.exp(x[0]), np.exp(x[1]), [10**means[0], 10**means[1]], [[fDL, 0], [0, x[2]]], [0.5, 0.5], np.array([[affDL[0], affDL[1]], [np.exp(x[4]), np.exp(x[5])]]))[0][1]

    return (offTargetBound) / (targetBound)


def optimizeDesignDL(ax, targetPop, fDL, affDL):
    "Runs optimization and determines optimal parameters for selectivity of one population vs. another with inclusion of dead ligand"
    targMeans, offTargMeans = genOnevsAll(targetPop)

    npoints = 5
    ticks = np.full([npoints], None)
    affScan = np.logspace(affDL[0], affDL[1], npoints)
    ticks[0], ticks[-1] = "1e" + str(affDL[0]), "1e" + str(affDL[1])
    bounds = ((np.log(1e-9), np.log(1e-9)), (np.log(1e-15), np.log(1e-9)), (1, 16), (0, 1), (np.log(1e2), np.log(1e10)), (np.log(1e2), np.log(1e10)))
    xnot = np.array([np.log(1e-9), np.log(1e-9), 1, 1, np.log(10e8), np.log(10e6)])

    ratioDF = pd.DataFrame(columns=affScan, index=affScan)

    for ii, aff1 in enumerate(affScan):
        sampMeans = np.zeros(npoints)
        for jj, aff2 in enumerate(np.flip(affScan)):
            optimized = minimize(minSelecFuncDL, xnot, bounds=np.array(bounds), method="L-BFGS-B", args=(targMeans, offTargMeans, fDL, [aff1, aff2]), options={"eps": 1, "disp": True})
            sampMeans[jj] = 7 / optimized.fun
        ratioDF[ratioDF.columns[ii]] = sampMeans

    ratioDF = ratioDF.divide(ratioDF.iloc[npoints - 1, 0])
    Cbar = True

    sns.heatmap(ratioDF, ax=ax, xticklabels=ticks, yticklabels=np.flip(ticks), vmin=0, vmax=2, cbar=Cbar, cbar_kws={'label': 'Binding Ratio'}, annot=True)

    ax.set(xlabel="Dead Ligand Rec 1 Affinity ($K_a$, in M$^{-1}$)", ylabel="Dead Ligand Rec 2 Affinity ($K_a$, in M$^{-1}$)")
    ax.set_title(targetPop, fontsize=8)

    maxindices = ratioDF.to_numpy()
    (i, j) = np.unravel_index(maxindices.argmax(), maxindices.shape)
    maxaff1 = affScan[j]
    maxaff2 = affScan[npoints - 1 - i]
    optimized = minimize(minSelecFuncDL, xnot, bounds=np.array(bounds), method="L-BFGS-B", args=(targMeans, offTargMeans, fDL, [maxaff1, maxaff2]), options={"eps": 1, "disp": True})

    return optimized.x, np.array([maxaff1, maxaff2])


def modifyCellPops(cellPopsOriginal, optLig, dLigAff, fDL):
    "Modify cell pops by amount of dead ligand binding"
    x = optLig
    cellPopsNew = copy.copy(cellPopsOriginal)
    for i, row in cellPopsNew.items():
        Rbound = polyc(np.exp(x[0]), np.exp(x[1]), [10**row[0], 10**row[1]], [[fDL, 0], [0, x[2]]], [0.5, 0.5], np.array([[dLigAff[0], dLigAff[1]], [np.exp(x[4]), np.exp(x[5])]]))[1][0, :]
        row[0:2] = np.log10(10 ** np.array(row[0:2]) - Rbound)
        cellPopsNew[i] = row
    return cellPopsNew


def heatmapDL(ax, L0, KxStar, Kav, Comp, modPops, Cplx=None, vrange=(-2, 4), title="", cbar=True):
    "Makes a heatmap with modified cell population abundances according to dead ligand binding"
    nAbdPts = 70
    abunds = np.array(list(modPops.values()))[:, 0:2]
    abundRange = (np.amin(abunds) - 1, np.amax(abunds) + 1)
    abundScan = np.logspace(abundRange[0], abundRange[1], nAbdPts)

    func = np.vectorize(lambda abund1, abund2: polyc(L0, KxStar, [abund1, abund2], Cplx, Comp, Kav)[0][1])

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
        cbar.set_label("Log Ligand Bound")
    overlapCellPopulation(ax, abundRange, data=modPops)
