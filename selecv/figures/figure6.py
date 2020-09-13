"""
Figure 6.
"""
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.optimize import minimize
from matplotlib import pyplot as plt
import matplotlib.cm as cm
from .figureCommon import subplotLabel, getSetup, heatmapNorm, cellPopulations, overlapCellPopulation
from ..imports import getPopDict
from ..sampling import sigmaPop
from ..model import polyfc, polyc


def makeFigure():
    """ Make figure 6. """
    # Get list of axis objects
    ax, f = getSetup((24, 12), (4, 6))
    subplotLabel(ax, list(range(0, 22)))

    optimizeDesign(ax[0:6], [r"$R_1^{lo}R_2^{hi}$"], vrange=(0, 3))
    optimizeDesign(ax[6:12], [r"$R_1^{hi}R_2^{hi}$"], vrange=(0, 1.5))
    optimizeDesign(ax[12:18], [r"$R_1^{med}R_2^{med}$"], vrange=(0, 10))

    affDLsub = np.array([0, 10])
    fDLsub = 4
    optParams, DLaffs = optimizeDesignDL(ax[18], [r"$R_1^{med}R_2^{med}$"], fDLsub, affDLsub)
    heatmapDL(ax[19], np.exp(optParams[0]), np.exp(optParams[1]), np.array([[DLaffs[0], DLaffs[1]], [np.exp(optParams[4]), np.exp(optParams[5])]]),
              [0.5, 0.5], Cplx=np.array([[fDLsub, 0], [0, optParams[2]]]), vrange=(0, 10), cbar=False, highlight=r"$R_1^{med}R_2^{med}$")
    heatmapDL(ax[20], np.exp(optParams[0]), np.exp(optParams[1]), np.array([[DLaffs[0], DLaffs[1]], [np.exp(optParams[4]), np.exp(optParams[5])]]),
              [0.5, 0.5], Cplx=np.array([[fDLsub, 0], [0, optParams[2]]]), vrange=(0, 10), cbar=False, dead=True, highlight=r"$R_1^{med}R_2^{med}$")
    heatmapDL(ax[21], np.exp(optParams[0]) / 2, np.exp(optParams[1]), np.array([[DLaffs[0], DLaffs[1]], [np.exp(optParams[4]), np.exp(optParams[5])]]),
              [0, 1], Cplx=np.array([[fDLsub, 0], [0, optParams[2]]]), vrange=(0, 10), cbar=True, dead=False, jTherap=True, highlight=r"$R_1^{med}R_2^{med}$")
    ax[22].axis("off")
    ax[23].axis("off")

    return f


_, df = getPopDict()


def genPopMeans(popName):
    assert isinstance(popName, list)
    res = []
    for name in popName:
        dfPop = df[df["Population"] == name]
        res.append(np.array([dfPop["Receptor_1"].to_numpy()[0], dfPop["Receptor_2"].to_numpy()[0]]))
    return res


def minSelecFunc(x, tPops, offTPops):
    """Provides the function to be minimized to get optimal selectivity"""
    offTargetBound = 0
    tMeans, offTMeans = genPopMeans(tPops), genPopMeans(offTPops)

    targetBound = polyfc(np.exp(x[0]), np.exp(x[1]), x[2], [10**tMeans[0][0], 10**tMeans[0][1]], [x[3], 1 - x[3]],
                         np.array([[np.exp(x[4]), np.exp(x[5])], [np.exp(x[5]), np.exp(x[4])]]))[0]
    for means in offTMeans:
        offTargetBound += polyfc(np.exp(x[0]), np.exp(x[1]), x[2], [10**means[0], 10**means[1]], [x[3], 1 - x[3]],
                                 np.array([[np.exp(x[4]), np.exp(x[5])], [np.exp(x[5]), np.exp(x[4])]]))[0]

    return (offTargetBound) / (targetBound)


def genOnevsAll(targetPop, specPops=False, means=False):
    assert isinstance(targetPop, list)
    targPops, offTargPops = [], []
    if means:
        if specPops:
            for _, pop in enumerate(df["Population"].unique()):
                dfPop = df[df["Population"] == pop]
                if pop == targetPop[0]:
                    targPops.append(np.array([dfPop["Receptor_1"].to_numpy(), dfPop["Receptor_2"].to_numpy()]).flatten())
                elif pop in specPops:
                    offTargPops.append(np.array([dfPop["Receptor_1"].to_numpy(), dfPop["Receptor_2"].to_numpy()]).flatten())
        else:
            for _, pop in enumerate(df["Population"].unique()):
                dfPop = df[df["Population"] == pop]
                if pop == targetPop[0]:
                    targPops.append(np.array([dfPop["Receptor_1"].to_numpy(), dfPop["Receptor_2"].to_numpy()]).flatten())
                else:
                    offTargPops.append(np.array([dfPop["Receptor_1"].to_numpy(), dfPop["Receptor_2"].to_numpy()]).flatten())
    else:
        if specPops:
            for _, pop in enumerate(df["Population"].unique()):
                if pop == targetPop[0]:
                    targPops.append(pop)
                elif pop in specPops:
                    offTargPops.append(pop)
        else:
            for _, pop in enumerate(df["Population"].unique()):
                if pop == targetPop[0]:
                    targPops.append(pop)
                else:
                    offTargPops.append(pop)

    return targPops, offTargPops


def minSigmaVar(x, tPops, offTPops, h=None):
    targetBound, offTargetBound = 0, 0
    for tPop in tPops:
        targetBound += sum(sigmaPop(tPop, np.exp(x[0]), np.exp(x[1]), x[2], [x[3], 1 - x[3]],
                                    np.array([[np.exp(x[4]), np.exp(x[5])], [np.exp(x[5]), np.exp(x[4])]]), quantity=0, h=h))
    for offTPop in offTPops:
        offTargetBound += sum(sigmaPop(offTPop, np.exp(x[0]), np.exp(x[1]), x[2], [x[3], 1 - x[3]],
                                       np.array([[np.exp(x[4]), np.exp(x[5])], [np.exp(x[5]), np.exp(x[4])]]), quantity=0, h=h))
    return (offTargetBound) / (targetBound)


def optimize(pmOptNo, targPops, offTargPops, L0, KxStar, f, LigC, Kav, bound=None):
    """ A more general purpose optimizer """
    # OPT = [log L0, log KxStar, f, LigC[0], log Ka(diag), log Ka(offdiag)]
    Kav = np.array(Kav)
    xnot = np.array([np.log(L0), np.log(KxStar), f, LigC[0], np.log(Kav[0, 0]), np.log(Kav[0, 1])])
    Bnds = [(i, i) for i in xnot]
    for pmopt in pmOptNo:
        Bnds[pmopt] = optBnds[pmopt]
    if bound is not None:
        Bnds = bound
    print(targPops, offTargPops)
    optimized = minimize(minSigmaVar, xnot, bounds=np.array(Bnds), jac="3-point", method="L-BFGS-B", args=(targPops, offTargPops),
                         options={"eps": 0.1, "disp": False})
    return optimized


def optimizeDesign(ax, targetPop, vrange=(0, 5)):
    "Runs optimization and determines optimal parameters for selectivity of one population vs. another"
    targPops, offTargPops = genOnevsAll(targetPop)
    targMeans, offTargMeans = genPopMeans(targPops), genPopMeans(offTargPops)

    optDF = pd.DataFrame(columns=["Strategy", "Selectivity"])
    strats = ["Xnot", "Affinity", "Mixture", "Valency", "All"]
    pmOpts = [[], [1, 4, 5], [1, 3], [1, 3], [1, 3, 4, 5]]

    for i, strat in enumerate(strats):
        optimized = optimize(pmOpts[i], targPops, offTargPops, 1e-9, 1e-12, 1, [1, 0], np.ones((2, 2)) * 1e6, bound=bndsDict[strat])
        stratRow = pd.DataFrame({"Strategy": strat, "Selectivity": np.array([len(offTargMeans) / optimized.fun])})
        optDF = optDF.append(stratRow, ignore_index=True)
        optParams = optimized.x
        optParams[0:2] = np.exp(optParams[0:2])
        optParams[4::] = np.exp(optParams[4::])

        if i < 4:
            heatmapNorm(ax[i + 1], targMeans[0], optParams[0], optParams[1], [[optParams[4], optParams[5]], [optParams[4], optParams[5]]],
                        [optParams[3], 1 - optParams[3]], f=optParams[2], vrange=vrange, cbar=False, layover=True, highlight=targetPop[0])
        else:
            heatmapNorm(ax[i + 1], targMeans[0], optParams[0], optParams[1], [[optParams[4], optParams[5]], [optParams[4], optParams[5]]],
                        [optParams[3], 1 - optParams[3]], f=optParams[2], vrange=vrange, cbar=True, layover=True, highlight=targetPop[0])
        ax[i + 1].set(title=strat, xlabel="Receptor 1 Abundance ($cell^{-1}$))", ylabel="Receptor 2 Abundance ($cell^{-1}$))")

    sns.barplot(x="Strategy", y="Selectivity", data=optDF, ax=ax[0])
    ax[0].set(title="Optimization of " + targetPop[0])


optBnds = [(np.log(1e-11), np.log(1e-8)),  # log L0
           (np.log(1e-15), np.log(1e-9)),  # log KxStar
           (1, 16),  # f
           (0.0, 1.0),  # LigC[0]
           (np.log(1e2), np.log(1e9)),  # log Ka(diag)
           (np.log(1e2), np.log(1e9))]  # log Ka(offdiag)


bndsDict = {
    "Xnot": ((np.log(1e-9), np.log(1e-9)), (np.log(1e-12), np.log(1e-12)), (1, 1), (1, 1), (np.log(1e6), np.log(1e6)), (np.log(1e6), np.log(1e6))),
    "Affinity": ((np.log(1e-9), np.log(1e-9)), (np.log(1e-15), np.log(1e-9)), (1, 1), (1, 1), (np.log(1e2), np.log(1e10)), (np.log(1e2), np.log(1e10))),
    "Mixture": ((np.log(1e-9), np.log(1e-9)), (np.log(1e-15), np.log(1e-9)), (1, 1), (0, 1), (np.log(1e6), np.log(1e10)), (np.log(1e2), np.log(1e6))),
    "Valency": ((np.log(1e-9), np.log(1e-9)), (np.log(1e-15), np.log(1e-9)), (1, 16), (1, 1), (np.log(1e2), np.log(1e10)), (np.log(1e2), np.log(1e10))),
    "All": ((np.log(1e-9), np.log(1e-9)), (np.log(1e-15), np.log(1e-9)), (1, 16), (0, 1), (np.log(1e2), np.log(1e10)), (np.log(1e2), np.log(1e10)))
}


def optimizeDesignAnim(targetPop):
    "Runs optimization and determines optimal parameters for selectivity of one population vs. another"
    targPops, offTargPops = genOnevsAll(targetPop)
    targMeans, offTargMeans = genPopMeans(targPops), genPopMeans(offTargPops)

    optDF = pd.DataFrame(columns=["Strategy", "Selectivity"])
    strats = ["Xnot", "Affinity", "Mixture", "Valency", "All"]
    pmOpts = [[], [1, 4, 5], [1, 3], [1, 3], [1, 3, 4, 5]]
    optParamsHold = np.zeros([len(strats), 6])

    for i, strat in enumerate(strats):
        optimized = optimize(pmOpts[i], targMeans, offTargMeans, 1e-9, 1e-12, 1, [1, 0], np.ones((2, 2)) * 1e6, bound=bndsDict[strat])
        stratRow = pd.DataFrame({"Strategy": strat, "Selectivity": np.array([len(offTargMeans) / optimized.fun])})
        optDF = optDF.append(stratRow, ignore_index=True)
        optParams = optimized.x
        optParams[0:2] = np.exp(optParams[0:2])
        optParams[4::] = np.exp(optParams[4::])
        optParamsHold[i, :] = optParams

    return optParamsHold, strats


def minSelecFuncDL(x, tMeans, offTMeans, fDL, affDL):
    "Provides the function to be minimized to get optimal selectivity with addition of dead ligand"
    offTargetBound = 0
    targetBound = polyc(np.exp(x[0]), np.exp(x[1]), [10**tMeans[0][0], 10**tMeans[0][1]], [[fDL, 0], [0, x[2]]], [0.5, 0.5], np.array([[affDL[0], affDL[1]], [np.exp(x[4]), np.exp(x[5])]]))[0][1]
    for means in offTMeans:
        offTargetBound += polyc(np.exp(x[0]), np.exp(x[1]), [10**means[0], 10**means[1]], [[fDL, 0], [0, x[2]]], [0.5, 0.5], np.array([[affDL[0], affDL[1]], [np.exp(x[4]), np.exp(x[5])]]))[0][1]
    return (offTargetBound) / (targetBound)


def optimizeDesignDL(ax, targetPop, fDL, affDL, specPops=False):
    "Runs optimization and determines optimal parameters for selectivity of one population vs. another with inclusion of dead ligand"
    targMeans, offTargMeans = genOnevsAll(targetPop, specPops, means=True)
    print(targMeans[0])
    npoints = 5
    ticks = np.full([npoints], None)
    affScan = np.logspace(affDL[0], affDL[1], npoints)
    ticks[0], ticks[-1] = "1e" + str(affDL[0]), "1e" + str(affDL[1])
    bounds = ((np.log(1e-9), np.log(1e-9)), (np.log(1e-15), np.log(1e-9)), (1, 1), (0, 1), (np.log(1e2), np.log(1e10)), (np.log(1e2), np.log(1e10)))
    xnot = np.array([np.log(1e-9), np.log(1e-9), 1, 1, np.log(10e8), np.log(10e8)])
    ratioDF = pd.DataFrame(columns=affScan, index=affScan)

    for ii, aff1 in enumerate(affScan):
        sampMeans = np.zeros(npoints)
        for jj, aff2 in enumerate(np.flip(affScan)):
            optimized = minimize(minSelecFuncDL, xnot, bounds=np.array(bounds), method="L-BFGS-B", jac="3-point", args=(targMeans, offTargMeans, fDL, [aff1, aff2]), options={"eps": 0.1, "disp": False})
            sampMeans[jj] = 7 / optimized.fun
        ratioDF[ratioDF.columns[ii]] = sampMeans

    ratioDF = ratioDF.divide(ratioDF.iloc[npoints - 1, 0])
    Cbar = True
    sns.heatmap(ratioDF, ax=ax, xticklabels=ticks, yticklabels=np.flip(ticks), vmin=0, vmax=4, cbar=Cbar, cbar_kws={'label': 'Selectivity Ratio w Dead Ligand'}, annot=True)
    ax.set(xlabel="Dead Ligand Rec 1 Affinity ($K_a$, in M$^{-1}$)", ylabel="Dead Ligand Rec 2 Affinity ($K_a$, in M$^{-1}$)")

    if specPops:
        ax.set_title(targetPop[0] + " and Dead ligand of Valency " + str(fDL) + " vs " + specPops[0], fontsize=8)
    else:
        ax.set_title(targetPop[0] + " and Dead ligand of Valency " + str(fDL) + " vs all", fontsize=8)

    maxindices = ratioDF.to_numpy()
    (i, j) = np.unravel_index(maxindices.argmax(), maxindices.shape)
    maxaff1 = affScan[j]
    maxaff2 = affScan[npoints - 1 - i]
    optimized = minimize(minSelecFuncDL, xnot, bounds=np.array(bounds), method="L-BFGS-B", jac="3-point", args=(targMeans, offTargMeans, fDL, [maxaff1, maxaff2]), options={"eps": 0.1, "disp": False})

    return optimized.x, np.array([maxaff1, maxaff2])


def heatmapDL(ax, L0, KxStar, Kav, Comp, Cplx=None, vrange=(-2, 4), title="", cbar=True, dead=False, jTherap=False, highlight=[]):
    "Makes a heatmap with modified cell population abundances according to dead ligand binding"
    nAbdPts = 70
    abunds = np.array(list(cellPopulations.values()))[:, 0:2]
    abundRange = (np.amin(abunds) - 1, np.amax(abunds) + 1)
    abundScan = np.logspace(abundRange[0], abundRange[1], nAbdPts)
    if dead:
        func = np.vectorize(lambda abund1, abund2: polyc(L0, KxStar, [abund1, abund2], Cplx, Comp, Kav)[0][0])
        title = "Dead Ligand Bound"
    else:
        func = np.vectorize(lambda abund1, abund2: polyc(L0, KxStar, [abund1, abund2], Cplx, Comp, Kav)[0][1])
        if jTherap:
            title = "Live Ligand Bound without Dead Ligand"
        else:
            title = "Live Ligand Bound"

    X, Y = np.meshgrid(abundScan, abundScan)
    logZ = np.log(func(X, Y))
    contours = ax.contour(X, Y, logZ, levels=np.arange(-20, 20, 0.5), colors="black", linewidths=0.5)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title(title)
    plt.clabel(contours, inline=True, fontsize=6)
    ax.pcolor(X, Y, logZ, cmap='summer', vmin=vrange[0], vmax=vrange[1])
    norm = plt.Normalize(vmin=vrange[0], vmax=vrange[1])
    ax.set(xlabel="Receptor 1 Abundance (#/cell)", ylabel='Receptor 2 Abundance (#/cell)')
    if cbar:
        cbar = ax.figure.colorbar(cm.ScalarMappable(norm=norm, cmap='summer'), ax=ax)
        cbar.set_label("Log Ligand Bound")
    overlapCellPopulation(ax, abundRange, data=cellPopulations, highlight=highlight)
