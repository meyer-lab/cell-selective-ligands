"""
Figure 5. Optimization
"""
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.optimize import minimize
from .figureCommon import subplotLabel, setFontSize, getSetup, heatmapNorm
from ..imports import getPopDict
from ..sampling import sigmaPop
from valentbind import polyfc


def makeFigure():
    """ Make figure 5. """
    # Get list of axis objects
    ax, f = getSetup((16, 8), (3, 6))
    subplotLabel(ax)

    optimizeDesign(ax[0:6], [r"$R_1^{lo}R_2^{hi}$"], vrange=(0, 3))
    optimizeDesign(ax[6:12], [r"$R_1^{hi}R_2^{hi}$"], vrange=(0, 3))
    optimizeDesign(ax[12:18], [r"$R_1^{med}R_2^{med}$"], vrange=(0, 100))

    setFontSize(ax, 9,
                xsci=[1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17],
                ysci=[1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17])
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
                         np.array([[np.exp(x[4]), np.exp(x[5])], [np.exp(x[6]), np.exp(x[7])]]))[0]
    for means in offTMeans:
        offTargetBound += polyfc(np.exp(x[0]), np.exp(x[1]), x[2], [10**means[0], 10**means[1]], [x[3], 1 - x[3]],
                                 np.array([[np.exp(x[4]), np.exp(x[5])], [np.exp(x[6]), np.exp(x[7])]]))[0]

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


def minSigmaVar(x, tPops, offTPops, h=None, recFactor=1.0):
    targetBound, offTargetBound = 0, 0
    for tPop in tPops:
        targetBound += sum(sigmaPop(tPop, np.exp(x[0]), np.exp(x[1]), x[2], [x[3], 1 - x[3]],
                                    np.array([[np.exp(x[4]), np.exp(x[5])], [np.exp(x[6]), np.exp(x[7])]]), quantity=0, h=h, recFactor=recFactor))
    for offTPop in offTPops:
        offTargetBound += sum(sigmaPop(offTPop, np.exp(x[0]), np.exp(x[1]), x[2], [x[3], 1 - x[3]],
                                       np.array([[np.exp(x[4]), np.exp(x[5])], [np.exp(x[6]), np.exp(x[7])]]), quantity=0, h=h, recFactor=recFactor))
    return (offTargetBound) / (targetBound)


def optimize(pmOptNo, targPops, offTargPops, L0, KxStar, f, LigC, Kav, bound=None, recFactor=1.0):
    """ A more general purpose optimizer """
    # OPT = [log L0, log KxStar, f, LigC[0], log Ka(diag), log Ka(offdiag)]
    Kav = np.array(Kav)
    xnot = np.array([np.log(L0), np.log(KxStar), f, LigC[0], np.log(Kav[0, 0]), np.log(Kav[0, 1]), np.log(Kav[1, 0]), np.log(Kav[1, 1])])
    Bnds = [(i, i + 0.001) for i in xnot]

    for pmopt in pmOptNo:
        Bnds[pmopt] = optBnds[pmopt]
    if bound is not None:
        Bnds = bound

    optimized = minimize(minSigmaVar, xnot, bounds=np.array(Bnds), args=(targPops, offTargPops, None, recFactor))
    if not optimized.success:
        print(Bnds)
        print(optimized)

    return optimized


def optimizeDesign(ax, targetPop, vrange=(0, 5), recFactor=1.0):
    "Runs optimization and determines optimal parameters for selectivity of one population vs. another"
    targPops, offTargPops = genOnevsAll(targetPop)
    targMeans, offTargMeans = genPopMeans(targPops), genPopMeans(offTargPops)

    optDF = pd.DataFrame(columns=["Strategy", "Selectivity"])
    strats = ["Unoptimized", "Affinity", "Mixture+Affinity", "Valency+Affinity", "All"]
    pmOpts = [[], [1, 4, 5], [1, 3], [1, 3], [1, 3, 4, 5]]
    paramsArr = np.zeros((len(strats), 8))

    for i, strat in enumerate(strats):
        if strat == "Mixture+Affinity":
            optimized = optimize(pmOpts[i], targPops, offTargPops, 1e-9, 1e-12, 1, [0.9, 0.1], np.ones((2, 2)) * 1e6, bound=bndsDict[strat], recFactor=recFactor)
        elif strat == "All":
            optimized = optimize(pmOpts[i], targPops, offTargPops, 1e-9, optParams[1], optParams[2], [0.9, 0.1], np.ones((2, 2)) * 1e6, bound=bndsDict[strat], recFactor=recFactor)
        elif strat == "Valency+Affinity":
            optimized = optimize(pmOpts[i], targPops, offTargPops, 1e-9, 1e-12, 15, [1, 0], np.ones((2, 2)) * 1e6, bound=bndsDict[strat], recFactor=recFactor)
        else:
            optimized = optimize(pmOpts[i], targPops, offTargPops, 1e-9, 1e-12, 1, [1, 0], np.ones((2, 2)) * 1e6, bound=bndsDict[strat], recFactor=recFactor)

        optParams = optimized.x
        optParams[0:2] = np.exp(optParams[0:2])
        optParams[4::] = np.exp(optParams[4::])
        stratRow = pd.DataFrame({"Strategy": strat, "Selectivity": np.array([len(offTargMeans) / optimized.fun])})

        if strat == "All" and stratRow.Selectivity.values < optDF.Selectivity.max():
            bestStrat = optDF.loc[optDF.Selectivity == optDF.Selectivity.max()].Strategy.values[0]
            bestStratInd = optDF.loc[optDF.Strategy == bestStrat].index
            optParams = paramsArr[bestStratInd, :][0]
            stratRow = pd.DataFrame({"Strategy": [strat], "Selectivity": optDF.Selectivity.max()})

        optDF = optDF.append(stratRow, ignore_index=True)
        paramsArr[i, :] = optParams

        if i <= 0:
            heatmapNorm(ax[i + 1], targMeans[0], optParams[0], optParams[1],
                        [[optParams[4], optParams[5]], [optParams[6], optParams[7]]],
                        [optParams[3], 1 - optParams[3]], f=optParams[2], vrange=vrange, cbar=False, layover=2,
                        highlight=targetPop[0], lineN=41, recFactor=recFactor)
        elif i < 4:
            heatmapNorm(ax[i + 1], targMeans[0], optParams[0], optParams[1],
                        [[optParams[4], optParams[5]], [optParams[6], optParams[7]]],
                        [optParams[3], 1 - optParams[3]], f=optParams[2], vrange=vrange, cbar=False, layover=1,
                        highlight=targetPop[0], lineN=41, recFactor=recFactor)
        else:
            heatmapNorm(ax[i + 1], targMeans[0], optParams[0], optParams[1],
                        [[optParams[4], optParams[5]], [optParams[6], optParams[7]]],
                        [optParams[3], 1 - optParams[3]], f=optParams[2], vrange=vrange, cbar=True, layover=1,
                        highlight=targetPop[0], lineN=41, recFactor=recFactor)
        ax[i + 1].set(title=strat, xlabel="Receptor 1 Abundance [$cell^{-1}$]", ylabel="Receptor 2 Abundance [$cell^{-1}$]")

    sns.barplot(x="Strategy", y="Selectivity", data=optDF, ax=ax[0], color='k')
    ax[0].set(title="Optimization of " + targetPop[0])
    ax[0].set_xticklabels(ax[0].get_xticklabels(), rotation=35, horizontalalignment='right')
    if recFactor == 1.0:
        if targetPop == [r"$R_1^{med}R_2^{med}$"]:
            ax[0].set_ylim(0, 1)
        else:
            ax[0].set_ylim(0, 16)
    else:
        ax[0].set_ylim(0, 50)


optBnds = [(np.log(1e-11), np.log(1e-8)),  # log L0
           (np.log(1e-15), np.log(1e-9)),  # log KxStar
           (1, 16),  # f
           (0.0, 1.0),  # LigC[0]
           (np.log(1e2), np.log(1e9)),  # log Ka(diag)
           (np.log(1e2), np.log(1e9)),
           (np.log(1e2), np.log(1e9)),
           (np.log(1e2), np.log(1e9))]  # log Ka(offdiag)


cBnd = (np.log(1e-9), np.log(1.01e-9))

bndsDict = {
    "Unoptimized": (cBnd, (np.log(1e-12), np.log(1.01e-12)), (1, 1.01), (0.99, 1.00), (np.log(1e6), np.log(1.01e6)), (np.log(1e6), np.log(1.01e6)), (np.log(1e6), np.log(1.01e6)), (np.log(1e6), np.log(1.01e6))),
    "Affinity": (cBnd, (np.log(1e-12), np.log(1.01e-12)), (1, 1.01), (0.99, 1.00), (np.log(1e2), np.log(1e10)), (np.log(1e2), np.log(1e10)), (np.log(1e6), np.log(1.01e6)), (np.log(1e6), np.log(1.01e6))),
    "Mixture+Affinity": (cBnd, (np.log(1e-12), np.log(1.01e-12)), (1, 1.01), (0, 1), (np.log(1e2), np.log(1e10)), (np.log(1e2), np.log(1e10)), (np.log(1e2), np.log(1e10)), (np.log(1e2), np.log(1e10))),
    "Valency+Affinity": (cBnd, (np.log(1e-15), np.log(1e-9)), (1, 16), (0.99, 1.00), (np.log(1e2), np.log(1e10)), (np.log(1e2), np.log(1e10)), (np.log(1e6), np.log(1.01e6)), (np.log(1e6), np.log(1.01e6))),
    "All": (cBnd, (np.log(1e-15), np.log(1e-9)), (1, 16), (0, 1), (np.log(1e2), np.log(1e10)), (np.log(1e2), np.log(1e10)), (np.log(1e2), np.log(1e10)), (np.log(1e2), np.log(1e10)))
}


def optimizeDesignAnim(targetPop):
    """ Runs optimization and determines optimal parameters for selectivity of one population vs. another. """
    targPops, offTargPops = genOnevsAll(targetPop)
    targMeans, offTargMeans = genPopMeans(targPops), genPopMeans(offTargPops)

    optDF = pd.DataFrame(columns=["Strategy", "Selectivity"])
    strats = ["Unoptimized", "Affinity", "Mixture+Affinity", "Valency+Affinity", "All"]
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
