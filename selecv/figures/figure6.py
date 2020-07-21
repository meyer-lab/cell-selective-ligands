"""
Figure 6.
"""
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.optimize import minimize
from .figureCommon import subplotLabel, getSetup, heatmap, heatmapNorm
from ..imports import getPopDict
from ..sampling import sampleSpec
from ..model import polyfc


def makeFigure():
    """ Make figure 6. """
    # Get list of axis objects
    ax, f = getSetup((16, 8), (3, 6))
    subplotLabel(ax)

    # gridSearchTry(populationsdf, ['Pop5', 'Pop3'])
    optimizeDesign(ax[0:6], [r"$R_1^{lo}R_2^{hi}$"], vrange=(0, 3))
    optimizeDesign(ax[6:12], [r"$R_1^{hi}R_2^{hi}$"], vrange=(0, 1.5))
    optimizeDesign(ax[12:18], [r"$R_1^{med}R_2^{med}$"], vrange=(0, 10))

    return f


_, df = getPopDict()


def minSelecFunc(x, tMeans, offTMeans):
    "Provides the function to be minimized to get optimal selectivity"
    offTargetBound = 0

    targetBound = polyfc(np.exp(x[0]), np.exp(x[1]), x[2], [10**tMeans[0][0], 10**tMeans[0][1]], [x[3], 1 - x[3]], np.array([[np.exp(x[4]), np.exp(x[5])], [np.exp(x[5]), np.exp(x[4])]]))[0]

    for means in offTMeans:
        offTargetBound += polyfc(np.exp(x[0]), np.exp(x[1]), x[2], [10**means[0], 10**means[1]], [x[3], 1 - x[3]], np.array([[np.exp(x[4]), np.exp(x[5])], [np.exp(x[5]), np.exp(x[4])]]))[0]

    return (offTargetBound) / (targetBound)


def genOnevsAll(targetPop):
    assert isinstance(targetPop, list)
    targMeans, offTargMeans = [], []
    for _, pop in enumerate(df["Population"].unique()):
        dfPop = df[df["Population"] == pop]
        if pop == targetPop[0]:
            targMeans.append(np.array([dfPop["Receptor_1"].to_numpy(), dfPop["Receptor_2"].to_numpy()]).flatten())
        else:
            offTargMeans.append(np.array([dfPop["Receptor_1"].to_numpy(), dfPop["Receptor_2"].to_numpy()]).flatten())
    return targMeans, offTargMeans


def optimize(pmOptNo, targMeans, offTargMeans, L0, KxStar, f, LigC, Kav, bound=None):
    """ A more general purpose optimizer """
    # OPT = [log L0, log KxStar, f, LigC[0], log Ka(diag), log Ka(offdiag)]
    Kav = np.array(Kav)
    xnot = np.array([np.log(L0), np.log(KxStar), f, LigC[0], np.log(Kav[0, 0]), np.log(Kav[0, 1])])
    Bnds = [(i, i) for i in xnot]
    for pmopt in pmOptNo:
        Bnds[pmopt] = optBnds[pmopt]
    if bound is not None:
        Bnds = bound
    optimized = minimize(minSelecFunc, xnot, bounds=np.array(Bnds), method="L-BFGS-B", args=(targMeans, offTargMeans),
                         options={"eps": 1, "disp": True})
    return optimized


def optimizeDesign(ax, targetPop, vrange=(0, 5)):
    "Runs optimization and determines optimal parameters for selectivity of one population vs. another"
    targMeans, offTargMeans = genOnevsAll(targetPop)

    optDF = pd.DataFrame(columns=["Strategy", "Selectivity"])
    strats = ["Xnot", "Affinity", "Mixture", "Valency", "All"]
    pmOpts = [[], [1, 4, 5], [1, 3], [1, 3], [1, 3, 4, 5]]

    for i, strat in enumerate(strats):
        optimized = optimize(pmOpts[i], targMeans, offTargMeans, 1e-9, 1e-12, 1, [1, 0], np.ones((2, 2)) * 1e6, bound=bndsDict[strat])
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


searchdic = {
    "L0": np.logspace(-11, -8, 4),
    "Kx": np.logspace(-12, -8, 5),
    "Val": np.logspace(0.0, 4.0, base=2.0, num=5),
    "Mix": np.linspace(0, 0.5, 2),
    "Aff": np.logspace(5, 9, 2),
}


def gridSearchTry(df, popList):
    """Grid search for best params for selectivity. Works but slowly. Probably won't use."""
    recMeans, Covs = [], []
    for ii, pop in enumerate(popList):
        dfPop = df[df["Population"] == pop]
        recMeans.append(np.array([dfPop["Receptor_1"].to_numpy(), dfPop["Receptor_2"].to_numpy()]).flatten())
        Covs.append(dfPop.Covariance_Matrix.to_numpy()[0])

    resultsTensor = np.zeros([4, 5, 5, 3, 5, 5, 5, 5])
    for ii, conc in enumerate(searchdic["L0"]):
        for jj, kx in enumerate(searchdic["Kx"]):
            for kk, val in enumerate(searchdic["Val"]):
                for ll, mix in enumerate(searchdic["Mix"]):
                    for mm, aff1 in enumerate(searchdic["Aff"]):
                        for nn, aff2 in enumerate(searchdic["Aff"]):
                            for oo, aff3 in enumerate(searchdic["Aff"]):
                                for pp, aff4 in enumerate(searchdic["Aff"]):
                                    resultsTensor[ii, jj, kk, ll, mm, nn, oo, pp] = sampleSpec(
                                        conc, kx, val, recMeans, Covs, np.array([mix, 1 - mix]), np.array([[aff1, aff2], [aff3, aff4]])
                                    )[1]

    maxSelec = np.amax(resultsTensor)
    maxSelecCoords = np.unravel_index(np.argmax(resultsTensor), resultsTensor.shape)
    maxParams = np.array(
        [
            searchdic["L0"][maxSelecCoords[0]],
            searchdic["Kx"][maxSelecCoords[1]],
            searchdic["Val"][maxSelecCoords[2]],
            searchdic["Mix"][maxSelecCoords[3]],
            searchdic["Aff"][maxSelecCoords[4]],
            searchdic["Aff"][maxSelecCoords[5]],
            searchdic["Aff"][maxSelecCoords[6]],
            searchdic["Aff"][maxSelecCoords[7]],
        ]
    )

    return maxSelec, maxParams


def optimizeDesignAnim(targetPop):
    "Runs optimization and determines optimal parameters for selectivity of one population vs. another"
    targMeans, offTargMeans = genOnevsAll(targetPop)

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
