"""
Figure for creating Animations.
"""

import copy
import numpy as np
import pandas as pd
import seaborn as sns
from celluloid import Camera
from .figureCommon import getSetup, heatmapNorm
from .figure5 import optimizeDesignAnim, genOnevsAll, minSelecFunc


def makeFigure():
    """ Make figure 6. """
    # Get list of axis objects
    overlay = False
    if overlay:
        ax, f = getSetup((8, 3), (1, 2))
    else:
        ax, f = getSetup((8, 3), (1, 2))
    camera = Camera(f)
    npointsAn = 35
    vrange = (0, 3)
    targetPop = [r"$R_1^{hi}R_2^{hi}$"]
    targMeans, offTargMeans = genOnevsAll(targetPop)
    selecDF = pd.DataFrame(columns=["Xo", "Affinity", "Mixture", "Valency", "All"])
    overlay = False

    optParamsFrame, strats = optimizeDesignAnim(targetPop)
    optParamsFrame, stratList = spaceParams(optParamsFrame, npointsAn, strats)
    XnotRow = copy.copy(optParamsFrame[0, :])
    XnotRow[0:2] = np.log(XnotRow[0:2])
    XnotRow[4::] = np.log(XnotRow[4::])
    XnotSelec = 7 / minSelecFunc(XnotRow, targMeans, offTargMeans)

    selecDF = pd.DataFrame({"Strategy": ["Xo", "Affinity", "Mixture", "Valency", "All"], "Selectivity": np.tile(XnotSelec, [5])})

    for i in range(0, optParamsFrame.shape[0]):
        optParamsR = optParamsFrame[i, :]
        strat = stratList[i]
        if overlay:
            selecRow = copy.copy(optParamsR)
            selecRow[0:2] = np.log(selecRow[0:2])
            selecRow[4::] = np.log(selecRow[4::])
            selecDF.loc[(selecDF["Strategy"] == strat), "Selectivity"] = (7 / minSelecFunc(selecRow, targMeans, offTargMeans))
            sns.barplot(x="Strategy", y="Selectivity", data=selecDF, ax=ax[1])
            heatmapNorm(ax[0], targMeans[0], optParamsR[0], optParamsR[1], [[optParamsR[4], optParamsR[5]], [optParamsR[4], optParamsR[5]]],
                        [optParamsR[3], 1 - optParamsR[3]], f=optParamsR[2], vrange=vrange, cbar=True, layover=overlay, highlight=targetPop[0])
        else:
            heatmapNorm(ax[0], targMeans[0], optParamsR[0], optParamsR[1], [[optParamsR[4], optParamsR[5]], [optParamsR[4], optParamsR[5]]],
                        [optParamsR[3], 1 - optParamsR[3]], f=optParamsR[2], vrange=vrange, cbar=True, layover=overlay, highlight=False)
        ax[0].set(xlabel="Receptor 1 Abundance ($cell^{-1}$))", ylabel="Receptor 2 Abundance ($cell^{-1}$))")
        camera.snap()

    anim = camera.animate()
    anim.save('/home/brianoj/cell-selective-ligands/output/animation.gif', writer='imagemagick', fps=5)

    return f


def spaceParams(optParams, nPoints, strats):
    "Gives a matrix of spaced parameters for gif creation"
    animParams = np.zeros([nPoints * 4, optParams.shape[1]])
    stratList = []
    for i in range(0, (optParams.shape[0] - 1)):
        for j in range(0, (optParams.shape[1])):
            animParams[nPoints * i: nPoints * i + nPoints - 5, j] = np.linspace(optParams[0, j], optParams[i + 1, j], nPoints - 5)
        animParams[nPoints * i + nPoints - 5: nPoints * i + nPoints, :] = np.tile(optParams[i + 1, :], [5, 1])
        stratList.append(np.repeat(strats[i + 1], nPoints)[0::])

    return animParams, flatten(stratList)


def flatten(l):
    "Flattens a list to 1D"
    return [item for sublist in l for item in sublist]
