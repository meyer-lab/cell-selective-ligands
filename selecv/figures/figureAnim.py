"""
Figure for creating Animations.
"""

import copy
import numpy as np
import pandas as pd
import seaborn as sns
from celluloid import Camera
from .figureCommon import getSetup, heatmapNorm
from .figure6 import optimizeDesignAnim, genOnevsAll, minSelecFunc


def makeFigure():
    """ Make figure 6. """
    # Get list of axis objects
    ax, f = getSetup((8, 3), (1, 2))
    camera = Camera(f)
    npointsAn = 25
    vrange = (0, 3)
    targetPop = [r"$R_1^{lo}R_2^{hi}$"]
    targMeans, offTargMeans = genOnevsAll(targetPop)
    selecDF = pd.DataFrame(columns=["Xnot", "Affinity", "Mixture", "Valency", "All"])

    optParamsFrame, strats = optimizeDesignAnim(targetPop)
    optParamsFrame, stratList = spaceParams(optParamsFrame, npointsAn, strats)
    XnotRow = copy.copy(optParamsFrame[0, :])
    XnotRow[0:2] = np.log(XnotRow[0:2])
    XnotRow[4::] = np.log(XnotRow[4::])
    XnotSelec = 7 / minSelecFunc(XnotRow, targMeans, offTargMeans)

    selecDF = pd.DataFrame({"Strat": ["Xnot", "Affinity", "Mixture", "Valency", "All"], "Selectivity": np.tile(XnotSelec, [5])})

    for i in range(0, optParamsFrame.shape[0]):
        optParamsR = optParamsFrame[i, :]
        strat = stratList[i]
        selecRow = copy.copy(optParamsR)
        selecRow[0:2] = np.log(selecRow[0:2])
        selecRow[4::] = np.log(selecRow[4::])
        selecDF.loc[(selecDF["Strat"] == strat), "Selectivity"] = (7 / minSelecFunc(selecRow, targMeans, offTargMeans))
        heatmapNorm(ax[0], targMeans[0], optParamsR[0], optParamsR[1], [[optParamsR[4], optParamsR[5]], [optParamsR[4], optParamsR[5]]],
                    [optParamsR[3], 1 - optParamsR[3]], f=optParamsR[2], vrange=vrange, cbar=True, layover=True, highlight=targetPop[0])
        ax[0].set(xlabel="Receptor 1 Abundance ($cell^{-1}$))", ylabel="Receptor 2 Abundance ($cell^{-1}$))")
        sns.barplot(x="Strat", y="Selectivity", data=selecDF, ax=ax[1])
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
