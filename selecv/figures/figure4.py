"""
Figure 4. Mixtures for enhanced targeting.
"""

import numpy as np
from .figureCommon import subplotLabel, getSetup, PlotCellPops
from ..imports import getPopDict
from ..sampling import sampleSpec

ligConc = np.array([10e-9])
KxStarP = 10e-11
val = 1.0


def makeFigure():
    """ Make figure 4. """
    # Get list of axis objects
    ax, f = getSetup((7, 4), (2, 3))

    subplotLabel(ax)

    affinities = np.array([[10e8, 10e1], [10e1, 10e8]])
    _, populationsdf = getPopDict()
    PlotCellPops(ax[0], populationsdf)
    MixPlot(ax[1], populationsdf, ["Pop3", "Pop2"], affinities, 101)
    MixPlot(ax[2], populationsdf, ["Pop5", "Pop3", "Pop4"], affinities, 101)
    MixPlot(ax[3], populationsdf, ["Pop6", "Pop3", "Pop4"], affinities, 101)
    MixPlot(ax[4], populationsdf, ["Pop7", "Pop3", "Pop4"], affinities, 101)
    MixPlot(ax[5], populationsdf, ["Pop8", "Pop5", "Pop6"], affinities, 101)

    return f


def MixPlot(ax, df, popList, affinities, npoints):
    "Makes a line chart comparing binding ratios of populations at multiple mixture compositions"
    recMeans, Covs = [], []
    Title = popList[0] + " to " + popList[1]
    for ii, pop in enumerate(popList):
        dfPop = df[df["Population"] == pop]
        recMeans.append(np.array([dfPop["Receptor_1"].to_numpy(), dfPop["Receptor_2"].to_numpy()]).flatten())
        Covs.append(dfPop.Covariance_Matrix.to_numpy()[0])
        if ii >= 2:
            Title += "/" + pop

    sampMeans, underDev, overDev = np.zeros(npoints), np.zeros(npoints), np.zeros(npoints)
    mixRatio = np.linspace(0, 1, npoints)

    for jj, mixture1 in enumerate(mixRatio):
        underDev[jj], sampMeans[jj], overDev[jj] = sampleSpec(
            ligConc, KxStarP, val, recMeans, Covs, np.array([mixture1, 1 - mixture1]), np.array([affinities[0], affinities[1]])
        )

    ax.plot(mixRatio, sampMeans, color="royalblue")
    ax.fill_between(mixRatio, underDev, overDev, color="royalblue", alpha=0.1)
    if len(popList) == 2:
        ax.set(xlabel="Ligand 1 in Mixture", ylabel="Binding Ratio", ylim=(0, 10), xlim=(0, 1), title=Title + " binding ratio")
    else:
        ax.set(xlabel="Ligand 1 in Mixture", ylabel="Binding Ratio", ylim=(0, 4), xlim=(0, 1))
        ax.set_title(Title + " binding ratio", fontsize=8 - 0.4 * len(popList))
