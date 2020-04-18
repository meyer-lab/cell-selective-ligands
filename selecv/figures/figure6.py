"""
Figure 6. PolyC vs PolyFc.
"""

import numpy as np
from .figureCommon import subplotLabel, getSetup, PlotCellPops
from ..imports import getPopDict
from ..sampling import sampleSpec, sampleSpecC

ligConc = np.array([10e-10])
KxStarP = 10e-11
val = 2.0


def makeFigure():
    """ Make figure 4. """
    # Get list of axis objects
    ax, f = getSetup((7, 8), (4, 3))

    subplotLabel(ax)

    affinities = np.array([[10e8, 10e1], [10e1, 10e8]])
    valencyScan = np.logspace(0.0, 5.0, base=2.0, num=10)
    _, populationsdf = getPopDict()
    #ValencyPlotC(ax[0], populationsdf, valencyScan, ["Pop3", "Pop4"], 0.5)
    #ValencyPlotC(ax[1], populationsdf, valencyScan, ["Pop6", "Pop5"], 0.5)
    #ValencyPlotC(ax[2], populationsdf, valencyScan, ["Pop6", "Pop8"], 0.5)
    ValencyPlotC(ax[0], populationsdf, valencyScan, ["Pop3", "Pop2"], 0.5)
    ValencyPlotC(ax[1], populationsdf, valencyScan, ["Pop7", "Pop8"], 0.5)
    ValencyPlotC(ax[2], populationsdf, valencyScan, ["Pop7", "Pop5"], 0.5)
    ValencyPlotC(ax[3], populationsdf, valencyScan, ["Pop3", "Pop2"], 0.9)
    ValencyPlotC(ax[4], populationsdf, valencyScan, ["Pop7", "Pop8"], 0.9)
    ValencyPlotC(ax[5], populationsdf, valencyScan, ["Pop7", "Pop5"], 0.9)
    
    
    PlotCellPops(ax[6], populationsdf)
    CPlot(ax[7], populationsdf, ["Pop3", "Pop2"], affinities, 11)
    ax[1].set_ylim(0, 50)
    CPlot(ax[8], populationsdf, ["Pop7", "Pop5", "Pop6"], affinities, 11)
    CPlot(ax[9], populationsdf, ["Pop8", "Pop5", "Pop6"], affinities, 11)
    CPlot(ax[10], populationsdf, ["Pop7", "Pop8"], affinities, 11)
    CPlot(ax[11], populationsdf, ["Pop7", "Pop5"], affinities, 11)

    return f


def CPlot(ax, df, popList, affinities, npoints):
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
    sampMeansC, underDevC, overDevC = np.zeros(npoints), np.zeros(npoints), np.zeros(npoints)
    mixRatio = np.linspace(0, 1, npoints)
    

    for jj, mixture1 in enumerate(mixRatio):
        underDev[jj], sampMeans[jj], overDev[jj] = sampleSpec(
            val * ligConc, KxStarP, val, recMeans, Covs, np.array([mixture1, 1 - mixture1]), np.array([affinities[0], affinities[1]])
        )
        underDevC[jj], sampMeansC[jj], overDevC[jj] = sampleSpecC(
            ligConc, KxStarP, recMeans, Covs, np.array([[mixture1 * val, (1 - mixture1) * val]]), np.array([1.0]), np.array([affinities[0], affinities[1]])
        )

    ax.plot(mixRatio, sampMeans, color="royalblue", label="polyfc")
    ax.fill_between(mixRatio, underDev, overDev, color="royalblue", alpha=0.1)
    ax.plot(mixRatio, sampMeansC, color="orangered", label="polyc")
    ax.fill_between(mixRatio, underDevC, overDevC, color="orangered", alpha=0.1)
    ax.legend()
    if len(popList) == 2:
        ax.set(xlabel="Ligand 1 in Mixture", ylabel="Binding Ratio", ylim=(0, 10), xlim=(0, 1), title=Title + " binding ratio")
    else:
        ax.set(xlabel="Ligand 1 in Mixture", ylabel="Binding Ratio", ylim=(0, 4), xlim=(0, 1))
        ax.set_title(Title + " binding ratio", fontsize=8 - 0.4 * len(popList))

        
def ValencyPlotC(ax, df, valencies, popList, perc):
    "Makes a line chart comparing binding ratios of populations at multiple valencies"
    ligconc = 10e-11
    df1 = df[df["Population"] == popList[0]]
    df2 = df[df["Population"] == popList[1]]
    recMean1 = np.array([df1["Receptor_1"].to_numpy(), df1["Receptor_2"].to_numpy()]).flatten()
    recMean2 = np.array([df2["Receptor_1"].to_numpy(), df2["Receptor_2"].to_numpy()]).flatten()
    Cov1 = df1.Covariance_Matrix.to_numpy()[0]
    Cov2 = df2.Covariance_Matrix.to_numpy()[0]
    sampMeans, underDev, overDev = np.zeros_like(valencies), np.zeros_like(valencies), np.zeros_like(valencies)
    sampMeansC, underDevC, overDevC = np.zeros_like(valencies), np.zeros_like(valencies), np.zeros_like(valencies)
    #affinities = [10e5, 10e6, 10e7]
    concs = [10e-11, 10e-10, 10e-9]
    labels = ["Low Affinity", "High Affinity"]
    colors = ["lime", "blue"]
    affinities = np.array([[[10e6, 10e1], [10e1, 10e6]], [[10e8, 10e1], [10e1, 10e8]]])
    

    for ii, aff in enumerate(affinities):
        for jj, val in enumerate(valencies):
            underDev[jj], sampMeans[jj], overDev[jj] = sampleSpec(val * ligConc, KxStarP, val, [recMean1, recMean2], [Cov1, Cov2], np.array([perc, 1-perc]), np.array([aff[0], aff[1]]))
            underDevC[jj], sampMeansC[jj], overDevC[jj] = sampleSpecC(ligConc, KxStarP, [recMean1, recMean2], [Cov1, Cov2], np.array([[val, 0], [0, val]]), np.array([perc, 1-perc]), np.array([aff[0], aff[1]]))

        ax.plot(valencies, sampMeans, color=colors[ii], label=labels[ii] + "PolyFc")
        ax.fill_between(valencies, underDev, overDev, color=colors[ii], alpha=0.1)
        ax.plot(valencies, sampMeansC, color=colors[ii], label=labels[ii] + "PolyC", linestyle=":")
        ax.fill_between(valencies, underDevC, overDevC, color=colors[ii], alpha=0.1, linestyle=":")

    ax.set(xlabel="Valency", ylabel="Binding Ratio", title=popList[0] + " to " + popList[1] + " binding ratio", xlim=(1, 32), ylim=(0, 30))
    ax.legend(prop={"size": 6})