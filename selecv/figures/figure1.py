"""
Figure 1. Introduce the model system.
"""

import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from .figureCommon import subplotLabel, getSetup, heatmap, overlapCellPopulation
from ..sampling import sampleSpec, cellPopulations



ligConc = np.array([1e-8])
KxStarP = 1e-10
val = 16.0


def makeFigure():
    """ Make figure 1. """
    # Get list of axis objects
    ax, f = getSetup((10, 7), (2, 3))

    subplotLabel(ax, [0, 3, 4, 5])

    ax[0].axis("off")
    ax[1].axis("off")
    ax[2].axis("off")
    ax[3].axis("off")

    demoHeatmap(ax[4], vmin=1, vmax=5)
    demoPopulations(ax[5])
    #PlotCellPops(ax[5], getPopDict()[1])

    return f


def plotRealPops(ax, data, names):
    "Plot both real receptor abundances"
    for ii, cell in enumerate(names):
        ax.scatter(data[ii, 0], data[ii, 1], label=cell)
    ax.set(xlabel="IL2Rα Abundance", ylabel="IL-2Rβ Abundance", xscale="log", yscale="log")
    ax.legend()


def plotSampleConc(ax, df, concRange, popList):
    "Makes a line chart comparing binding ratios of populations at multiple concentrations"
    npoints = 100
    concScan = np.logspace(concRange[0], concRange[1], npoints)
    df1 = df[df["Population"] == popList[0]]
    df2 = df[df["Population"] == popList[1]]
    recMean1 = np.array([df1["Receptor_1"].to_numpy(), df1["Receptor_2"].to_numpy()]).flatten()
    recMean2 = np.array([df2["Receptor_1"].to_numpy(), df2["Receptor_2"].to_numpy()]).flatten()
    Cov1 = df1.Covariance_Matrix.to_numpy()[0]
    Cov2 = df2.Covariance_Matrix.to_numpy()[0]
    sampMeans, underDev, overDev = np.zeros(npoints), np.zeros(npoints), np.zeros(npoints)

    for ii, conc in enumerate(concScan):
        underDev[ii], sampMeans[ii], overDev[ii] = sampleSpec(
            conc, KxStarP, val, [recMean1, recMean2], [Cov1, Cov2], np.array([1]), np.array([[1e-8, 1e-8]])
        )

    sampMeans *= np.sum(np.power(10, recMean2)) / np.sum(np.power(10, recMean1))
    underDev *= np.sum(np.power(10, recMean2)) / np.sum(np.power(10, recMean1))
    overDev *= np.sum(np.power(10, recMean2)) / np.sum(np.power(10, recMean1))

    ax.plot(concScan, sampMeans, color="royalblue")
    ax.fill_between(concScan, underDev, overDev, color="royalblue", alpha=0.1)
    ax.set(xscale="log", xlim=(np.power(10, concRange[0]), np.power(10, concRange[1])), ylabel="Binding Ratio", xlabel="Concentration (nM)")


def affPlot(ax, affDF):
    """
    Plot affinities for both cytokines and antibodies.
    """
    sns.barplot(x="Receptor/Ligand Pair", y="Affinity", hue="Type", data=affDF, ax=ax)
    ax.set(yscale="log", ylabel="Affinity ($K_a$)")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=35, rotation_mode="anchor", ha="right", fontsize=7, position=(0, 0.02))
    ax.legend()


def demoHeatmap(ax, vmin=1, vmax=4):
    heatmap(ax, 1e-9, 1e-12, [[1e5, 1e7]], [1.0], f=1, vrange=(vmin, vmax), cbar=True, layover=False)

    ax.plot([10 ** 2, 10 ** 3], [10 ** 3, 10 ** 3], color="w")
    ax.plot([10 ** 2, 10 ** 2], [10 ** 3, 10 ** 4], color="w")

    ax_new = ax.twinx().twiny()
    ax_new.set_xscale("linear")
    ax_new.set_yscale("linear")
    ax_new.set_xticks([])
    ax_new.set_yticks([])
    ax_new.set_xlim((1.5, 4.5))
    ax_new.set_ylim((1.5, 4.5))

    ax_new.add_artist(plt.Circle((2, 3), 0.2, color='w'))
    ax_new.text(2, 3, "1", size=11, color='red', weight='semibold', horizontalalignment='center',
            verticalalignment='center', backgroundcolor='w')
    ax_new.add_artist(plt.Circle((3, 3), 0.2, color='w'))
    ax_new.text(3, 3, "2", size=11, color='red', weight='semibold', horizontalalignment='center',
            verticalalignment='center', backgroundcolor='w')
    ax_new.add_artist(plt.Circle((2, 4), 0.2, color='w'))
    ax_new.text(2, 4, "3", size=11, color='red', weight='semibold', horizontalalignment='center',
            verticalalignment='center', backgroundcolor='w')
    ax.set(xlabel="Receptor 1", ylabel="Receptor 2")


def demoPopulations(ax):
    ax.set_facecolor('darkgoldenrod')
    ax.set_xlim((10 ** 1.5, 10 ** 4.5))
    ax.set_ylim((10 ** 1.5, 10 ** 4.5))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set(xscale="log", yscale="log", xlabel="Receptor 1", ylabel="Receptor 2")
    overlapCellPopulation(ax, (1.5, 4.5), data=cellPopulations)
