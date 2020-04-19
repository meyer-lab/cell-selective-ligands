"""
Contains utilities and functions that are commonly used in the figure creation files.
"""
from string import ascii_lowercase
from matplotlib import gridspec, pyplot as plt
from scipy.stats import multivariate_normal
import seaborn as sns
import pandas as pds
import numpy as np
from ..sampling import sampleSpec


def getSetup(figsize, gridd):
    """ Establish figure set-up with subplots. """
    sns.set(style="whitegrid", font_scale=0.7, color_codes=True, palette="colorblind", rc={"grid.linestyle": "dotted", "axes.linewidth": 0.6})

    # Setup plotting space and grid
    f = plt.figure(figsize=figsize, constrained_layout=True)
    gs1 = gridspec.GridSpec(*gridd, figure=f)

    # Get list of axis objects
    ax = list()
    for x in range(gridd[0] * gridd[1]):
        ax.append(f.add_subplot(gs1[x]))

    return (ax, f)


def subplotLabel(axs):
    """ Place subplot labels on figure. """
    for ii, ax in enumerate(axs):
        ax.text(-0.2, 1.25, ascii_lowercase[ii], transform=ax.transAxes, fontsize=16, fontweight="bold", va="top")


def PlotCellPops(ax, df, bbox=False):
    "Plots theoretical populations"
    sampleData = sampleReceptors(df, 20000)
    sns.set_palette("husl", 8)
    for pop in sampleData.Population.unique():
        popDF = sampleData.loc[sampleData["Population"] == pop]
        sns.kdeplot(popDF.Receptor_1, popDF.Receptor_2, ax=ax, label=pop, shade=True, shade_lowest=False, legend=False)
    if bbox:
        ax.legend(fontsize=6, bbox_to_anchor=(1.02, 1), loc="center right")
    else:
        ax.legend(fontsize=7)
    ax.set(xscale="log", yscale="log")


def sampleReceptors(df, nsample=100):
    """
    Generate samples in each sample space
    """
    Populations = df.Population.unique()
    sampledf = pds.DataFrame(columns=["Population", "Receptor_1", "Receptor_2"])
    for population in Populations:
        populationdf = df[df["Population"] == population]
        RtotMeans = np.array([populationdf.Receptor_1.to_numpy(), populationdf.Receptor_2.to_numpy()]).flatten()
        RtotCovs = populationdf.Covariance_Matrix.to_numpy()[0]
        pop = np.power(10.0, multivariate_normal.rvs(mean=RtotMeans, cov=RtotCovs, size=nsample))
        popdf = pds.DataFrame({"Population": population, "Receptor_1": pop[:, 0], "Receptor_2": pop[:, 1]})
        sampledf = sampledf.append(popdf)

    return sampledf


def getFuncDict():
    """Directs key word to given function"""
    FuncDict = {"Aff": affHeatMap,
                "Valency": ValencyPlot,
                "Mix": MixPlot}
    return FuncDict


def popCompare(ax, popList, df, scanKey, Kav, L0=1e-9, KxStar=10 ** -12.2, f=1):
    """Takes in populations and parameters to scan over and creates line plot"""
    funcDict = getFuncDict()
    recMeans, Covs = [], []
    Title = popList[0] + " to " + popList[1]
    for ii, pop in enumerate(popList):
        dfPop = df[df["Population"] == pop]
        recMeans.append(np.array([dfPop["Receptor_1"].to_numpy(), dfPop["Receptor_2"].to_numpy()]).flatten())
        Covs.append(dfPop.Covariance_Matrix.to_numpy()[0])
        if ii >= 2:
            Title += "/" + pop
    Title = Title + " binding ratio"
    funcDict[scanKey](ax, recMeans, Covs, Kav, L0, KxStar, f, Title)


def affHeatMap(ax, recMeans, Covs, Kav, L0, KxStar, f, Title, Cbar=True):
    "Makes a heatmap comparing binding ratios of populations at a range of binding affinities"
    npoints = 15
    ticks = np.full([npoints], None)
    affScan = np.logspace(Kav[0], Kav[1], npoints)
    ticks[0], ticks[-1] = "10e" + str(Kav[0] - 1), "10e" + str(Kav[1] - 1)

    sampMeans = np.zeros(npoints)
    ratioDF = pds.DataFrame(columns=affScan, index=affScan)

    for ii, aff1 in enumerate(affScan):
        for jj, aff2 in enumerate(affScan):
            _, sampMeans[jj], _ = sampleSpec(L0, KxStar, f, recMeans, Covs, np.array([1]), np.array([[aff1, aff2]]))
        ratioDF[ratioDF.columns[ii]] = sampMeans

    sns.heatmap(ratioDF, ax=ax, xticklabels=ticks, yticklabels=ticks, vmin=0, vmax=12.5, cbar=Cbar)
    ax.set(xlabel="Rec 1 Affinity ($K_a$)", ylabel="Rec 2 Affinity ($K_a$)")
    ax.set_title(Title, fontsize=8)


def ValencyPlot(ax, recMeans, Covs, Kav, L0, KxStar, f, Title):
    "Makes a line chart comparing binding ratios of populations at multiple valencies"
    assert len(f) > 1
    assert len(L0) > 1
    assert len(L0) == len(Kav)
    sampMeans, underDev, overDev = np.zeros_like(f), np.zeros_like(f), np.zeros_like(f)
    labels = ["Low Affinity/Conc", "Med Affinity/Conc", "High Affinity/Conc"]
    colors = ["lime", "blue", "red"]

    for ii, aff in enumerate(Kav):
        for jj, val in enumerate(f):
            underDev[jj], sampMeans[jj], overDev[jj] = sampleSpec(
                L0[ii], KxStar, val, [recMeans[0], recMeans[1]], [Covs[0], Covs[1]], np.array([1]), np.array([[aff, aff]])
            )

        ax.plot(f, sampMeans, color=colors[ii], label=labels[ii])
        ax.fill_between(f, underDev, overDev, color=colors[ii], alpha=0.1)
    ax.set(xlabel="Valency", ylabel="Binding Ratio", title=Title, xlim=(1, max(f)), ylim=(0, 150))
    ax.legend(prop={"size": 6})


def MixPlot(ax, recMeans, Covs, Kav, L0, KxStar, f, Title):
    "Makes a line chart comparing binding ratios of populations at multiple mixture compositions"
    npoints = 51
    sampMeans, underDev, overDev = np.zeros(npoints), np.zeros(npoints), np.zeros(npoints)
    mixRatio = np.linspace(0, 1, npoints)

    for jj, mixture1 in enumerate(mixRatio):
        underDev[jj], sampMeans[jj], overDev[jj] = sampleSpec(
            L0, KxStar, f, recMeans, Covs, np.array([mixture1, 1 - mixture1]), np.array([Kav[0], Kav[1]])
        )

    ax.plot(mixRatio, sampMeans, color="royalblue")
    ax.fill_between(mixRatio, underDev, overDev, color="royalblue", alpha=0.1)
    if len(Covs) == 2:
        ax.set(xlabel="Ligand 1 in Mixture", ylabel="Binding Ratio", ylim=(0, 10), xlim=(0, 1), title=Title + " binding ratio")
    else:
        ax.set(xlabel="Ligand 1 in Mixture", ylabel="Binding Ratio", ylim=(0, 4), xlim=(0, 1))
        ax.set_title(Title)
