"""
Contains utilities and functions that are commonly used in the figure creation files.
"""
from string import ascii_lowercase
from matplotlib import gridspec, pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.cm as cm
from scipy.stats import multivariate_normal
import seaborn as sns
import pandas as pds
import numpy as np
from ..sampling import sampleSpec
from ..model import polyc, polyfc


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


def subplotLabel(axs, indices=False):
    """ Place subplot labels on figure. """
    if not indices:
        for ii, ax in enumerate(axs):
            ax.text(-0.2, 1.25, ascii_lowercase[ii], transform=ax.transAxes, fontsize=16, fontweight="bold", va="top")
    else:
        for jj, index in enumerate(indices):
            axs[index].text(-0.2, 1.25, ascii_lowercase[jj], transform=axs[index].transAxes, fontsize=16, fontweight="bold", va="top")


def PlotCellPops(ax, df, bbox=False):
    "Plots theoretical populations"
    sampleData = sampleReceptors(df, 20000)
    sns.set_palette("husl", 8)
    for pop in sampleData.Population.unique():
        popDF = sampleData.loc[sampleData["Population"] == pop]
        plot = sns.kdeplot(popDF.Receptor_1, popDF.Receptor_2, ax=ax, label=pop, shade=True, shade_lowest=False, legend=False)
    plot.text(100, 100, r'$R_1^{lo}R_2^{lo}$', size='small', color='black', weight='semibold', horizontalalignment='center', verticalalignment='center')
    plot.text(1000, 100, r"$R_1^{med}R_2^{lo}$", size='small', color='black', weight='semibold', horizontalalignment='center', verticalalignment='center')
    plot.text(10000, 100, r"$R_1^{hi}R_2^{lo}$", size='small', color='black', weight='semibold', horizontalalignment='center', verticalalignment='center')
    plot.text(100, 10000, r"$R_1^{lo}R_2^{hi}$", size='small', color='black', weight='semibold', horizontalalignment='center', verticalalignment='center')
    plot.text(1250, 1250, r"$R_1^{med}R_2^{med}$", size='small', color='black', weight='semibold', horizontalalignment='center', verticalalignment='center')
    plot.text(10000, 10000, r"$R_1^{hi}R_2^{hi}$", size='small', color='black', weight='semibold', horizontalalignment='center', verticalalignment='center')
    plot.text(8000, 1000, r"$R_1^{hi}R_2^{med}$", size='small', color='black', weight='semibold', horizontalalignment='center', verticalalignment='center')
    plot.text(1000, 8000, r"$R_1^{med}R_2^{hi}$", size='small', color='black', weight='semibold', horizontalalignment='center', verticalalignment='center')
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


def popCompare(ax, popList, df, scanKey, Kav, L0=1e-9, KxStar=10 ** -10.0, f=1):
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
    npoints = 3
    ticks = np.full([npoints], None)
    affScan = np.logspace(Kav[0], Kav[1], npoints)
    ticks[0], ticks[-1] = "1e" + str(Kav[0]), "1e" + str(Kav[1])

    sampMeans = np.zeros(npoints)
    ratioDF = pds.DataFrame(columns=affScan, index=affScan)
    tens = np.array([10, 10])

    for ii, aff1 in enumerate(affScan):
        for jj, aff2 in enumerate(np.flip(affScan)):
            sampMeans[jj] = polyfc(L0, KxStar, f, np.power(10, recMeans[0]), [1], np.array([[aff1, aff2]]))[0] / polyfc(L0, KxStar, f, np.power(10, recMeans[1]), [1], np.array([[aff1, aff2]]))[0]
        ratioDF[ratioDF.columns[ii]] = sampMeans

    sns.heatmap(ratioDF, ax=ax, xticklabels=ticks, yticklabels=np.flip(ticks), cbar=Cbar, cbar_kws={'label': 'Binding Ratio'}, annot=True)
    ax.set(xlabel="Rec 1 Affinity ($K_a$)", ylabel="Rec 2 Affinity $K_a$)")
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
                L0[ii] / val, KxStar, val, [recMeans[0], recMeans[1]], [Covs[0], Covs[1]], np.array([1]), np.array([[aff, aff]])
            )

        ax.plot(f, sampMeans, color=colors[ii], label=labels[ii])
        ax.fill_between(f, underDev, overDev, color=colors[ii], alpha=0.1)
    ax.set(xlabel="Valency", ylabel="Binding Ratio", title=Title, xlim=(1, max(f)), ylim=(0, 100))
    ax.set_xticks((4, 8, 12, 16))
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
        ax.set(xlabel="Ligand 1 in Mixture", ylabel="Binding Ratio", ylim=(0, 10), xlim=(0, 1))  # , title=Title + " binding ratio")
        ax.set_title(Title + " ratio", fontsize=4)
    else:
        ax.set(xlabel="Ligand 1 in Mixture", ylabel="Binding Ratio", ylim=(0, 4), xlim=(0, 1))
        ax.set_title(Title, fontsize=7)


cellPopulations = {
    r"$R_1^{lo}R_2^{lo}$": [2, 2, 0.5, 0.25, 45],
    r"$R_1^{med}R_2^{lo}$": [3, 2, 0.5, 0.25, 0],
    r"$R_1^{hi}R_2^{lo}$": [4, 2, 0.5, 0.25, 0],
    r"$R_1^{lo}R_2^{hi}$": [2, 4, 0.3, 0.6, 0],
    r"$R_1^{med}R_2^{hi}$": [3.1, 3.9, 0.5, 0.25, 45],
    r"$R_1^{hi}R_2^{med}$": [3.9, 3.1, 0.5, 0.25, 45],
    r"$R_1^{hi}R_2^{hi}$": [4, 4, 0.5, 0.25, 45],
    r"$R_1^{med}R_2^{med}$": [3.1, 3.1, 0.25, 1, 45],
}

abundRange = (1.5, 4.5)


def overlapCellPopulation(ax, scale, data=cellPopulations):
    ax_new = ax.twinx().twiny()
    ax_new.set_xscale("linear")
    ax_new.set_yscale("linear")
    ax_new.set_xticks([])
    ax_new.set_yticks([])
    ax_new.set_xlim(scale)
    ax_new.set_ylim(scale)
    for label, item in data.items():
        ax_new.add_patch(Ellipse(xy=(item[0], item[1]),
                                 width=item[2],
                                 height=item[3],
                                 angle=item[4],
                                 facecolor="blue",
                                 fill=True,
                                 alpha=0.5,
                                 linewidth=1))
        ax_new.text(item[0], item[1], label,
                    horizontalalignment='center',
                    verticalalignment='center',
                    fontweight='bold',
                    color='white')


def abundHeatMap(ax, abundRange, L0, KxStar, Kav, Comp, f=None, Cplx=None, vmin=-2, vmax=4, cbar=False, layover=True):
    assert bool(f is None) != bool(Cplx is None)
    nAbdPts = 70
    abundScan = np.logspace(abundRange[0], abundRange[1], nAbdPts)

    if f is None:
        func = np.vectorize(lambda abund1, abund2: polyc(L0, KxStar, [abund1, abund2], Cplx, Comp, Kav)[0])
    else:
        func = np.vectorize(lambda abund1, abund2: polyfc(L0, KxStar, f, [abund1, abund2], Comp, Kav)[0])

    X, Y = np.meshgrid(abundScan, abundScan)
    logZ = np.log(func(X, Y))

    contours = ax.contour(X, Y, logZ, levels=np.arange(-10, 20, 0.3), colors="black", linewidths=0.5)
    ax.set_xscale("log")
    ax.set_yscale("log")
    plt.clabel(contours, inline=True, fontsize=3)
    ax.pcolor(X, Y, logZ, cmap='cividis', vmin=vmin, vmax=vmax)
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    if cbar:
        cbar = ax.figure.colorbar(cm.ScalarMappable(norm=norm, cmap='cividis'), ax=ax)
        cbar.set_label("Log Ligand Bound")
    if layover:
        overlapCellPopulation(ax, abundRange)


def affinity(fig, axs, L0, KxStar, Comp, ff=None, Cplx=None, vmin=-2, vmax=4):
    nAffPts = 3

    fig.suptitle("Lbound when $L_0$={}, $f$={}, $K_x^*$={:.2e}, $LigC$={}".format(L0, ff, KxStar, Comp))

    affRange = (5., 7.)
    affScan = np.logspace(affRange[0], affRange[1], nAffPts)
    for i1, aff1 in enumerate(affScan):
        for i2, aff2 in enumerate(np.flip(affScan)):
            cbar = False
            if i2 * nAffPts + i1 in [2, 5, 8]:
                cbar = True
            abundHeatMap(axs[i2 * nAffPts + i1], abundRange,
                         L0, KxStar, [[aff1, aff2]], Comp, f=ff, Cplx=Cplx,
                         vmin=vmin, vmax=vmax, cbar=cbar)
            plt.plot([3.3, 3.7], [2, 2], color="w", marker=2)
            plt.text(3.5, 2.1, "b", size='large', color='white', weight='semibold', horizontalalignment='center', verticalalignment='center')
            plt.plot([3.3, 3.7], [3.6, 3.2], color="w", marker=2)
            plt.text(3.4, 3.63, "c", size='large', color='white', weight='semibold', horizontalalignment='center', verticalalignment='center')
            plt.plot([3.3, 3.8], [3.2, 3.7], color="w", marker=2)
            plt.text(3.7, 3.85, "d", size='large', color='white', weight='semibold', horizontalalignment='center', verticalalignment='center')
            plt.plot([2, 3.7], [3.5, 2.2], color="w", marker=2)
            plt.text(2.3, 3.5, "e", size='large', color='white', weight='semibold', horizontalalignment='center', verticalalignment='center')
            axs[i2 * nAffPts + i1].set_title("$K_1$ = {:.1e} $K_2$ = {:.1e}".format(aff1, aff2))
    return fig


def valency(fig, axs, L0, KxStar, Comp, Kav=[[1e6, 1e5], [1e5, 1e6]], Cplx=None, vmin=-2, vmax=4):
    ffs = [1, 4, 16]

    fig.suptitle("Lbound when $L_0$={}, $Kav$={}, $K_x^*$={:.2e}, $LigC$={}".format(L0, Kav, KxStar, Comp))

    for i, v in enumerate(ffs):
        cbar = False
        if i in [2]:
            cbar = True
        abundHeatMap(axs[i], abundRange, L0, KxStar, Kav, Comp, f=v, Cplx=Cplx, vmin=vmin, vmax=vmax, cbar=cbar)
        plt.plot([3.32, 3.7], [2, 2], color="w", marker=2)
        plt.text(3.5, 2.1, "b", size='large', color='white', weight='semibold', horizontalalignment='center', verticalalignment='center')
        plt.plot([3.3, 3.8], [3.2, 3.7], color="w", marker=2)
        plt.text(3.65, 3.8, "c", size='large', color='white', weight='semibold', horizontalalignment='center', verticalalignment='center')
        plt.plot([3.1, 3.1], [3.4, 3.7], color="w", marker=1, markersize=4)
        plt.text(3.25, 3.55, "d", size='large', color='white', weight='semibold', horizontalalignment='center', verticalalignment='center')
        axs[i].set_title("$f$ = {}".format(v))

    return fig


def mixture(fig, axs, L0, KxStar, Kav=[[1e6, 1e5], [1e5, 1e6]], ff=5, vmin=-2, vmax=4):
    comps = [0.0, 0.2, 0.5, 0.8, 1.0]

    fig.suptitle("Lbound when $L_0$={}, $Kav$={}, $f$={}, $K_x^*$={:.2e}".format(L0, Kav, ff, KxStar))

    for i, comp in enumerate(comps):
        cbar = False
        if i in [2, 4]:
            cbar = True
        abundHeatMap(axs[i], abundRange, L0, KxStar, Kav, [comp, 1 - comp], f=ff, Cplx=None, vmin=vmin, vmax=vmax, cbar=cbar)
        axs[i].set_title("$LigC$ = [{}, {}]".format(comp, 1 - comp))

    return fig


def complex(L0, KxStar, Kav=[[1e6, 1e5], [1e5, 1e6]], vmin=-2, vmax=4):
    cplx = [0, 2, 4]
    axs, fig = getSetup((8, 8), (len(cplx), len(cplx)))

    fig.suptitle("Lbound when $L_0$={}, $Kav$={}, $K_x^*$={:.2e}".format(L0, Kav, KxStar))

    for i1, cplx1 in enumerate(cplx):
        for i2, cplx2 in enumerate(np.flip(cplx)):
            abundHeatMap(axs[i2 * len(cplx) + i1], abundRange, L0, KxStar, Kav, [1],
                         Cplx=[[cplx1, cplx2]], vmin=vmin, vmax=vmax)
            axs[i2 * len(cplx) + i1].set_title("$Cplx$ = [{}, {}]".format(cplx1, cplx2))
    return fig
