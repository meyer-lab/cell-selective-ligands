"""
Contains utilities and functions that are commonly used in the figure creation files.
"""
import sys
from logging import basicConfig, INFO, info
from time import time
from string import ascii_lowercase
from matplotlib import gridspec, pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.cm as cm
from matplotlib import rcParams
import matplotlib.ticker as mticker
from scipy.stats import multivariate_normal
import seaborn as sns
import pandas as pds
import numpy as np
import svgutils.transform as st
from ..sampling import sampleSpec, cellPopulations
from valentbind import polyc, polyfc
import matplotlib
matplotlib.use('AGG')

fdir = './output/'

rcParams['pcolor.shading'] = 'auto'
rcParams['svg.fonttype'] = 'none'

LR, HR = 1.5, 6.5


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


def setFontSize(ax, fsize, xsci=[], ysci=[], nolegend=[]):
    for i, subax in enumerate(ax):
        subax.tick_params(axis="x", labelsize=fsize - 1)
        subax.tick_params(axis="y", labelsize=fsize - 1)
        subax.set_xlabel(subax.get_xlabel(), fontsize=fsize)
        subax.set_ylabel(subax.get_ylabel(), fontsize=fsize)
        subax.set_title(subax.get_title(), fontsize=fsize)
        if subax.get_legend() is not None and i not in nolegend:
            subax.legend(prop={'size': fsize - 1})
        if i in xsci:
            subax.set_xscale('log')
        if i in ysci:
            subax.set_yscale('log')


def overlayCartoon(figFile, cartoonFile, x, y, scalee=1, scale_x=1, scale_y=1):
    """ Add cartoon to a figure file. """

    # Overlay Figure cartoons
    template = st.fromfile(figFile)
    cartoon = st.fromfile(cartoonFile).getroot()

    cartoon.moveto(x, y, scale_x=scalee * scale_x, scale_y=scalee * scale_y)

    template.append(cartoon)
    template.save(figFile)


def genFigure():
    basicConfig(format='%(levelname)s:%(message)s', level=INFO)
    start = time()
    nameOut = 'figure' + sys.argv[1]

    exec('from .' + nameOut + ' import makeFigure', globals())
    ff = makeFigure()
    ff.savefig(fdir + nameOut + '.svg', dpi=ff.dpi, bbox_inches='tight', pad_inches=0)

    if sys.argv[1] == '1':
        # Overlay Figure 1 cartoon
        overlayCartoon(fdir + 'figure1.svg',
                       './selecv/graphics/figure_1a.svg', 10, 15, scalee=0.02, scale_x=0.45, scale_y=0.45)
        overlayCartoon(fdir + 'figure1.svg',
                       './selecv/graphics/figure_1b.svg', 0, 280, scalee=0.24, scale_x=1, scale_y=1)
    if sys.argv[1] == '2':
        overlayCartoon(fdir + 'figure2.svg',
                       './selecv/graphics/figure_2a.svg', 10, 0, scalee=0.18, scale_x=1, scale_y=1)
    if sys.argv[1] == '3':
        overlayCartoon(fdir + 'figure3.svg',
                       './selecv/graphics/figure_3a.svg', 30, 0, scalee=0.22, scale_x=1, scale_y=1)
    if sys.argv[1] == '4':
        overlayCartoon(fdir + 'figure4.svg',
                       './selecv/graphics/figure_4a.svg', 10, 0, scalee=0.18, scale_x=1, scale_y=1)

    info('%s is done after %s seconds.', nameOut, time() - start)


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


def popCompare(ax, popList, scanKey, Kav, L0=1e-9, KxStar=1e-10, f=1):
    """Takes in populations and parameters to scan over and creates line plot"""
    funcDict = getFuncDict()
    Title = popList[0] + " to " + popList[1]
    for ii, pop in enumerate(popList):
        if ii >= 2:
            Title += "/" + pop
    Title = Title + " binding ratio"
    funcDict[scanKey](ax, popList, Kav, L0, KxStar, f, Title)


def affHeatMap(ax, names, Kav, L0, KxStar, f, Title, Cbar=True):
    "Makes a heatmap comparing binding ratios of populations at a range of binding affinities"
    npoints = 3
    ticks = np.full([npoints], None)
    affScan = np.logspace(Kav[0], Kav[1], npoints)
    ticks[0], ticks[-1] = "${}$".format(int(10**(9 - Kav[0]))), "${}$".format(int(10**(9 - Kav[1])))

    sampMeans = np.zeros(npoints)
    ratioDF = pds.DataFrame(columns=affScan, index=affScan)

    for ii, aff1 in enumerate(affScan):
        for jj, aff2 in enumerate(np.flip(affScan)):
            recMeans0 = np.array([cellPopulations[names[0]][0], cellPopulations[names[0]][1]])
            recMeans1 = np.array([cellPopulations[names[1]][0], cellPopulations[names[1]][1]])
            sampMeans[jj] = polyfc(L0, KxStar, f, np.power(10, recMeans0), [1], np.array([[aff1, aff2]]))[0] / polyfc(L0, KxStar, f, np.power(10, recMeans1), [1], np.array([[aff1, aff2]]))[0]
        ratioDF[ratioDF.columns[ii]] = sampMeans

    if ratioDF.max().max() < 15:
        sns.heatmap(ratioDF, ax=ax, xticklabels=ticks, yticklabels=np.flip(ticks), vmin=0, vmax=10, cbar=Cbar, cbar_kws={'label': 'Binding Ratio'}, annot=True)
    else:
        max = np.round(np.ceil(ratioDF.max().max() / 10) * 10, -1)
        sns.heatmap(ratioDF, ax=ax, xticklabels=ticks, yticklabels=np.flip(ticks), vmin=0, vmax=max, cbar=Cbar, cbar_kws={'label': 'Binding Ratio'}, annot=True)
    ax.set(xlabel="Rec 1 Affinity ($K_d$ [nM])", ylabel="Rec 2 Affinity ($K_d$ [nM])")
    ax.set_title(Title, fontsize=10)


def ValencyPlot(ax, names, Kav, L0, KxStar, f, Title):
    "Makes a line chart comparing binding ratios of populations at multiple valencies"
    assert len(f) > 1
    sampMeans, underDev, overDev = np.zeros_like(f), np.zeros_like(f), np.zeros_like(f)
    labels = [r"Low $R_1$ Affinity", r"Med $R_1$ Affinity", r"High $R_1$ Affinity"]
    colors = ["lime", "blue", "red"]

    for ii, aff in enumerate(Kav):
        for jj, val in enumerate(f):
            underDev[jj], sampMeans[jj], overDev[jj] = sampleSpec(L0 / val, KxStar, val, names, np.array([1]), np.array([[aff, 0.01]]))

        ax.plot(f, sampMeans, color=colors[ii], label=labels[ii])
        ax.fill_between(f, underDev, overDev, color=colors[ii], alpha=0.1)
    ax.set(xlabel="Valency", ylabel="Binding Ratio", title=Title, xlim=(1, max(f)), ylim=(0, 480))
    ax.set_xticks((2, 4, 6, 8))
    #ax.set_xticks((4, 8, 12, 16))
    ax.legend(prop={"size": 7})


def MixPlot(ax, names, Kav, L0, KxStar, f, Title):
    "Makes a line chart comparing binding ratios of populations at multiple mixture compositions"
    npoints = 51
    sampMeans, underDev, overDev = np.zeros(npoints), np.zeros(npoints), np.zeros(npoints)
    mixRatio = np.linspace(0, 1, npoints)

    for jj, mixture1 in enumerate(mixRatio):
        underDev[jj], sampMeans[jj], overDev[jj] = sampleSpec(L0, KxStar, f, names, np.array([mixture1, 1 - mixture1]), np.array([Kav[0], Kav[1]]))

    ax.plot(mixRatio, sampMeans, color="royalblue")
    ax.fill_between(mixRatio, underDev, overDev, color="royalblue", alpha=0.1)
    ax.set_xticks(np.linspace(0, 1, 5))
    ax.set_xticklabels(["{}%".format(x) for x in np.linspace(0, 100, 5)])
    if len(names) == 2:
        ax.set(xlabel="Ligand 1 in Mixture", ylabel="Binding Ratio", ylim=(0, 96), xlim=(0, 1))  # , title=Title + " binding ratio")
        ax.set_title(Title, fontsize=8)
        ax.grid()
    else:
        ax.set(xlabel="Ligand 1 in Mixture", ylabel="Binding Ratio", ylim=(0, 20), xlim=(0, 1))
        ax.set_title(Title, fontsize=8)
        ax.grid()


def overlapCellPopulation(ax, scale, data=cellPopulations, highlight=[], lowlight=[], recFactor=0.0, pname=True):
    ax_new = ax.twinx().twiny()
    ax_new.set_xscale("linear")
    ax_new.set_yscale("linear")
    ax_new.set_xticks([])
    ax_new.set_yticks([])
    ax_new.set_xlim(scale)
    ax_new.set_ylim(scale)
    for label, item in data.items():
        if not lowlight or label in [highlight[0], lowlight[0]]:
            color = "dimgray"
            if label in highlight:
                color = "red"
            ax_new.add_patch(Ellipse(xy=(item[0] + recFactor, item[1] + recFactor),
                                     width=item[2],
                                     height=item[3],
                                     angle=item[4],
                                     edgecolor=None,
                                     facecolor=color,
                                     fill=True,
                                     alpha=0.9,
                                     linewidth=0.0))
            if pname:
                ax_new.text(item[0] + recFactor, item[1] + recFactor, label,
                            horizontalalignment='center',
                            verticalalignment='center',
                            fontsize=11.3,
                            fontweight='bold',
                            color='black')
                ax_new.text(item[0] + recFactor, item[1] + recFactor, label,
                            horizontalalignment='center',
                            verticalalignment='center',
                            fontsize=11,
                            fontweight='light',
                            color='white')


def heatmap(ax, L0, KxStar, Kav, Comp, f=None, Cplx=None, vrange=(-2, 4), title="", cbar=False, layover=2, fully=False, highlight=[]):
    assert bool(f is None) != bool(Cplx is None)
    nAbdPts = 70
    abundRange = (LR, HR)
    abundScan = np.logspace(abundRange[0], abundRange[1], nAbdPts)

    if f is None:
        if fully:
            func = np.vectorize(lambda abund1, abund2: np.sum(polyc(L0, KxStar, [abund1, abund2], Cplx, Comp, Kav)[2]))
        else:
            func = np.vectorize(lambda abund1, abund2: np.sum(polyc(L0, KxStar, [abund1, abund2], Cplx, Comp, Kav)[0]))
    else:
        func = np.vectorize(lambda abund1, abund2: polyfc(L0, KxStar, f, [abund1, abund2], Comp, Kav)[0])

    X, Y = np.meshgrid(abundScan, abundScan)
    logZ = np.log(func(X, Y))

    vmed = int((vrange[0] + vrange[1]) / 2)
    contours0 = ax.contour(X, Y, logZ, levels=np.arange(-20, vmed, 1), colors="white", linewidths=0.5)
    contours1 = ax.contour(X, Y, logZ, levels=np.arange(vmed, 20, 1), colors="black", linewidths=0.5)
    ax.set_xscale("log")
    ax.set_yscale("log")

    ax.yaxis.set_major_formatter(mticker.ScalarFormatter(useOffset=False, useMathText=True))

    ax.set_title(title)
    plt.clabel(contours0, inline=True, fontsize=8)
    plt.clabel(contours1, inline=True, fontsize=8)
    ax.pcolor(X, Y, logZ, cmap='viridis', vmin=vrange[0], vmax=vrange[1])
    norm = plt.Normalize(vmin=vrange[0], vmax=vrange[1])
    if cbar:
        cbar = ax.figure.colorbar(cm.ScalarMappable(norm=norm, cmap='viridis'), ax=ax)
        cbar.set_label("Log Ligand Bound")
    # layover: 2 = with name; 1 = only pop w/o name; 0 = none
    if layover == 2:
        overlapCellPopulation(ax, abundRange, highlight=highlight, pname=True)
    elif layover == 1:
        overlapCellPopulation(ax, abundRange, highlight=highlight, pname=False)


def heatmapNorm(ax, R0, L0, KxStar, Kav, Comp, f=None, Cplx=None, vrange=(0, 5), title="", cbar=False, layover=2, highlight=[], lineN=101, recFactor=1.0):
    assert bool(f is None) != bool(Cplx is None)
    nAbdPts = 70
    abundRange = (LR + np.log10(recFactor), HR + np.log10(recFactor))
    abundScan = np.logspace(abundRange[0], abundRange[1], nAbdPts)

    if f is None:
        func = np.vectorize(lambda abund1, abund2: polyc(L0, KxStar, [abund1, abund2], Cplx, Comp, Kav)[2][0])
    else:
        func = np.vectorize(lambda abund1, abund2: polyfc(L0, KxStar, f, [abund1, abund2], Comp, Kav)[0])

    func0 = func(10**(R0[0] + np.log10(recFactor)), 10**(R0[1] + np.log10(recFactor)))
    X, Y = np.meshgrid(abundScan, abundScan)
    Z = func(X, Y) / func0

    contours1 = ax.contour(X, Y, Z, levels=np.logspace(0, 10, (lineN - 1) // 2 + 1)[1:], colors="black", linewidths=0.5)
    contours0 = ax.contour(X, Y, Z, levels=np.logspace(-10, 0, (lineN - 1) // 2 + 1), colors="white", linewidths=0.5)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title(title)
    plt.clabel(contours1, inline=True, fontsize=8, fmt="%3.1g")
    plt.clabel(contours0, inline=True, fontsize=8, fmt="%3.1g")
    ax.pcolor(X, Y, Z, cmap='viridis', vmin=vrange[0], vmax=vrange[1])
    norm = plt.Normalize(vmin=vrange[0], vmax=vrange[1])
    if cbar:
        cbar = ax.figure.colorbar(cm.ScalarMappable(norm=norm, cmap='viridis'), ax=ax)
        cbar.set_label("Relative Ligand Bound")
    # layover: 2 = with name; 1 = only pop w/o name; 0 = none
    if layover == 2:
        overlapCellPopulation(ax, abundRange, highlight=highlight, recFactor=np.log10(recFactor), pname=True)
    elif layover == 1:
        overlapCellPopulation(ax, abundRange, highlight=highlight, recFactor=np.log10(recFactor), pname=False)
