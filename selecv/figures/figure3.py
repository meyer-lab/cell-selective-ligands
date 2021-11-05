"""
Figure 3. Heterovalent bispecific
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from .figureCommon import getSetup, subplotLabel, heatmap, cellPopulations, overlapCellPopulation
from valentbind import polyc, polyfc


pairs = [(r"$R_1^{hi}R_2^{lo}$", r"$R_1^{med}R_2^{lo}$"), (r"$R_1^{hi}R_2^{hi}$", r"$R_1^{med}R_2^{med}$"),
         (r"$R_1^{med}R_2^{hi}$", r"$R_1^{hi}R_2^{med}$"), (r"$R_1^{hi}R_2^{lo}$", r"$R_1^{lo}R_2^{hi}$")]


def makeFigure():
    """ main function for Figure 3 """
    ax, f = getSetup((10, 10), (3, 3))
    subplotLabel(ax, list(range(9)))
    fsize = 9.5

    L0 = 1e-8
    Kav = [[1e7, 1e5], [1e5, 1e6]]

    KxStar = 1e-12
    heatmap(ax[0], L0, KxStar, Kav, [1.0], Cplx=[[1, 1]], vrange=(-4, 7), fully=False,
            title="Bispecific Lbound, $K_x^*$={} cell·M".format(KxStar), cbar=False, layover=1)
    heatmap(ax[1], L0 * 2, KxStar, Kav, [0.5, 0.5], f=1, vrange=(-4, 7), fully=False,
            title="Mixture of monovalents Lbound, $K_x^*$={} cell·M".format(KxStar), cbar=False, layover=1)
    heatmap(ax[2], L0, KxStar, Kav, [0.5, 0.5], Cplx=[[2, 0], [0, 2]], vrange=(-4, 7), fully=False,
            title="Mixture of bivalents Lbound, $K_x^*$={} cell·M".format(KxStar), cbar=True, layover=1)

    for i, KxStar in enumerate([1e-10, 1e-12, 1e-14]):
        heatmap(ax[i + 3], L0, KxStar, Kav, [1.0], Cplx=[[1, 1]], vrange=(-4, 7), fully=True,
                title="Bispecific log fully bound with $K_x^*$={} cell·M".format(KxStar), cbar=(i == 2), layover=1)

    for i in range(6):
        ax[i].set(xlabel="Receptor 1 Abundance (#/cell)", ylabel='Receptor 2 Abundance (#/cell)')

    KxStarVary(ax[6], L0, Kav, ylim=(-9, 9), compare="tether")
    KxStarVary(ax[7], L0, Kav, ylim=(-9, 9), compare="bisp", fully=True)
    ax[8].axis("off")

    for subax in ax:
        yticks = subax.get_yticks()
        xticks = subax.get_xticks()
        subax.set_xticklabels(xticks, fontsize=fsize)
        subax.set_yticklabels(yticks, fontsize=fsize)
        subax.set_xlabel(subax.get_xlabel(), fontsize=fsize)
        subax.set_ylabel(subax.get_ylabel(), fontsize=fsize)
        subax.set_title(subax.get_title(), fontsize=fsize)

    return f


def tetheredYN(L0, KxStar, Rtot, Kav, fully=True):
    """ Compare tethered (bispecific) vs monovalent  """
    if fully:
        return polyc(L0, KxStar, Rtot, [[1, 1]], [1.0], Kav)[2][0] / \
            polyfc(L0 * 2, KxStar, 1, Rtot, [0.5, 0.5], Kav)[0]
    else:
        return polyc(L0, KxStar, Rtot, [[1, 1]], [1.0], Kav)[0][0] / \
            polyfc(L0 * 2, KxStar, 1, Rtot, [0.5, 0.5], Kav)[0]


def mixBispecYN(L0, KxStar, Rtot, Kav, fully=True):
    """ Compare bispecific to mixture of bivalent  """
    if fully:
        return polyc(L0, KxStar, Rtot, [[1, 1]], [1.0], Kav)[2][0] / \
            np.sum(polyc(L0, KxStar, Rtot, [[2, 0], [0, 2]], [0.5, 0.5], Kav)[2])
    else:
        return polyc(L0, KxStar, Rtot, [[1, 1]], [1.0], Kav)[0][0] / \
            np.sum(polyc(L0, KxStar, Rtot, [[2, 0], [0, 2]], [0.5, 0.5], Kav)[0])


def normHeatmap(ax, L0, KxStar, Kav, vrange=(-4, 2), title="", cbar=False, fully=True, layover=True, normby=tetheredYN):
    """ Make a heatmap normalized by another binding value """
    nAbdPts = 70
    abundRange = (1.5, 4.5)
    abundScan = np.logspace(abundRange[0], abundRange[1], nAbdPts)

    func = np.vectorize(lambda abund1, abund2: normby(L0, KxStar, [abund1, abund2], Kav, fully=fully))
    X, Y = np.meshgrid(abundScan, abundScan)
    logZ = np.log(func(X, Y))

    contours = ax.contour(X, Y, logZ, levels=np.arange(-20, 20, 0.5), colors="black", linewidths=0.5)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title(title)
    plt.clabel(contours, inline=True, fontsize=6)
    ax.pcolor(X, Y, logZ, cmap='RdYlGn', vmin=vrange[0], vmax=vrange[1])
    norm = plt.Normalize(vmin=vrange[0], vmax=vrange[1])
    if cbar:
        cbar = ax.figure.colorbar(cm.ScalarMappable(norm=norm, cmap='RdYlGn'), ax=ax)
        cbar.set_label("Log ratio")
    if layover:
        overlapCellPopulation(ax, abundRange)


def selectivity(pop1name, pop2name, L0, KxStar, Cplx, Ctheta, Kav, fully=True, untethered=False):
    """ Always calculate the full binding of the 1st kind of complex """
    pop1 = cellPopulations[pop1name][0], cellPopulations[pop1name][1]
    pop2 = cellPopulations[pop2name][0], cellPopulations[pop2name][1]
    if untethered:  # mixture of monovalent
        return polyfc(L0, KxStar, 1, np.power(10, pop1), [0.5, 0.5], Kav)[0] \
            / polyfc(L0, KxStar, 1, np.power(10, pop2), [0.5, 0.5], Kav)[0]
    if fully:
        return np.sum(polyc(L0, KxStar, np.power(10, pop1), Cplx, Ctheta, Kav)[2]) \
            / np.sum(polyc(L0, KxStar, np.power(10, pop2), Cplx, Ctheta, Kav)[2])
    else:
        return np.sum(polyc(L0, KxStar, np.power(10, pop1), Cplx, Ctheta, Kav)[0]) \
            / np.sum(polyc(L0, KxStar, np.power(10, pop2), Cplx, Ctheta, Kav)[0])


def KxStarVary(ax, L0, Kav, ylim=(-7, 5), fully=True, compare=None):
    """ Line plot for selectivity with different KxStar """
    nPoints = 50
    Kxaxis = np.logspace(-15, -7, nPoints)

    colors = ["royalblue", "orange", "limegreen", "orangered"]
    sHolder = np.zeros((nPoints))
    for i, pair in enumerate(pairs):
        for j, KxStar in enumerate(Kxaxis):
            if compare == "tether":
                sHolder[j] = selectivity(pair[0], pair[1], L0, KxStar, [[1, 1]], [1], Kav, fully=fully, untethered=False) \
                    / selectivity(pair[0], pair[1], L0 * 2, KxStar, None, None, Kav, untethered=True)
            elif compare == "bisp":
                sHolder[j] = selectivity(pair[0], pair[1], L0, KxStar, [[1, 1]], [1], Kav, fully=fully, untethered=False) \
                    / selectivity(pair[0], pair[1], L0, KxStar, [[2, 0], [0, 2]], [0.5, 0.5], Kav, fully=fully)
            elif compare == " fully":
                sHolder[j] = selectivity(pair[0], pair[1], L0, KxStar, [[1, 1]], [1], Kav, fully=True, untethered=False) \
                    / selectivity(pair[0], pair[1], L0, KxStar, [[1, 1]], [1], Kav, fully=False, untethered=False)
            else:
                sHolder[j] = np.log(selectivity(pair[0], pair[1], L0, KxStar, [[1, 1]], [1], Kav, fully=fully, untethered=False))
        ax.plot(Kxaxis, sHolder, color=colors[i], label=pair[0] + " to " + pair[1], linestyle="-")

    ax.set(xlim=(1e-15, 1e-7), ylim=ylim,
           xlabel="$K_x^*$")
    ax.set_xscale('log')
    if compare == "tether":
        ax.set_ylabel("Bispecific selectivity / Monovalent selectivity")
        ax.set_title("Bispecific advantage over monovalent mixture")
    elif compare == "bisp":
        ax.set_ylabel("Bispecific selectivity / Bivalent selectivity")
        ax.set_title("Bispecific advantage over homo-bivalent mixture")
    elif compare == "fully":
        ax.set_ylabel("Ratio of selectivity")
        ax.set_title("Fully bound selectivity / Ligand bound selectivity")
    else:
        ax.set_ylabel("Log selectivity of [1, 1]")
        if fully:
            ax.set_title("Log selectivity varies with $K_x^*$ for Lfbnd")
        else:
            ax.set_title("Log selectivity varies with $K_x^*$ for Lbound")
    ax.legend(loc='lower right', fancybox=True, framealpha=1)
