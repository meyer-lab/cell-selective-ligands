"""
Figure 3. Exploration of Valency.
"""
import numpy as np
from matplotlib.lines import Line2D
from .figureCommon import subplotLabel, getSetup, PlotCellPops, popCompare
from ..imports import getPopDict
from ..model import polyfc

ligConc = np.array([10e-9])
KxStarP = 10e-11
affinity = 10e7  # 7


def makeFigure():
    """ Make figure 3. """
    # Get list of axis objects
    ax, f = getSetup((7, 6), (3, 3))
    subplotLabel(ax)

    valencyScan = np.logspace(0.0, 4.0, base=2.0, num=10)
    _, populationsdf = getPopDict()
    ax[8].axis("off")

    PlotCellPops(ax[0], populationsdf)
    valDemo(ax[1])
    ConcValPlot(ax[2])
    popCompare(ax[3], ["Pop3", "Pop2"], populationsdf, "Valency", Kav=[10e5, 10e6, 10e7], L0=[10e-10, 10e-9, 10e-8], f=valencyScan)
    popCompare(ax[4], ["Pop7", "Pop8"], populationsdf, "Valency", Kav=[10e5, 10e6, 10e7], L0=[10e-10, 10e-9, 10e-8], f=valencyScan)
    popCompare(ax[5], ["Pop6", "Pop8"], populationsdf, "Valency", Kav=[10e5, 10e6, 10e7], L0=[10e-10, 10e-9, 10e-8], f=valencyScan)
    #popCompare(ax[6], ["Pop5", "Pop4"], populationsdf, "Valency", Kav=[10e5, 10e6, 10e7], L0=[10e-10, 10e-9, 10e-8], KxStar=10e-11, f=valencyScan)
    #ax[6].set_ylim(0, 2)
    popCompare(ax[6], ["Pop5", "Pop6"], populationsdf, "Valency", Kav=[10e5, 10e6, 10e7], L0=[10e-10, 10e-9, 10e-8], KxStar=10e-11, f=valencyScan)
    ax[6].set_ylim(0, 2)
    popCompare(ax[7], ["Pop3", "Pop4"], populationsdf, "Valency", Kav=[10e5, 10e6, 10e7], L0=[10e-10, 10e-9, 10e-8], KxStar=10e-11, f=valencyScan)
    ax[7].set_ylim(0, 2)

    return f


def valDemo(ax):
    "Demonstrate effect of valency"
    affs = [10e7, 10e6]
    colors = ["royalblue", "orange", "limegreen", "orangered"]
    lines = ["-", ":"]
    nPoints = 100
    recScan = np.logspace(0, 8, nPoints)
    labels = ["Monovalent", "Bivalent", "Trivalent", "Tetravalent"]
    percHold = np.zeros(nPoints)
    for ii, aff in enumerate(affs):
        for jj, valencyLab in enumerate(labels):
            for kk, recCount in enumerate(recScan):
                percHold[kk] = polyfc(ligConc / (jj + 1), KxStarP, jj + 1, recCount, [1], np.array([[aff]]))[0] / recCount
            ax.plot(recScan, percHold, label=valencyLab, linestyle=lines[ii], color=colors[jj])

    ax.set(xlim=(1, 100000000), xlabel="Receptor Abundance", ylabel="Lig bound / Receptor", xscale="log")  # ylim=(0, 1),
    handles, _ = ax.get_legend_handles_labels()
    handles = handles[0:4]
    line = Line2D([], [], color="black", marker="_", linestyle="None", markersize=6, label="High Affinity")
    point = Line2D([], [], color="black", marker=".", linestyle="None", markersize=6, label="Low Affinity")
    handles.append(line)
    handles.append(point)
    ax.legend(handles=handles, prop={"size": 6})


def ConcValPlot(ax):
    "Keep valency constant and high - vary concentration"
    concScan = np.logspace(-11, -7, 5)
    valency = 4
    recScan = np.logspace(0, 8, 100)
    percHold = np.zeros(100)

    for conc in concScan:
        for jj, recCount in enumerate(recScan):
            percHold[jj] = polyfc(conc / valency, KxStarP, valency, recCount, [1], np.array([[affinity]]))[0] / recCount
        ax.plot(recScan, percHold, label=str(conc * 10e8) + " nM")

    ax.set(xlim=(1, 100000000), xlabel="Receptor Abundance", ylabel="Lig Bound / Receptor", xscale="log")  # ylim=(0, 1),
    ax.legend(prop={"size": 6})
