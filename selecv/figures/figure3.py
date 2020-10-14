"""
Figure 3. Exploration of Valency.
"""
import numpy as np
import pandas as pds
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from .figureCommon import subplotLabel, getSetup, popCompare, heatmap
from ..model import polyfc

ligConc = np.array([1e-8])
KxStarP = 1e-10
affinity = 1e8  # 7


def makeFigure():
    """ Make figure 3. """
    # Get list of axis objects
    ax, f = getSetup((9, 9), (3, 3))
    subplotLabel(ax, [0] + list(range(3, 9)))

    valency(f, ax[0:3], 1e-9, 10 ** -10, [1.0], Kav=[[1e7, 0.01]], vmin=0.0, vmax=9)
    valencyScan = np.logspace(0.0, 4.0, base=2.0, num=10)
    popCompare(ax[3], [r"$R_1^{hi}R_2^{lo}$", r"$R_1^{med}R_2^{lo}$"], "Valency", Kav=[1e6, 1e7, 1e8], L0=[1e-8], f=valencyScan)
    popCompare(ax[4], [r"$R_1^{hi}R_2^{hi}$", r"$R_1^{med}R_2^{med}$"], "Valency", Kav=[1e6, 1e7, 1e8], L0=[1e-8], f=valencyScan)
    popCompare(ax[5], [r"$R_1^{hi}R_2^{med}$", r"$R_1^{med}R_2^{med}$"], "Valency", Kav=[1e6, 1e7, 1e8], L0=[1e-8], f=valencyScan)
    vieqPlot(ax[6], 1e4, 8)
    vieqPlot(ax[7], 1e3, 8)
    ratePlot(ax[8])

    return f


def valency(fig, axs, L0, KxStar, Comp, Kav=[[1e6, 1e5], [1e5, 1e6]], Cplx=None, vmin=-2, vmax=4):
    ffs = [1, 4, 16]

    for i, v in enumerate(ffs):
        cbar = False
        if i in [2]:
            cbar = True
        heatmap(axs[i], L0, KxStar, Kav, Comp, f=v, Cplx=Cplx, vrange=(vmin, vmax), cbar=cbar)
        axs[i].set(xlabel="Receptor 1 Abundance (#/cell)", ylabel='Receptor 2 Abundance (#/cell)')
        plt.plot([3.32, 3.7], [2, 2], color="w", marker=2)
        plt.text(3.5, 2.1, "b", size='large', color='white', weight='semibold', horizontalalignment='center', verticalalignment='center')
        plt.plot([3.3, 3.8], [3.2, 3.7], color="w", marker=2)
        plt.text(3.65, 3.8, "c", size='large', color='white', weight='semibold', horizontalalignment='center', verticalalignment='center')
        plt.plot([3.4, 3.6], [3, 3], color="w", marker=1, markersize=4)
        plt.text(3.6, 2.8, "d", size='large', color='white', weight='semibold', horizontalalignment='center', verticalalignment='center')
        axs[i].set_title("Valency = {}".format(v))

    return fig


def valDemo(ax):
    "Demonstrate effect of valency"
    affs = [1e8, 1e7]
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
        ax.plot(recScan, percHold, label=str(conc * 1e9) + " nM")

    ax.set(xlim=(1, 100000000), xlabel="Receptor Abundance", ylabel="Lig Bound / Receptor", xscale="log")  # ylim=(0, 1),
    ax.legend(prop={"size": 6})


def vieqPlot(ax, recCount, val):
    "Demonstrate effect of valency"
    vieqDF = pds.DataFrame(columns=["Degree of Binding", "# Ligand Bound", "$K_d$ nM"])
    Conc = 1e-9
    affs = [1e8, 1e7, 1e6]
    afflabs = ["10", "100", "1000"]
    for ii, aff in enumerate(affs):
        vieq = polyfc(Conc / (val), KxStarP, val, recCount, [1], np.array([[aff]]))[2]  # val + 1
        for jj, bound in enumerate(vieq):
            ligboundDF = pds.DataFrame({"Degree of Binding": jj + 1, "# Ligand Bound": [bound], "$K_d$ nM": afflabs[ii]})
            vieqDF = vieqDF.append(ligboundDF)
    sns.stripplot(x="Degree of Binding", y="# Ligand Bound", hue="$K_d$ nM", data=vieqDF, ax=ax)
    ax.set(yscale="log", ylim=(0.1, 1e4), title="Valency of Binding to " + str(int(recCount)) + " Receptors", ylabel="Ligand Bound", xlabel="Receptors Bound")


def ratePlot(ax):
    "Plots rate of bivalent binding over dissocation rate for monovalently bound complexes"
    # kxstar * Ka, * val-1 * rec-1
    recScan = np.logspace(0, 4, 100)
    val = np.arange(1, 5)
    affinities = [1e8, 1e6]
    KxStarPl = 10 ** -10.0
    lines = ["-", ":"]
    colors = ["royalblue", "orange", "limegreen", "orangered"]
    rateHolder = np.zeros([100])
    for ii, Ka in enumerate(affinities):
        for jj, f in enumerate(val):
            for kk, recCount in enumerate(recScan):
                rateHolder[kk] = KxStarPl * Ka * (f - 1) * recCount
            ax.plot(recScan, rateHolder, color=colors[jj], label="Valency = " + str(f), linestyle=lines[ii])
    ax.set(xlim=(1, 100000), xlabel="Receptor Abundance", ylabel="Forward/Reverse Rate", xscale="log", ylim=(0.1, 5))  # ylim=(0, 1),
    handles, _ = ax.get_legend_handles_labels()
    handles = handles[0:4]
    line = Line2D([], [], color="black", marker="_", linestyle="None", markersize=6, label="$K_d$ nM = 10")
    point = Line2D([], [], color="black", marker=".", linestyle="None", markersize=6, label="$K_d$ nM = 1000")
    handles.append(line)
    handles.append(point)
    ax.legend(handles=handles, prop={"size": 6})
