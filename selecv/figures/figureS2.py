"""
Figure S2. Exploration of Valency.
"""
import numpy as np
import pandas as pds
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from .figureCommon import subplotLabel, setFontSize, getSetup, popCompare, heatmap
from valentbind import polyfc

ligConc = np.array([1e-8])
KxStarP = 1e-10
affinity = 1e8  # 7


def makeFigure():
    # Get list of axis objects
    ax, f = getSetup((9, 6), (2, 3))
    subplotLabel(ax, [0] + list(range(3, 6)))

    valency(f, ax[0:3], 1e-9, 10 ** -10, [1.0], Kav=[[1e6, 0.01]], vmin=0.0, vmax=12)
    valencyScan = np.linspace(1, 8, num=32)
    popCompare(ax[3], [r"$R_1^{hi}R_2^{hi}$", r"$R_1^{med}R_2^{med}$"], "Valency", Kav=[1e6, 1e7, 1e8], L0=[1e-8], f=valencyScan)
    popCompare(ax[4], [r"$R_1^{hi}R_2^{med}$", r"$R_1^{med}R_2^{med}$"], "Valency", Kav=[1e6, 1e7, 1e8], L0=[1e-8], f=valencyScan)
    ratePlot(ax[5], fsize=9)

    setFontSize(ax, 10, xsci=[0, 1, 2, 5], ysci=[0, 1, 2, 5], nolegend=[5])
    return f


def valency(fig, axs, L0, KxStar, Comp, Kav=[[1e6, 1e5], [1e5, 1e6]], Cplx=None, vmin=-2, vmax=4):
    ffs = [1, 4, 16]

    for i, v in enumerate(ffs):
        cbar = False
        if i in [2]:
            cbar = True
        heatmap(axs[i], L0, KxStar, Kav, Comp, f=v, Cplx=Cplx, vrange=(vmin, vmax), cbar=cbar, layover=1)
        axs[i].set(xlabel="Receptor 1 Abundance [#/cell]", ylabel='Receptor 2 Abundance [#/cell]')
        #plt.plot([4.3, 5.1], [2, 2], color="black", marker=2)
        #plt.text(5.0, 2.2, "b", size='large', color='black', weight='semibold', horizontalalignment='center', verticalalignment='center')
        #plt.plot([4.3, 5.3], [4.4, 5.4], color="black", marker=2)
        #plt.text(5.0, 5.6, "c", size='large', color='black', weight='semibold', horizontalalignment='center', verticalalignment='center')
        #plt.plot([4.5, 5.0], [4.0, 4.0], color="black", marker=1, markersize=4)
        #plt.text(5.0, 3.6, "d", size='large', color='black', weight='semibold', horizontalalignment='center', verticalalignment='center')
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
    ax.set(yscale="log", ylim=(0.1, 1e4), title="Binding Degrees Distribution, " + str(int(recCount)) + " receptors", ylabel="Ligand Bound", xlabel="Degree of Binding")


def ratePlot(ax, fsize=6):
    "Plots rate of bivalent binding over dissocation rate for monovalently bound complexes"
    # kxstar * Ka, * val-1 * rec-1
    recScan = np.logspace(0, 6, 100)
    val = np.arange(2, 5)
    affinities = [1e8, 1e6]
    KxStarPl = 10 ** -10.0
    lines = ["-", ":"]
    colors = ["orange", "limegreen", "orangered"]
    rateHolder = np.zeros([100])
    for ii, Ka in enumerate(affinities):
        for jj, f in enumerate(val):
            for kk, recCount in enumerate(recScan):
                rateHolder[kk] = KxStarPl * Ka * (f - 1) * recCount
            ax.plot(recScan, rateHolder, color=colors[jj], label="Valency = " + str(f), linestyle=lines[ii])
    ax.set(xlim=(1, 1000000), xlabel="Receptor Abundance", ylabel="Forward/Reverse Rate", xscale="log", ylim=(0.1, 5))  # ylim=(0, 1),
    handles, _ = ax.get_legend_handles_labels()
    handles = handles[0:3]
    line = Line2D([], [], color="black", marker="_", linestyle="None", markersize=6, label="$K_d$ = 10 nM")
    point = Line2D([], [], color="black", marker=".", linestyle="None", markersize=6, label="$K_d$ = 1000 nM")
    handles.append(line)
    handles.append(point)
    ax.legend(handles=handles, prop={"size": fsize})
    ax.set_yscale("log")
    ax.set_ylim(1e-1, 1e3)
    ax.set_title("Reaction rate for second degree binding")
