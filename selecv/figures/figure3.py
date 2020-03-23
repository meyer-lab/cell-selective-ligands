"""
Figure 3. Exploration of Valency.
"""
import numpy as np
from matplotlib.lines import Line2D
from .figureCommon import subplotLabel, getSetup
from ..imports import getPopDict
from ..sampling import sampleSpec
from ..model import polyfc

ligConc = np.array([10e-9])
KxStarP = 10e-11
affinity = 10e7  # 7


def makeFigure():
    """ Make figure 3. """
    # Get list of axis objects
    ax, f = getSetup((7, 6), (3, 3))
    subplotLabel(ax)

    valencyScan = np.logspace(0.0, 5.0, base=2.0, num=10)
    _, populationsdf = getPopDict()

    valDemo(ax[0])
    ConcValPlot(ax[1])
    KxValPlot(ax[2])
    ValencyPlot(ax[3], populationsdf, valencyScan, ['Pop3', 'Pop1'])
    ValencyPlot(ax[4], populationsdf, valencyScan, ['Pop3', 'Pop2'])
    ValencyPlot(ax[5], populationsdf, valencyScan, ['Pop7', 'Pop5'])
    ValencyPlot(ax[6], populationsdf, valencyScan, ['Pop5', 'Pop3'])
    ax[6].set_ylim(0, 2)
    ValencyPlot(ax[7], populationsdf, valencyScan, ['Pop5', 'Pop6'])
    ax[7].set_ylim(0, 2)
    ValencyPlot(ax[8], populationsdf, valencyScan, ['Pop3', 'Pop4'])
    ax[8].set_ylim(0, 2)

    return f


def ValencyPlot(ax, df, valencies, popList):
    "Makes a line chart comparing binding ratios of populations at multiple valencies"
    df1 = df[df['Population'] == popList[0]]
    df2 = df[df['Population'] == popList[1]]
    recMean1 = np.array([df1['Receptor_1'].to_numpy(), df1['Receptor_2'].to_numpy()]).flatten()
    recMean2 = np.array([df2['Receptor_1'].to_numpy(), df2['Receptor_2'].to_numpy()]).flatten()
    Cov1 = df1.Covariance_Matrix.to_numpy()[0]
    Cov2 = df2.Covariance_Matrix.to_numpy()[0]
    sampMeans, underDev, overDev = np.zeros_like(valencies), np.zeros_like(valencies), np.zeros_like(valencies)

    for ii, val in enumerate(valencies):
        underDev[ii], sampMeans[ii], overDev[ii] = sampleSpec(ligConc, KxStarP, val, [recMean1, recMean2], [Cov1, Cov2], np.array([1]), np.array([[affinity, affinity]]))

    ax.plot(valencies, sampMeans, color='royalblue')
    ax.fill_between(valencies, underDev, overDev, color='royalblue', alpha=.1)
    ax.set(xlabel='Valency', ylabel='Binding Ratio', title=popList[0] + ' to ' + popList[1] + ' binding ratio', xlim=(1, 32), ylim=(0, 100))


def valDemo(ax):
    "Demonstrate effect of valency"
    affs = [10e7, 10e6]
    colors = ['royalblue', 'orange', 'limegreen', 'orangered']
    lines = ['-', ':']
    nPoints = 100
    recScan = np.logspace(0, 4, nPoints)
    labels = ['Monovalent', 'Bivalent', 'Trivalent', 'Tetravalent']
    percHold = np.zeros(nPoints)
    for ii, aff in enumerate(affs):
        for jj, valencyLab in enumerate(labels):
            for kk, recCount in enumerate(recScan):
                percHold[kk] = polyfc(ligConc, KxStarP, jj + 1, recCount, [1], np.array([[aff]])) / recCount
            ax.plot(recScan, percHold, label=valencyLab, linestyle=lines[ii], color=colors[jj])

    ax.set(xlim=(1, 10000), ylim=(0, 1), xlabel='Receptor Abundance', ylabel='Fraction Receptors Bound', xscale='log')
    handles, _ = ax.get_legend_handles_labels()
    handles = handles[0:4]
    line = Line2D([], [], color='black', marker='_', linestyle='None', markersize=6, label='High Affinity')
    point = Line2D([], [], color='black', marker='.', linestyle='None', markersize=6, label='Low Affinity')
    handles.append(line)
    handles.append(point)
    ax.legend(handles=handles, prop={'size': 6})


def ConcValPlot(ax):
    "Keep valency constant and high - vary concentration"
    concScan = np.logspace(-11, -7, 5)
    valency = 4
    recScan = np.logspace(0, 4, 100)
    percHold = np.zeros(100)

    for conc in concScan:
        for jj, recCount in enumerate(recScan):
            percHold[jj] = polyfc(conc, KxStarP, valency, recCount, [1], np.array([[affinity]])) / recCount
        ax.plot(recScan, percHold, label=str(conc * 10e8) + ' nM')

    ax.set(xlim=(1, 10000), ylim=(0, 1), xlabel='Receptor Abundance', ylabel='Fraction Receptors Bound', xscale='log')
    ax.legend(prop={'size': 6})


def KxValPlot(ax):
    "Keep valency constant and high - vary concentration"
    concs = [ligConc, 10e-11]
    KxScan = np.logspace(-13, -9, 5)
    valency = 4
    recScan = np.logspace(0, 4, 100)
    percHold = np.zeros(100)
    colors = ['royalblue', 'orange', 'limegreen', 'orangered', 'blueviolet']
    lines = ['-', ':']

    for ii, conc in enumerate(concs):
        for jj, Kx in enumerate(KxScan):
            for kk, recCount in enumerate(recScan):
                percHold[kk] = polyfc(conc, Kx, valency, recCount, [1], np.array([[affinity]])) / recCount
            ax.plot(recScan, percHold, label='Kx* = ' + str(Kx), linestyle=lines[ii], color=colors[jj])

    ax.set(xlim=(1, 10000), ylim=(0, 1), xlabel='Receptor Abundance', ylabel='Fraction Receptors Bound', xscale='log')
    handles, _ = ax.get_legend_handles_labels()
    handles = handles[0:5]
    line = Line2D([], [], color='black', marker='_', linestyle='None', markersize=6, label='10 nM')
    point = Line2D([], [], color='black', marker='.', linestyle='None', markersize=6, label='0.1 nM')
    handles.append(line)
    handles.append(point)
    ax.legend(handles=handles, prop={'size': 6})
