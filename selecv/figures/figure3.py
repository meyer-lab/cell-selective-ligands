"""
Figure 3. Exploration of Valency.
"""
import numpy as np
from .figureCommon import subplotLabel, getSetup
from ..imports import getPopDict
from ..sampling import sampleSpec
from ..model import polyfc

ligConc = np.array([10e-9])
KxStarP = 10e-11
affinity = 10e7


def makeFigure():
    """ Make figure 3. """
    # Get list of axis objects
    ax, f = getSetup((7, 6), (4, 3))
    subplotLabel(ax)

    valencyScan = np.logspace(0.0, 5.0, base=2.0, num=10)
    _, populationsdf = getPopDict()

    ValencyPlot(ax[0], populationsdf, valencyScan, ['Pop3', 'Pop1'])
    ValencyPlot(ax[1], populationsdf, valencyScan, ['Pop3', 'Pop2'])
    ValencyPlot(ax[2], populationsdf, valencyScan, ['Pop7', 'Pop5'])
    ValencyPlot(ax[3], populationsdf, valencyScan, ['Pop5', 'Pop3'])
    ax[3].set_ylim(0, 2)
    ValencyPlot(ax[4], populationsdf, valencyScan, ['Pop5', 'Pop6'])
    ax[4].set_ylim(0, 2)
    ValencyPlot(ax[5], populationsdf, valencyScan, ['Pop3', 'Pop4'])
    ax[5].set_ylim(0, 2)
    valDemo(ax[6])

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
    nPoints = 100
    recScan = np.logspace(0, 4, nPoints)
    labels = ['Monovalent', 'Bivalent', 'Trivalent', 'Tetravalent']
    percHold = np.zeros(nPoints)

    for ii, valencyLab in enumerate(labels):
        for jj, recCount in enumerate(recScan):
            percHold[jj] = polyfc(ligConc, KxStarP, ii + 1, recCount, [1], np.array([[affinity]])) / recCount

            #print(polyfc(ligConc, KxStarP, ii + 1, recCount, [1], np.array([[affinity]])))
            # print(recCount)
            #print(ligConc, KxStarP, ii + 1, recCount, [1], np.array([[affinity]]))

        ax.plot(recScan, percHold, label=valencyLab)
    ax.set(xlim=(1, 1000), ylim=(0, 1), xlabel='Receptor Abundance', ylabel='Fraction Receptors Bound', xscale='log')
    ax.legend()
