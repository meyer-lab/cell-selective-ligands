"""
Figure 3. Exploration of Valency.
"""
import numpy as np
from .figureCommon import subplotLabel, getSetup
from ..imports import getPopDict
from ..sampling import sampleSpec

ligConc = np.array([10e-9])
KxStarP = 10e3
affinity = 10e-9


def makeFigure():
    """ Make figure 3. """
    # Get list of axis objects
    ax, f = getSetup((7, 6), (4, 3))
    subplotLabel(ax)

    valencyScan = np.logspace(0.0, 5.0, base=2.0, num=10)
    _, populationsdf = getPopDict()

    ValencyPlot(ax[0], populationsdf, valencyScan, ['Pop1', 'Pop3'])
    ValencyPlot(ax[1], populationsdf, valencyScan, ['Pop2', 'Pop3'])
    ValencyPlot(ax[2], populationsdf, valencyScan, ['Pop3', 'Pop4'])

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
    ax.set(xlabel='Valency', ylabel='Binding Ratio', title=popList[0] + ' to ' + popList[1] + ' binding ratio')
