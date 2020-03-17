"""
Figure 4. Mixtures for enhanced targeting.
"""

import numpy as np
from .figureCommon import subplotLabel, getSetup
from ..imports import getPopDict
from ..sampling import sampleSpec

ligConc = np.array([10e-9])
KxStarP = 1.0
val = 1.0


def makeFigure():
    """ Make figure 4. """
    # Get list of axis objects
    ax, f = getSetup((7, 6), (3, 4))

    subplotLabel(ax)

    affinities = np.array([[10e-7, 10e-9], [10e-9, 10e-7]])
    _, populationsdf = getPopDict()
    MixPlot(ax[0], populationsdf, ['Pop2', 'Pop3'], affinities, 101)
    MixPlot(ax[1], populationsdf, ['Pop3', 'Pop4'], affinities, 101)
    MixPlot(ax[2], populationsdf, ['Pop5', 'Pop6'], affinities, 101)

    return f


def MixPlot(ax, df, popList, affinities, npoints):
    "Makes a line chart comparing binding ratios of populations at multiple mixture compositions"
    df1 = df[df['Population'] == popList[0]]
    df2 = df[df['Population'] == popList[1]]
    recMean1 = np.array([df1['Receptor_1'].to_numpy(), df1['Receptor_2'].to_numpy()]).flatten()
    recMean2 = np.array([df2['Receptor_1'].to_numpy(), df2['Receptor_2'].to_numpy()]).flatten()
    Cov1 = df1.Covariance_Matrix.to_numpy()[0]
    Cov2 = df2.Covariance_Matrix.to_numpy()[0]
    sampMeans, sampDevs = np.zeros([npoints]), np.zeros([npoints])
    mixRatio = np.linspace(0, 1, npoints)

    for ii, mixture1 in enumerate(mixRatio):
        sampMeans[ii], sampDevs[ii] = sampleSpec(ligConc, KxStarP, val, [recMean1, recMean2], [Cov1, Cov2], np.array([mixture1, 1 - mixture1]), np.array([affinities[0], affinities[1]]))

    sampMeans *= np.sum(np.power(10, recMean2)) / np.sum(np.power(10, recMean1))
    sampDevs *= np.sum(np.power(10, recMean2)) / np.sum(np.power(10, recMean1))
    underDev, overDev = sampMeans - sampDevs, sampMeans + sampDevs
    ax.plot(mixRatio, sampMeans, color='royalblue')
    ax.fill_between(mixRatio, underDev, overDev, color='royalblue', alpha=.1)
    ax.set(xlabel='Ligand 1 in Mixture', ylabel='Binding Ratio', title=popList[0] + ' to ' + popList[1] + ' binding ratio')
