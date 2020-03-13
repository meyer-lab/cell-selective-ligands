"""
Figure 2. Explore selectivity vs. affinity.
"""

import numpy as np
import seaborn as sns
import pandas as pds
from .figureCommon import subplotLabel, getSetup
from ..imports import getPopDict
from ..sampling import sampleSpec

ligConc = np.array([10e-9])
KxStarP = 10e-9
val = 1.0


def makeFigure():
    """ Make figure 2. """
    # Get list of axis objects
    ax, f = getSetup((7, 6), (4, 3))
    _, populationsdf = getPopDict()
    affHeatMap(ax[0], populationsdf, [-9, -7], ['Pop2', 'Pop3'])
    affHeatMap(ax[1], populationsdf, [-9, -7], ['Pop3', 'Pop5'])
    affHeatMap(ax[2], populationsdf, [-9, -7], ['Pop3', 'Pop4'])

    subplotLabel(ax)

    return f


def affHeatMap(ax, df, affRange, popList):
    npoints = 10
    affScan = np.logspace(affRange[0], affRange[1], npoints)
    df1 = df[df['Population'] == popList[0]]
    df2 = df[df['Population'] == popList[1]]
    recMean1 = np.array([df1['Receptor_1'].to_numpy(), df1['Receptor_2'].to_numpy()]).flatten()
    recMean2 = np.array([df2['Receptor_1'].to_numpy(), df2['Receptor_2'].to_numpy()]).flatten()
    Cov1 = df1.Covariance_Matrix.to_numpy()[0]
    Cov2 = df2.Covariance_Matrix.to_numpy()[0]
    sampMeans = np.zeros([npoints, 1])
    ratioDF = pds.DataFrame(columns=affScan, index=affScan)
    for ii, aff1 in enumerate(affScan):
        for jj, aff2 in enumerate(affScan):
            sampMeans[jj, 0], _ = sampleSpec(ligConc, KxStarP, val, [recMean1, recMean2], [Cov1, Cov2], np.array([1]), np.array([[aff1, aff2]]), nsample=100)
        ratioDF[ratioDF.columns[ii]] = sampMeans

    sns.heatmap(ratioDF, ax=ax, xticklabels=False, yticklabels=False)
    ax.set(title=popList[0] + ' to ' + popList[1] + ' binding ratio', xlabel='Rec 1 Affinity', ylabel='Rec 2 Affinity')
