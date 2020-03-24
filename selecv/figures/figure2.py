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
    affHeatMap(ax[0], populationsdf, [5, 9], ['Pop3', 'Pop2'])
    affHeatMap(ax[1], populationsdf, [5, 9], ['Pop5', 'Pop3'])
    affHeatMap(ax[2], populationsdf, [5, 9], ['Pop6', 'Pop3'])
    affHeatMap(ax[3], populationsdf, [5, 9], ['Pop7', 'Pop4'])
    affHeatMap(ax[4], populationsdf, [5, 9], ['Pop5', 'Pop6'])
    affHeatMap(ax[5], populationsdf, [5, 9], ['Pop7', 'Pop4', 'Pop3'])

    subplotLabel(ax)

    return f


def affHeatMap(ax, df, affRange, popList):
    "Makes a heatmap comparing binding ratios of populations at a range of binding affinities"
    npoints = 15
    ticks = np.full([npoints], None)
    affScan = np.logspace(affRange[0], affRange[1], npoints)
    ticks[0], ticks[-1] = '10e' + str(affRange[0]), '10e' + str(affRange[1])
    recMeans, Covs = [], []
    Title = popList[0] + ' to ' + popList[1]
    for ii, pop in enumerate(popList):
        dfPop = df[df['Population'] == pop]
        recMeans.append(np.array([dfPop['Receptor_1'].to_numpy(), dfPop['Receptor_2'].to_numpy()]).flatten())
        Covs.append(dfPop.Covariance_Matrix.to_numpy()[0])
        if ii >= 2:
            Title += '/' + pop
    sampMeans = np.zeros(npoints)
    ratioDF = pds.DataFrame(columns=affScan, index=affScan)

    for ii, aff1 in enumerate(affScan):
        for jj, aff2 in enumerate(affScan):
            _, sampMeans[jj], _ = sampleSpec(ligConc, KxStarP, val, recMeans, Covs, np.array([1]), np.array([[aff1, aff2]]))
        ratioDF[ratioDF.columns[ii]] = sampMeans

    sns.heatmap(ratioDF, ax=ax, xticklabels=ticks, yticklabels=ticks, vmin=0, vmax=10)
    ax.set(title=Title + ' binding ratio', xlabel='Rec 1 Affinity', ylabel='Rec 2 Affinity', fontsize=8 - 0.4 * len(popList))
