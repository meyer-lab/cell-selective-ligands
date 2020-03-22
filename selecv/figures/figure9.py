"""

"""

from .figureCommon import getSetup


import numpy as np
import pandas as pd

def makeFigure(L0 = 1e-9, KxStar = 10**-12.2, ff = 4, offdiag = 1e5, comp = [[4, 0], [0, 4]]):
    nAffPts = 3
    ax, fig = getSetup((7, 6), (nAffPts, nAffPts))
    affRange = (5., 7.)
    affScan = np.logspace(affRange[0], affRange[1], nAffPts)
    abundRange = (1., 4.)

    for i1, aff1 in enumerate(affScan):
        for i2, aff2 in enumerate(np.flip(affScan)):
            abundHeatMap(ax[i2 * nAffPts + i1], abundRange, [[aff1, offdiag], [offdiag, aff2]],
                         L0 = L0, KxStar = KxStar, f = ff, comp = comp)

    return fig


def abundHeatMap(ax, abundRange, Kav, L0 = 1e-9, KxStar = 10**-12.2, f = 4, comp = [[4, 0], [0, 4]]):
    "Makes a heatmap comparing bindings of populations at multiple receptor abundances"
    nAbdPts = 70
    abundScan = np.logspace(abundRange[0], abundRange[1], nAbdPts)
    bindDF = pd.DataFrame(columns=abundScan, index=np.flip(abundScan))

    for i1, abund1 in enumerate(abundScan):
        sampMeans = np.zeros(nAbdPts)
        for i2, abund2 in enumerate(np.flip(abundScan)):
            #sampMeans[i2] = polyfc(L0, KxStar, f, [abund1, abund2], comp, Kav)
            sampMeans[i2] = polyc(L0/np.sum(comp)*4, KxStar, [abund1, abund2], comp, [0.5, 0.5], Kav)
        bindDF[bindDF.columns[i1]] = np.log(sampMeans)


    sns.heatmap(bindDF, ax=ax, xticklabels=nAbdPts - 1, yticklabels=nAbdPts - 1, vmin=1, vmax=6)
    ax.set(xlabel='Rec 1 abundance', ylabel='Rec 2 abundance')


