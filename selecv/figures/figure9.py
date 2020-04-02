"""

"""

from .figureCommon import getSetup
from ..model import polyfc, polyc

import numpy as np
import pandas as pd


def makeFigure(L0, KxStar, Comp, ff = None, Cplx = None, offdiag = 1e5):
    nAffPts = 3
    ax, fig = getSetup((7, 6), (nAffPts, nAffPts))
    affRange = (6., 8.)
    affScan = np.logspace(affRange[0], affRange[1], nAffPts)
    abundRange = (1., 4.)
    if ff is not None:
        assert len(Comp) == 2, "Take two types of ligands"
    else:
        assert Cplx.shape[1] == 2, "Take two types of ligands"

    for i1, aff1 in enumerate(affScan):
        for i2, aff2 in enumerate(np.flip(affScan)):
            abundHeatMap(ax[i2 * nAffPts + i1], abundRange, L0, KxStar, \
                         [[aff1, offdiag], [offdiag, aff2]], Comp, f = ff, Cplx = Cplx)

    return fig

def abundHeatMap(ax, abundRange, L0, KxStar, Kav, Comp, f = None, Cplx = None):
    "Makes a heatmap comparing bindings of populations at multiple receptor abundances"
    assert bool(f is None) != bool(Cplx is None)
    nAbdPts = 70
    abundScan = np.logspace(abundRange[0], abundRange[1], nAbdPts)
    bindDF = pd.DataFrame(columns=abundScan, index=np.flip(abundScan))

    for i1, abund1 in enumerate(abundScan):
        sampMeans = np.zeros(nAbdPts)
        for i2, abund2 in enumerate(np.flip(abundScan)):
            if f is None:
                sampMeans[i2] = polyc(L0, KxStar, [abund1, abund2], Cplx, Comp, Kav)[0]
            else:
                sampMeans[i2] = polyfc(L0, KxStar, f, [abund1, abund2], Comp, Kav)[0]

        bindDF[bindDF.columns[i1]] = np.log(sampMeans)

    sns.heatmap(bindDF, ax=ax, xticklabels=nAbdPts - 1, yticklabels=nAbdPts - 1, vmin=0, vmax=6)
    ax.set(xlabel='Rec 1 abundance', ylabel='Rec 2 abundance')
