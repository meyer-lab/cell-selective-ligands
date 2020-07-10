"""
Figure 5. Heterovalent and bispecific
"""

import numpy as np
from .figureCommon import getSetup, heatmap
from ..model import polyc


def makeFigure():
    ax, f = getSetup((9, 9), (3, 3))

    L0 = 1e-9
    KxStar = 1e-12
    Kav = [[1e7, 1e5], [1e5, 1e6]]

    for i, s in enumerate([[1, 1], [1, 0], [0, 1], [2, 0], [0, 2], [2, 1], [1, 2]]):
        heatmap(ax[i], L0, KxStar, Kav, [1.0], Cplx=[s], vrange=(0, 5), title="{} Lbound".format(s))

    f.suptitle("$K_x^* = ${}".format(KxStar))
    return f
