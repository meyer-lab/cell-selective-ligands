"""
Figure S3. Exploration of Valency.
"""
import numpy as np
from .figureCommon import subplotLabel, setFontSize, getSetup, popCompare
from .figureS3 import vieqPlot

ligConc = np.array([1e-8])
KxStarP = 1e-10
affinity = 1e8  # 7


def makeFigure():
    # Get list of axis objects
    ax, f = getSetup((9, 6), (2, 3))
    subplotLabel(ax, [0] + list(range(3, 6)))

    ax[0].axis("off")
    ax[1].axis("off")
    ax[2].axis("off")

    valencyScan = np.linspace(1, 8, num=32)
    popCompare(ax[3], [r"$R_1^{hi}R_2^{lo}$", r"$R_1^{med}R_2^{lo}$"], "Valency", Kav=[1e6, 1e7, 1e8], L0=[1e-8], f=valencyScan)
    vieqPlot(ax[4], 1e4, 8)
    vieqPlot(ax[5], 1e3, 8)

    setFontSize(ax, 10)
    return f