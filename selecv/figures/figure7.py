import numpy as np
from selecv.figures.figureCommon import subplotLabel, getSetup
from selecv.model import *
import seaborn as sns


L0 = 1e-9
KxStar = 1e-12
Rtot1 = [1e4, 1e4]
Rtot2 = [1e4, 1e2]
Cplx = np.array([[1, 1]])
Ctheta = [1.0]
Kav = np.array([[1e6, 5e4], [5e4, 1e6]])

x = np.logspace(-15, -9, num = 7)
y = []

for KxStar in x:
    Lbound1, _ = polyc(L0, KxStar, Rtot1, Cplx, Ctheta, Kav)
    Lbound2, _ = polyc(L0, KxStar, Rtot2, Cplx, Ctheta, Kav)
    ratio = Lbound1[0] / Lbound2[0]
    y.append(ratio)


def makeFigure():
    """ Make figure 7. """
    ax, f = getSetup((7, 6), (2, 3))
    subplotLabel(ax)

    ax[0] = sns.lineplot(x=x, y=y)
    ax[0].set(xscale="log", ylim=(0, 10))

    return f