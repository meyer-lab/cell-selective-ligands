from .figureA import affinity
from .figureCommon import subplotLabel, getSetup, popCompare

def makeFigure():
    """ Heatmaps for diagonal affnities """
    ax, f = getSetup((10, 12), (3, 3))
    print("Ok")
    affinity(f, ax[0:9], 1e-9, 10 ** -10, [1.0], ff=1, vmin=-1, vmax=5.5)
