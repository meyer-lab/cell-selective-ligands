from .figureA import mixture

def makeFigure():
    """ Heatmaps for monomer mixtures """
    return mixture(1e-9, 10**-10, ff=5, vmin = 0, vmax = 10)
