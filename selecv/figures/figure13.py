from .figureA import mixture


def makeFigure():
    """ Heatmaps for monomer mixtures """
    return mixture(1e-9, 10 ** -10, ff=1, vmin=-2, vmax=3.5)
