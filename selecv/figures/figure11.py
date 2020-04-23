from .figureA import affinity


def makeFigure():
    """ Heatmaps for diagonal affnities """
    return affinity(1e-9, 10 ** -10, [1.0], ff=1, vmin=-1, vmax=5.5)
