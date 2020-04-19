from .figureA import affinity

def makeFigure():
    """ Heatmaps for diagonal affnities """
    return affinity(1e-9, 10 ** -10, [0.5, 0.5], ff=4, vmin=0, vmax=10)
