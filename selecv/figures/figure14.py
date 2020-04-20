from .figureA import complex


def makeFigure():
    """ Heatmaps for complex compositions """
    return complex(1e-9, 10**-10, vmin=0, vmax=10)
