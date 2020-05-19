from .figureA import valency


def makeFigure():
    """ Heatmaps for various valency """
    return valency(1e-9, 10 ** -10, [1.0], Kav=[[3e6, 3e6]], vmin=0.0, vmax=9)
