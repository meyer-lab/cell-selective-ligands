from .figureA import valency

def makeFigure():
    """ Heatmaps for various valency """
    return valency(1e-9, 10 ** -10, [0.5, 0.5], vmin=-1, vmax=10)
