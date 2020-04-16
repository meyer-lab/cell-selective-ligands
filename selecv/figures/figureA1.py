from .figureA import *

def makeFigure():
    return affinity(1e-9, 10 ** -10, [0.5, 0.5], ff=4, vmin=0, vmax=10)