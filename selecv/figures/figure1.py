"""
Figure 1. Introduce the model system.
"""
from .figureCommon import subplotLabel, getSetup
from ..imports import import_Rexpr, getPopDict
import pandas as pds
import seaborn as sns


def makeFigure():
    """ Make figure 1. """
    # Get list of axis objects
    ax, f = getSetup((7, 6), (2, 2))
    data, npdata, cell_names = import_Rexpr()
    _, populations = getPopDict()
    plotCellpops(ax[0:2], npdata, cell_names, populations)

    return f


def plotCellpops(ax, data, names, df):
    for ii, cell in enumerate(names):
        ax[0].scatter(data[ii, 0], data[ii, 1], label=cell)
    ax[0].set(ylabel='IL2Ra Abundance', xlabel='IL2Rb/gc Abundance', xscale='log', yscale='log')
    ax[0].legend()

    df = df.reset_index()
    sns.scatterplot(x='Receptor_1', y='Receptor_2', hue='Population', data=df, ax=ax[1])
    ax[1].set(xscale="log", yscale="log")
