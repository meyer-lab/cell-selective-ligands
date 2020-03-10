"""
Figure 1. Introduce the model system.
"""

import seaborn as sns
import pandas as pds
import numpy as np
from scipy.stats import multivariate_normal
from .figureCommon import subplotLabel, getSetup
from ..imports import import_Rexpr, getPopDict


def makeFigure():
    """ Make figure 1. """
    # Get list of axis objects
    ax, f = getSetup((7, 6), (2, 2))

    subplotLabel(ax)

    _, npdata, cell_names = import_Rexpr()
    _, populations = getPopDict()
    plotCellpops(ax[0:2], npdata, cell_names, populations)

    return f


def plotCellpops(ax, data, names, df):
    "Plot both theoretical and real receptor abundances"
    for ii, cell in enumerate(names):
        ax[0].scatter(data[ii, 0], data[ii, 1], label=cell)
    ax[0].set(ylabel='IL2Ra Abundance', xlabel='IL2Rb/gc Abundance', xscale='log', yscale='log')
    ax[0].legend()

    sampleData = sampleReceptors(df, 1000)

    sns.scatterplot(x='Receptor_1', y='Receptor_2', hue='Population', data=sampleData, ax=ax[1])
    ax[1].set(xscale="log", yscale="log")


def sampleReceptors(df, nsample=100):
    """
    Generate samples in each sample space
    """
    Populations = df.Population.unique()
    sampledf = pds.DataFrame(columns=['Population', 'Receptor_1', 'Receptor_2'])
    for population in Populations:
        populationdf = df[df['Population'] == population]
        RtotMeans = np.array([populationdf.Receptor_1.to_numpy(), populationdf.Receptor_2.to_numpy()]).flatten()
        RtotCovs = populationdf.Covariance_Matrix.to_numpy()[0]
        pop = np.power(10.0, multivariate_normal.rvs(mean=RtotMeans, cov=RtotCovs, size=nsample))
        popdf = pds.DataFrame({'Population': population, 'Receptor_1': pop[:, 0], 'Receptor_2': pop[:, 1]})
        sampledf = sampledf.append(popdf)

    return sampledf
