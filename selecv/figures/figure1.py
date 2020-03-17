"""
Figure 1. Introduce the model system.
"""

import seaborn as sns
import pandas as pds
import numpy as np
from scipy.stats import multivariate_normal
from .figureCommon import subplotLabel, getSetup
from ..imports import import_Rexpr, getPopDict
from ..sampling import sampleSpec

ligConc = np.array([10e-9])
KxStarP = 10e2
val = 16.0


def makeFigure():
    """ Make figure 1. """
    # Get list of axis objects
    ax, f = getSetup((7, 6), (2, 2))

    subplotLabel(ax)

    _, npdata, cell_names = import_Rexpr()
    _, populations = getPopDict()
    plotCellpops(ax[0:2], npdata, cell_names, populations)
    plotSampleConc(ax[2], populations, [-12., 4., ], ['Pop2', 'Pop3'])

    return f


def plotCellpops(ax, data, names, df):
    "Plot both theoretical and real receptor abundances"
    for ii, cell in enumerate(names):
        ax[0].scatter(data[ii, 0], data[ii, 1], label=cell)
    ax[0].set(ylabel='IL2Rα Abundance', xlabel='IL-2Rβ Abundance', xscale='log', yscale='log')
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


def plotSampleConc(ax, df, concRange, popList):
    "Makes a line chart comparing binding ratios of populations at multiple concentrations"
    npoints = 100
    concScan = np.logspace(concRange[0], concRange[1], npoints)
    df1 = df[df['Population'] == popList[0]]
    df2 = df[df['Population'] == popList[1]]
    recMean1 = np.array([df1['Receptor_1'].to_numpy(), df1['Receptor_2'].to_numpy()]).flatten()
    recMean2 = np.array([df2['Receptor_1'].to_numpy(), df2['Receptor_2'].to_numpy()]).flatten()
    Cov1 = df1.Covariance_Matrix.to_numpy()[0]
    Cov2 = df2.Covariance_Matrix.to_numpy()[0]
    sampMeans, sampDevs = np.zeros([npoints]), np.zeros([npoints])

    for ii, conc in enumerate(concScan):
        sampMeans[ii], sampDevs[ii] = sampleSpec(conc, KxStarP, val, [recMean1, recMean2], [Cov1, Cov2], np.array([1]), np.array([[10e-9, 10e-9]]), nsample=100)

    sampMeans *= np.sum(np.power(10, recMean2)) / np.sum(np.power(10, recMean1))
    sampDevs *= np.sum(np.power(10, recMean2)) / np.sum(np.power(10, recMean1))
    underDev, overDev = sampMeans - sampDevs, sampMeans + sampDevs
    ax.plot(concScan, sampMeans, color='royalblue')
    ax.fill_between(concScan, underDev, overDev, color='royalblue', alpha=.1)
    ax.set(xscale='log', xlim=(np.power(10, concRange[0]), np.power(10, concRange[1])))
