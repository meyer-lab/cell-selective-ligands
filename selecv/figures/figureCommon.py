"""
Contains utilities and functions that are commonly used in the figure creation files.
"""
from string import ascii_lowercase
from matplotlib import gridspec, pyplot as plt
from scipy.stats import multivariate_normal
import seaborn as sns
import pandas as pds
import numpy as np


def getSetup(figsize, gridd):
    """ Establish figure set-up with subplots. """
    sns.set(style="whitegrid", font_scale=0.7, color_codes=True, palette="colorblind", rc={"grid.linestyle": "dotted", "axes.linewidth": 0.6})

    # Setup plotting space and grid
    f = plt.figure(figsize=figsize, constrained_layout=True)
    gs1 = gridspec.GridSpec(*gridd, figure=f)

    # Get list of axis objects
    ax = list()
    for x in range(gridd[0] * gridd[1]):
        ax.append(f.add_subplot(gs1[x]))

    return (ax, f)


def subplotLabel(axs):
    """ Place subplot labels on figure. """
    for ii, ax in enumerate(axs):
        ax.text(-0.2, 1.25, ascii_lowercase[ii], transform=ax.transAxes, fontsize=16, fontweight="bold", va="top")


def PlotCellPops(ax, df, bbox=False):
    "Plots theoretical populations"
    sampleData = sampleReceptors(df, 20000)
    sns.set_palette("husl", 8)
    for pop in sampleData.Population.unique():
        popDF = sampleData.loc[sampleData['Population'] == pop]
        sns.kdeplot(popDF.Receptor_1, popDF.Receptor_2, ax=ax, label=pop, shade=True, shade_lowest=False, legend=False)
    if bbox:
        ax.legend(fontsize=6, bbox_to_anchor=(1.02, 1), loc='center right')
    else:
        ax.legend(fontsize=7)
    ax.set(xscale="log", yscale="log")


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
