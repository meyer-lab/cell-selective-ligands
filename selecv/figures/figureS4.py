"""
Figure 7.
"""
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.optimize import minimize
from .figureCommon import subplotLabel, getSetup, heatmap, heatmapNorm
from ..imports import getPopDict
from ..sampling import sampleSpec
from .figure6 import genOnevsAll
from ..model import polyc


def makeFigure():
    """ Make figure S4. """
    # Get list of axis objects
    ax, f = getSetup((18, 6), (1, 3))
    subplotLabel(ax)
    affDLsub = np.array([0, 15])
    fDLsub = 1

    # gridSearchTry(populationsdf, ['Pop5', 'Pop3'])
    optimizeDesignDL(ax[0], [r"$R_1^{lo}R_2^{hi}$"], fDLsub, affDLsub)
    optimizeDesignDL(ax[1], [r"$R_1^{hi}R_2^{hi}$"], fDLsub, affDLsub)
    optimizeDesignDL(ax[2], [r"$R_1^{med}R_2^{med}$"], fDLsub, affDLsub)

    return f


_, df = getPopDict()


def minSelecFuncDL(x, tMeans, offTMeans, fDL, affDL):
    "Provides the function to be minimized to get optimal selectivity"
    offTargetBound = 0

    #print(polyc(np.exp(x[0]), np.exp(x[1]), [10**tMeans[0][0], 10**tMeans[0][1]], [[fDL, 0], [0, x[2]]], [0.5, 0.5], np.array([[affDL[0], affDL[1]], [np.exp(x[4]), np.exp(x[5])]])))
    targetBound = polyc(np.exp(x[0]), np.exp(x[1]), [10**tMeans[0][0], 10**tMeans[0][1]], [[fDL, 0], [0, x[2]]], [0.5, 0.5], np.array([[affDL[0], affDL[1]], [np.exp(x[4]), np.exp(x[5])]]))[0][1]

    for means in offTMeans:
        offTargetBound += polyc(np.exp(x[0]), np.exp(x[1]), [10**means[0], 10**means[1]], [[fDL, 0], [0, x[2]]], [0.5, 0.5], np.array([[affDL[0], affDL[1]], [np.exp(x[4]), np.exp(x[5])]]))[0][1]

    return (offTargetBound) / (targetBound)


def optimizeDesignDL(ax, targetPop, fDL, affDL):
    "Runs optimization and determines optimal parameters for selectivity of one population vs. another"
    targMeans, offTargMeans = genOnevsAll(targetPop)
    
    npoints = 3
    ticks = np.full([npoints], None)
    affScan = np.logspace(affDL[0], affDL[1], npoints)
    ticks[0], ticks[-1] = "1e" + str(affDL[0]), "1e" + str(affDL[1])
    bounds = ((np.log(1e-9), np.log(1e-9)), (np.log(1e-15), np.log(1e-9)), (1, 16), (0, 1), (np.log(1e2), np.log(1e8)), (np.log(1e2), np.log(1e8)))
    xnot = np.array([np.log(1e-9), np.log(1e-9), 1, 1, np.log(10e8), np.log(10e6)])

    sampMeans = np.zeros(npoints)
    ratioDF = pd.DataFrame(columns=affScan, index=affScan)

    for ii, aff1 in enumerate(affScan):
        for jj, aff2 in enumerate(np.flip(affScan)):
            optimized = minimize(minSelecFuncDL, xnot, bounds=np.array(bounds), method="L-BFGS-B", args=(targMeans, offTargMeans, fDL, affDL),options={"eps": 1, "disp": True})
            sampMeans[jj] = 7/optimized.fun
        
        ratioDF[ratioDF.columns[ii]] = sampMeans

    print(ratioDF)
    print(ratioDF.iloc[0, 0])
    ratioDF = ratioDF.divide(ratioDF.iloc[0, 0])
    Cbar=True

    if ratioDF.max().max() < 15:
        sns.heatmap(ratioDF, ax=ax, xticklabels=ticks, yticklabels=np.flip(ticks), vmin=0, vmax=10, cbar=Cbar, cbar_kws={'label': 'Binding Ratio'}, annot=True)
    else:
        sns.heatmap(ratioDF, ax=ax, xticklabels=ticks, yticklabels=np.flip(ticks), cbar=Cbar, cbar_kws={'label': 'Binding Ratio'}, annot=True)
    ax.set(xlabel="Dead Ligand Rec 1 Affinity ($K_a$, in M$^{-1}$)", ylabel="Dead Ligand Rec 2 Affinity ($K_a$, in M$^{-1}$)")
    ax.set_title(targetPop, fontsize=8)
