"""
Figure 5.
"""
import numpy as np
from scipy.optimize import minimize
from .figureCommon import subplotLabel, getSetup
from ..imports import getPopDict
from ..sampling import sampleSpec


xNaught = [10e-9, 10e-9, 1.0, 0.5, 10e-9, 10e-9, 10e-9, 10e-9, ]
xBnds = ((10e-11, 10e-7), (10e-11, 10e3), (1, 32), (0, 1), (10e-11, 10e-7), (10e-11, 10e-7), (10e-11, 10e-7), (10e-11, 10e-7))


def makeFigure():
    """ Make figure 5. """
    # Get list of axis objects
    ax, f = getSetup((7, 6), (3, 4))

    subplotLabel(ax)

    return f


def minSelecFunc(x, recMeansM, CovsM):
    "Provides the function to be minimized to get optimal selectivity"
    return 1 / sampleSpec(x[0], x[1], x[2], recMeans, Covs, np.array([x[3], 1 - x[3]]), np.array([[x[4], x[5]], [x[6], x[7]]]))[1]


def optimizeDesign(df, popList):
    "Runs optimization and determines optimal parameters for selectivity of one population vs. another"
    recMeans, Covs = [], []
    for _, pop in enumerate(popList):
        dfPop = df[df['Population'] == pop]
        recMeans.append(np.array([dfPop['Receptor_1'].to_numpy(), dfPop['Receptor_2'].to_numpy()]).flatten())
        Covs.append(dfPop.Covariance_Matrix.to_numpy()[0])
    
    optimized = minimize(minSelecFunc, xNaught, bounds=xBnds, method='L-BFGS-B', args=(reMeans, Covs))
    print(optimized.x)
    print(minSelecFunc(optimized.x, recMeans, Covs)^-1)
    print(minSelecFunc(xNaught, recMeans, Covs)^-1)
    return optimized