"""
Figure 5.
"""
import numpy as np
from scipy.optimize import minimize
from .figureCommon import subplotLabel, getSetup
from ..imports import getPopDict
from ..sampling import sampleSpec

ligConc = 10e-9
KxStar = 10e3
xNaught = [1.0, 0.5, 10e-9, 10e-9, 10e-9, 10e-9] #Conc, KxStar, Valency, Mix1, aff11, aff12, aff21, aff22
xBnds = ((1, 32), (0, 1), (10e-11, 10e-7), (10e-11, 10e-7), (10e-11, 10e-7), (10e-11, 10e-7))


def makeFigure():
    """ Make figure 5. """
    # Get list of axis objects
    ax, f = getSetup((7, 6), (2, 3))
    subplotLabel(ax)

    _, populationsdf = getPopDict()
    optimizeDesign(populationsdf, ['Pop3', 'Pop4'])

    return f


def minSelecFunc(x, recMeansM, CovsM):
    "Provides the function to be minimized to get optimal selectivity"
    return 1 / sampleSpec(ligConc, KxStar, x[0], recMeansM, CovsM, np.array([x[1], 1 - x[1]]), np.array([[x[2], x[3]], [x[4], x[5]]]))[1]


def optimizeDesign(df, popList):
    "Runs optimization and determines optimal parameters for selectivity of one population vs. another"
    recMeans, Covs = [], []
    for _, pop in enumerate(popList):
        dfPop = df[df['Population'] == pop]
        recMeans.append(np.array([dfPop['Receptor_1'].to_numpy(), dfPop['Receptor_2'].to_numpy()]).flatten())
        Covs.append(dfPop.Covariance_Matrix.to_numpy()[0])
    
    optimized = minimize(minSelecFunc, xNaught, bounds=xBnds, method='L-BFGS-B', args=(recMeans, Covs), options={'disp': True})
    print(optimized.x)
    print(minSelecFunc(optimized.x, recMeans, Covs))
    print(minSelecFunc(xNaught, recMeans, Covs))
    print("Valency, Mix1, aff11, aff12, aff21, aff22")
    return optimized