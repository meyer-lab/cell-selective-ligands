"""
Figure S2. Old functions for optimization
"""

import numpy as np
from scipy.optimize import minimize
from .figureCommon import subplotLabel, getSetup
from ..imports import getPopDict
from ..sampling import sampleSpec
from ..model import polyfc


def makeFigure():
    """ Make figure 5. """
    # Get list of axis objects
    ax, f = getSetup((10, 8), (2, 3))
    subplotLabel(ax)

    _, populationsdf = getPopDict()
    # gridSearchTry(populationsdf, ['Pop5', 'Pop3'])
    optimizeDesign(populationsdf, [r"$R_1^{hi}R_2^{lo}$", r"$R_1^{med}R_2^{lo}$"])
    optimizeDesign(populationsdf, [r"$R_1^{lo}R_2^{hi}$", r"$R_1^{hi}R_2^{lo}$"])
    optimizeDesign(populationsdf, [r"$R_1^{hi}R_2^{hi}$", r"$R_1^{med}R_2^{med}$"])

    return f


def minSelecFunc(x, recMeansM):
    "Provides the function to be minimized to get optimal selectivity"

    return polyfc(np.exp(x[0]), np.exp(x[1]), x[2], [10**recMeansM[1][0], 10**recMeansM[1][1]], [x[3], 1 - x[3]], np.array([[np.exp(x[4]), np.exp(x[5])], [np.exp(x[5]), np.exp(x[4])]]))[0] / \
        polyfc(np.exp(x[0]), np.exp(x[1]), x[2], [10**recMeansM[0][0], 10**recMeansM[0][1]], [x[3], 1 - x[3]], np.array([[np.exp(x[4]), np.exp(x[5])], [np.exp(x[5]), np.exp(x[4])]]))[0]


def optimizeDesign(df, popList):
    "Runs optimization and determines optimal parameters for selectivity of one population vs. another"
    recMeans, Covs = [], []
    for _, pop in enumerate(popList):
        dfPop = df[df["Population"] == pop]
        recMeans.append(np.array([dfPop["Receptor_1"].to_numpy(), dfPop["Receptor_2"].to_numpy()]).flatten())
        Covs.append(dfPop.Covariance_Matrix.to_numpy()[0])

    xnot = np.array([np.log(1e-9), np.log(1e-12), 1, 1, np.log(1e8), np.log(1e1)])
    xBnds = ((np.log(1e-15), np.log(1e-6)), (np.log(1e-15), np.log(1e-9)), (1, 16), (0, 1), (np.log(1e4), np.log(1e9)), (np.log(1e0), np.log(1e2)))
    optimized = minimize(minSelecFunc, xnot, bounds=xBnds, args=(recMeans))
    assert optimized.success
    optimizers = optimized.x
    optimizers = [np.exp(optimizers[0]), np.exp(optimizers[1]), optimizers[2], optimizers[3], np.exp(optimizers[4]), np.exp(optimizers[5])]

    return optimizers


searchdic = {
    "L0": np.logspace(-11, -8, 4),
    "Kx": np.logspace(-12, -8, 5),
    "Val": np.logspace(0.0, 4.0, base=2.0, num=5),
    "Mix": np.linspace(0, 0.5, 2),
    "Aff": np.logspace(5, 9, 2),
}


def gridSearchTry(df, popList):
    """Grid search for best params for selectivity. Works but slowly. Probably won't use."""
    recMeans, Covs = [], []
    for ii, pop in enumerate(popList):
        dfPop = df[df["Population"] == pop]
        recMeans.append(np.array([dfPop["Receptor_1"].to_numpy(), dfPop["Receptor_2"].to_numpy()]).flatten())
        Covs.append(dfPop.Covariance_Matrix.to_numpy()[0])

    resultsTensor = np.zeros([4, 5, 5, 3, 5, 5, 5, 5])
    for ii, conc in enumerate(searchdic["L0"]):
        for jj, kx in enumerate(searchdic["Kx"]):
            for kk, val in enumerate(searchdic["Val"]):
                for ll, mix in enumerate(searchdic["Mix"]):
                    for mm, aff1 in enumerate(searchdic["Aff"]):
                        for nn, aff2 in enumerate(searchdic["Aff"]):
                            for oo, aff3 in enumerate(searchdic["Aff"]):
                                for pp, aff4 in enumerate(searchdic["Aff"]):
                                    resultsTensor[ii, jj, kk, ll, mm, nn, oo, pp] = sampleSpec(
                                        conc, kx, val, recMeans, Covs, np.array([mix, 1 - mix]), np.array([[aff1, aff2], [aff3, aff4]])
                                    )[1]

    maxSelec = np.amax(resultsTensor)
    maxSelecCoords = np.unravel_index(np.argmax(resultsTensor), resultsTensor.shape)
    maxParams = np.array(
        [
            searchdic["L0"][maxSelecCoords[0]],
            searchdic["Kx"][maxSelecCoords[1]],
            searchdic["Val"][maxSelecCoords[2]],
            searchdic["Mix"][maxSelecCoords[3]],
            searchdic["Aff"][maxSelecCoords[4]],
            searchdic["Aff"][maxSelecCoords[5]],
            searchdic["Aff"][maxSelecCoords[6]],
            searchdic["Aff"][maxSelecCoords[7]],
        ]
    )

    return maxSelec, maxParams
