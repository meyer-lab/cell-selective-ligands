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
xNaught = [4.0, 1, 10e-9, 10e-9, 10e-9, 10e-9]  # Conc, KxStar, Valency, Mix1, aff11, aff12, aff21, aff22
xBnds = ((1, 32), (1, 1), (10e-9, 10e-9), (10e-9, 10e-9), (10e-9, 10e-9), (10e-9, 10e-9))


def makeFigure():
    """ Make figure 5. """
    # Get list of axis objects
    ax, f = getSetup((7, 6), (2, 3))
    subplotLabel(ax)

    _, populationsdf = getPopDict()
    gridSearchTry(populationsdf, ['Pop3', 'Pop2'])
    _, _, _ = optimizeDesign(populationsdf, ['Pop3', 'Pop2'])

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

    optimized = minimize(minSelecFunc, xNaught, bounds=xBnds, method='L-BFGS-B', args=(recMeans, Covs), options={'eps': 1, 'disp': True})
    params = optimized.x
    optSelec = sampleSpec(ligConc, KxStar, params[0], recMeans, Covs, np.array([params[1], 1 - params[1]]), np.array([[params[2], params[3]], [params[4], params[5]]]))[1]
    selecNot = sampleSpec(ligConc, KxStar, xNaught[0], recMeans, Covs, np.array([xNaught[1], 1 - xNaught[1]]), np.array([[xNaught[2], xNaught[3]], [xNaught[4], xNaught[5]]]))[1]

    return optimized, optSelec, selecNot


searchdic = {'L0': np.logspace(-11, -8, 4),
             'Kx': np.logspace(-12, -8, 5),
             'Val': np.logspace(0.0, 4.0, base=2.0, num=5),
             'Mix': np.linspace(0, 0.5, 2),
             'Aff': np.logspace(5, 9, 2)
             }


def gridSearchTry(df, popList):
    recMeans, Covs = [], []
    for ii, pop in enumerate(popList):
        dfPop = df[df['Population'] == pop]
        recMeans.append(np.array([dfPop['Receptor_1'].to_numpy(), dfPop['Receptor_2'].to_numpy()]).flatten())
        Covs.append(dfPop.Covariance_Matrix.to_numpy()[0])

    resultsTensor = np.zeros([4, 5, 5, 3, 5, 5, 5, 5])
    for ii, conc in enumerate(searchdic['L0']):
        print(conc)
        for jj, kx in enumerate(searchdic['Kx']):
            print(kx)
            for kk, val in enumerate(searchdic['Val']):
                for ll, mix in enumerate(searchdic['Mix']):
                    for mm, aff1 in enumerate(searchdic['Aff']):
                        for nn, aff2 in enumerate(searchdic['Aff']):
                            for oo, aff3 in enumerate(searchdic['Aff']):
                                for pp, aff4 in enumerate(searchdic['Aff']):
                                    resultsTensor[ii, jj, kk, ll, mm, nn, oo, pp] = sampleSpec(conc, kx, val, recMeans, Covs, np.array([mix, 1 - mix]), np.array([[aff1, aff2], [aff3, aff4]]))[1]

    maxSelec = np.amax(resultsTensor)
    print(maxSelec)
    maxSelecCoords = np.unravel_index(np.argmax(resultsTensor), resultsTensor.shape)
    print(maxSelecCoords)
    maxParams = np.array([searchdic['L0'][maxSelecCoords[0]], searchdic['Kx'][maxSelecCoords[1]], searchdic['Val'][maxSelecCoords[2]], searchdic['Mix'][maxSelecCoords[3]],
                          searchdic['Aff'][maxSelecCoords[4]], searchdic['Aff'][maxSelecCoords[5]], searchdic['Aff'][maxSelecCoords[6]], searchdic['Aff'][maxSelecCoords[7]]])
    print(maxParams)

    return maxSelec, maxParams
