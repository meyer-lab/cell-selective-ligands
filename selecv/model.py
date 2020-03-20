"""
Implementation of a simple multivalent binding model.
"""

import numpy as np
from numba import njit
from scipy.optimize import root


@njit
def Req_func(Req, Rtot, L0fA, AKxStar, f):
    """ Mass balance. Transformation to account for bounds. """
    Phisum = np.dot(AKxStar, Req.T)
    return Req + L0fA * Req * (1 + Phisum)**(f - 1) - Rtot


def polyfc(L0, KxStar, f, Rtot, LigC, Kav):
    """
    The main function. Generate all info for heterogenenous binding case
    L0: concentration of ligand.
    KxStar: detailed balance-corrected Kx.
    f: valency
    Rtot: numbers of each receptor appearing on the cell.
    LigC: the composition of the mixture used.
    Kav: a matrix of Ka values. row = IgG's, col = FcgR's
    """
    # Data consistency check
    Kav = np.array(Kav)
    Rtot = np.array(Rtot)
    assert Rtot.ndim <= 1
    LigC = np.array(LigC)
    assert LigC.ndim <= 1
    LigC = LigC / np.sum(LigC)
    assert LigC.size == Kav.shape[0]
    assert Rtot.size == Kav.shape[1]
    assert Kav.ndim == 2

    # Run least squares to get Req
    Req = Req_Regression(L0, KxStar, f, Rtot, LigC, Kav)

    nr = Rtot.size  # the number of different receptors

    Phi = np.ones((LigC.size, nr + 1)) * LigC.reshape(-1, 1)
    Phi[:, :nr] *= Kav * Req * KxStar
    Phisum = np.sum(Phi[:, :nr])

    Lbound = L0 / KxStar * ((1 + Phisum)**f - 1)

    return Lbound


def Req_Regression(L0, KxStar, f, Rtot, LigC, Kav):
    '''Run least squares regression to calculate the Req vector'''
    A = np.dot(LigC.T, Kav)
    L0fA = L0 * f * A
    AKxStar = A * KxStar

    # Identify an initial guess just on max monovalent interaction
    x0 = np.max(L0fA, axis=0)
    x0 = np.multiply(1.0 - np.divide(x0, 1 + x0), Rtot)

    # Solve Req by calling least_squares() and Req_func()
    lsq = root(Req_func, x0, method="lm", args=(Rtot, L0fA, AKxStar, f), options={'maxiter': 3000})

    assert lsq['success'], \
        "Failure in rootfinding. " + str(lsq)

    return lsq['x'].reshape(1, -1)
