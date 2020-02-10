"""
Implementation of a simple multivalent binding model.
"""

import numpy as np
from scipy.optimize import root


def Req_func(Req, Rtot, L0fA, AKxStar, f):
    """ Mass balance. Transformation to account for bounds. """
    Phisum = np.dot(AKxStar, Req.T)
    return Req + L0fA * Req * (1 + Phisum)**(f - 1) - Rtot


def polyfc(L0, KxStar, f, Rtot, IgGC, Kav):
    """
    The main function. Generate all info for heterogenenous binding case
    L0: concentration of ligand.
    KxStar: detailed balance-corrected Kx.
    f: valency
    Rtot: numbers of each receptor appearing on the cell.
    IgGC: the composition of the mixture IgGC used.
    Kav: a matrix of Ka values. row = IgG's, col = FcgR's
    (Optional: the activity indices ActV)
    """
    # Data consistency check
    Kav = np.array(Kav)
    Rtot = np.array(Rtot)
    assert Rtot.ndim <= 1
    IgGC = np.array(IgGC)
    assert IgGC.ndim <= 1
    IgGC = IgGC / np.sum(IgGC)
    assert IgGC.size == Kav.shape[0]
    assert Rtot.size == Kav.shape[1]
    assert Kav.ndim == 2

    # Run least squares to get Req
    Req = Req_Regression(L0, KxStar, f, Rtot, IgGC, Kav)

    nr = Rtot.size  # the number of different receptors

    Phi = np.ones((IgGC.size, nr + 1)) * IgGC.reshape(-1, 1)
    Phi[:, :nr] *= Kav * Req * KxStar
    Phisum = np.sum(Phi[:, :nr])

    Lbound = L0 / KxStar * ((1 + Phisum)**f - 1)
    Rbound = L0 / KxStar * f * Phisum * (1 + Phisum)**(f - 1)

    return Lbound, Rbound


def Req_Regression(L0, KxStar, f, Rtot, IgGC, Kav):
    '''Run least squares regression to calculate the Req vector'''
    A = np.dot(IgGC.T, Kav)
    L0fA = L0 * f * A
    AKxStar = A * KxStar

    # Identify an initial guess just on max monovalent interaction
    # Correction factor at end is just empirical
    x0 = np.max(L0fA, axis=0)
    x0 = np.multiply(1.0 - np.divide(x0, 1 + x0), Rtot)

    # Solve Req by calling least_squares() and Req_func()
    lsq = root(Req_func, x0, method="lm", args=(Rtot, L0fA, AKxStar, f), options={'maxiter': 3000})

    assert lsq['success'], \
        "Failure in rootfinding. " + str(lsq)

    return lsq['x'].reshape(1, -1)
