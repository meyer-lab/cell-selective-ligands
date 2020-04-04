"""
Implementation of a simple multivalent binding model.
"""

import numpy as np
from numba import njit
from scipy.optimize import root


def paramCheck(Kav, Rtot, LigC):
    """ Common checks of input parameters. """
    Kav = np.array(Kav)
    Rtot = np.array(Rtot)
    LigC = np.array(LigC)
    assert Kav.ndim == 2
    assert LigC.ndim == 1
    assert Rtot.ndim <= 1
    assert Rtot.size == Kav.shape[1]
    assert np.isclose(np.sum(LigC), 1.0)
    return Kav, Rtot, LigC


@njit
def Req_func(Req, Rtot, LmonoA, AKxStar, f):
    """ Mass balance. Transformation to account for bounds. """
    Phisum = np.dot(AKxStar, Req.T)
    return Req + LmonoA * Req * (1 + Phisum) ** (f - 1) - Rtot


def polyfc(Lmono, KxStar, f, Rtot, LigC, Kav):
    """
    The main function. Generate all info for heterogenenous binding case
    Lmono: concentration of ligand monomers.
    KxStar: detailed balance-corrected Kx.
    f: valency
    Rtot: numbers of each receptor appearing on the cell.
    LigC: the composition of the mixture used.
    Kav: a matrix of Ka values. row = IgG's, col = FcgR's
    """
    # Data consistency check
    L0 = L0 / f
    Kav, Rtot, LigC = paramCheck(Kav, Rtot, LigC)
    assert LigC.size == Kav.shape[0]

    # Run least squares to get Req
    A = np.dot(LigC.T, Kav)
    L0fA = L0 * f * A
    AKxStar = A * KxStar

    # Solve Req by calling least_squares() and Req_func()
    lsq = root(Req_func, Rtot, method="lm", args=(Rtot, L0fA, AKxStar, f), options={"maxiter": 3000})
    assert lsq["success"], "Failure in rootfinding. " + str(lsq)
    Req = lsq["x"].reshape(1, -1)

    Phi = np.ones((LigC.size, Rtot.size + 1)) * LigC.reshape(-1, 1)
    Phi[:, :Rtot.size] *= Kav * Req * KxStar
    Phisum = np.sum(Phi[:, :Rtot.size])

    Lbound = Lmono / f / KxStar * ((1 + Phisum)**f - 1)
    Rbound = Lmono / KxStar * Phisum * (1 + Phisum)**(f - 1)

    return Lbound, Rbound


def polyfcm(KxStar, f, Rtot, Lig, Kav):
    return polyfc(np.sum(Lig) / f, KxStar, f, Rtot, Lig / np.sum(Lig), Kav)


def Req_func2(Req, L0, KxStar, Rtot, Cplx, Ctheta, Kav):
    Psi = np.ones((Kav.shape[0], Kav.shape[1] + 1))
    Psi[:, : Kav.shape[1]] *= Req * Kav * KxStar
    Psirs = np.sum(Psi, axis=1).reshape(-1, 1)
    Psinorm = (Psi / Psirs)[:, :-1]

    Rbound = L0 / KxStar * np.sum(Ctheta.reshape(-1, 1) * np.dot(Cplx, Psinorm) * np.exp(np.dot(Cplx, np.log1p(Psirs - 1))), axis=0)
    return Req + Rbound - Rtot


def polyc(L0, KxStar, Rtot, Cplx, Ctheta, Kav):
    """
    The main function to be called for multivalent binding
    :param L0: concentration of ligand complexes
    :param KxStar: Kx for detailed balance correction
    :param Rtot: numbers of each receptor on the cell
    :param Cplx: the monomer ligand composition of each complex
    :param Ctheta: the composition of complexes
    :param Kav: Ka for monomer ligand to receptors
    :return: Lbound
    """
    # Consistency check
    Kav, Rtot, Ctheta = paramCheck(Kav, Rtot, Ctheta)
    Cplx = np.array(Cplx)
    assert Cplx.ndim == 2
    assert Kav.shape[0] == Cplx.shape[1]
    assert Cplx.shape[0] == Ctheta.size

    # Solve Req
    lsq = root(Req_func2, Rtot, method="lm", args=(L0, KxStar, Rtot, Cplx, Ctheta, Kav), options={"maxiter": 3000})
    assert lsq["success"], "Failure in rootfinding. " + str(lsq)
    Req = lsq["x"].reshape(1, -1)

    # Calculate the results
    Psi = np.ones((Kav.shape[0], Rtot.size + 1))
    Psi[:, : Rtot.size] *= Req * Kav * KxStar
    Psirs = np.sum(Psi, axis=1).reshape(-1, 1)

    Lbound = L0 / KxStar * np.sum(Ctheta * np.expm1(np.dot(Cplx, np.log1p(Psirs - 1))).flatten())
    Rbound = L0 / KxStar * np.sum(Ctheta * np.dot(Cplx, 1 - 1 / Psirs).flatten()
                                  * np.exp(np.dot(Cplx, np.log(Psirs))).flatten())
    return Lbound, Rbound


def polycm(KxStar, Rtot, Cplx, Ltheta, Kav):
    return polyc(np.sum(Ltheta), KxStar, Rtot, Cplx, Ltheta / np.sum(Ltheta), Kav)
