"""
Sample from population to calculation population specificities.
"""

import numpy as np
from copy import copy
from valentbind import polyfc, polyc


nsample = 200


cellPopulations = {
    r"$R_1^{lo}R_2^{lo}$": [2.2, 2.2, 1.5, 0.5, 45],
    r"$R_1^{med}R_2^{lo}$": [4, 2.2, 1.0, 0.5, 0],
    r"$R_1^{hi}R_2^{lo}$": [5.8, 2.2, 1.0, 0.5, 0],
    r"$R_1^{lo}R_2^{hi}$": [2.2, 5.8, 0.5, 1.0, 0],
    r"$R_1^{med}R_2^{hi}$": [4.0, 5.6, 1.5, 0.5, 45],
    r"$R_1^{hi}R_2^{med}$": [5.6, 4.0, 1.5, 0.5, 45],
    r"$R_1^{hi}R_2^{hi}$": [5.8, 5.8, 1.5, 0.5, 45],
    r"$R_1^{med}R_2^{med}$": [3.9, 3.9, 0.5, 2.0, 45],
}


def sigmapts(name, h=None, recFactor=1.0):
    if h is None:
        h = np.sqrt(3)
    l = copy(cellPopulations[name])
    l[0:2] = l[0:2] + np.log10(recFactor)
    x = np.array([l[0], l[1]])
    rot = np.array([[np.cos(np.deg2rad(l[4])), -np.sin(np.deg2rad(l[4]))], [np.sin(np.deg2rad(l[4])), np.cos(np.deg2rad(l[4]))]])
    srlamb = np.diag([l[2], l[3]])
    srcov = rot @ srlamb @ np.transpose(rot)
    return np.power(10, [x, x + h * srcov[:, 0], x - h * srcov[:, 0], x + h * srcov[:, 1], x - h * srcov[:, 1]])


def sigmaPop(name, L0, KxStar, f, LigC, Kav, quantity=0, h=None, recFactor=1.0):
    return np.array([polyfc(L0, KxStar, f, Rtot, LigC, Kav)[quantity] for Rtot in sigmapts(name, h=h, recFactor=recFactor)]).reshape(-1)


def sigmaPopC(name, L0, KxStar, Cplx, Ctheta, Kav, quantity=0, h=None):
    return np.array([polyc(L0, KxStar, Rtot, Cplx, Ctheta, Kav)[quantity][1] for Rtot in sigmapts(name, h=h)]).reshape(-1)


def sampleSpec(L0, KxStar, f, names, LigC, Kav):
    """
    Sample the specificity between populations.
    RtotMeans: Tuple of receptor population expression means.
    """
    quantsNum = sigmaPop(names[0], L0, KxStar, f, LigC, Kav)
    quants = np.zeros([len(names) - 1, len(quantsNum)**2])
    qmean = np.zeros([len(names) - 1])

    for ii in range(1, len(names)):
        calc = sigmaPop(names[ii], L0, KxStar, f, LigC, Kav)
        quants[ii - 1, :] = np.reshape(quantsNum.reshape(-1, 1) / calc, -1)
        qmean[ii - 1] = quantsNum[0] / calc[0]
        quants[ii - 1, :] = np.sort(quants[ii - 1])

    quants = np.min(quants, axis=0)
    mean = np.mean(np.log(quants))
    std = np.std(np.log(quants))
    res = np.array([np.exp(mean - std * 0.5), np.min(qmean), np.exp(mean + std * 0.5)])

    return res


def sampleSpecC(L0, KxStar, names, LigCplx, Ctheta, Kav):
    """
    Sample the specificity between populations.
    RtotMeans: Tuple of receptor population expression means.
    """

    quantsNum = sigmaPopC(names[0], L0, KxStar, LigCplx, Ctheta, Kav)
    quants = np.zeros([len(names) - 1, len(quantsNum)])
    qmean = np.zeros([len(names) - 1])

    for ii in range(1, len(names)):
        calc = sigmaPopC(names[ii], L0, KxStar, LigCplx, Ctheta, Kav)
        quants[ii - 1, :] = np.reshape(quantsNum.reshape(-1, 1) / calc, -1)
        qmean[ii - 1] = quantsNum[0] / calc[0]
        quants[ii - 1, :] = np.sort(quants[ii - 1])

    quants = np.min(quants, axis=0)
    mean = np.mean(np.log(quants))
    std = np.std(np.log(quants))
    res = np.array([np.exp(mean - std * 0.5), np.min(qmean), np.exp(mean + std * 0.5)])
    return res
