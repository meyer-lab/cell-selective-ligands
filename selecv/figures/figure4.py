"""
Figure 4. Explore antagonism
"""
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import matplotlib.cm as cm
from scipy.optimize import minimize
from .figureCommon import subplotLabel, setFontSize, getSetup, cellPopulations, overlapCellPopulation
from .figure5 import genOnevsAll, genPopMeans
from valentbind import polyc
from ..sampling import sigmaPopC


def makeFigure():
    """ Make figure 4. """
    # Get list of axis objects
    ax, f = getSetup((10, 9), (3, 3))
    subplotLabel(ax[0:11])
    affDLsub = np.array([6, 10])
    fDLsub = 4
    ax[0].axis("off")
    optParams, DLaffs = optimizeDesignDL(ax[1], [r"$R_1^{med}R_2^{lo}$"], fDLsub, affDLsub, specPops=[r"$R_1^{hi}R_2^{lo}$"])
    heatmapDL(ax[2], np.exp(optParams[0]), np.exp(optParams[1]), np.array([[DLaffs[0], DLaffs[1]], [np.exp(optParams[4]), np.exp(optParams[5])]]),
              [0.5, 0.5], Cplx=np.array([[fDLsub, 0], [0, optParams[2]]]), vrange=(-7, 3), cbar=True, highlight=[r"$R_1^{med}R_2^{lo}$"], lowlight=[r"$R_1^{hi}R_2^{lo}$"])
    heatmapDL(ax[3], np.exp(optParams[0]), np.exp(optParams[1]), np.array([[DLaffs[0], DLaffs[1]], [np.exp(optParams[4]), np.exp(optParams[5])]]),
              [0.5, 0.5], Cplx=np.array([[fDLsub, 0], [0, optParams[2]]]), vrange=(3, 14), cbar=False, dead=True, highlight=[r"$R_1^{med}R_2^{lo}$"], lowlight=[r"$R_1^{hi}R_2^{lo}$"])
    heatmapDL(ax[4], np.exp(optParams[0]) / 2, np.exp(optParams[1]), np.array([[DLaffs[0], DLaffs[1]], [np.exp(optParams[4]), np.exp(optParams[5])]]),
              [0, 1], Cplx=np.array([[fDLsub, 0], [0, optParams[2]]]), vrange=(3, 14), cbar=True, dead=False, jTherap=True, highlight=[r"$R_1^{med}R_2^{lo}$"], lowlight=[r"$R_1^{hi}R_2^{lo}$"])

    valScanOpt(ax[5:7], [r"$R_1^{med}R_2^{lo}$"], specPops=[r"$R_1^{hi}R_2^{lo}$"])

    mixScanOpt(ax[7:9], [r"$R_1^{med}R_2^{lo}$"], specPops=[r"$R_1^{hi}R_2^{lo}$"])

    setFontSize(ax, 10.5, xsci=[2, 3, 4, 7, 8], ysci=[2, 3, 4])
    return f


cBnd = (np.log(1e-9), np.log(1.01e-9))


def minSelecFuncDL(x, targPop, offTpops, fDL, affDL):
    """ Provides the function to be minimized to get optimal selectivity with addition of dead ligand. """
    offTargetBound, targetBound = 0, 0
    for tPop in targPop:
        targetBound += sum(sigmaPopC(tPop, np.exp(x[0]), np.exp(x[1]), [[fDL, 0], [0, x[2]]], [0.5, 0.5], np.array([[affDL[0], affDL[1]], [np.exp(x[4]), np.exp(x[5])]]), quantity=0))
    for offTPop in offTpops:
        offTargetBound += sum(sigmaPopC(offTPop, np.exp(x[0]), np.exp(x[1]), [[fDL, 0], [0, x[2]]], [0.5, 0.5], np.array([[affDL[0], affDL[1]], [np.exp(x[4]), np.exp(x[5])]]), quantity=0))
    return offTargetBound / targetBound


def optimizeDesignDL(ax, targetPop, fDL, affDL, specPops=False):
    "Runs optimization and determines optimal parameters for selectivity of one population vs. another with inclusion of dead ligand"
    if not specPops:
        targPops, offTargPops = genOnevsAll(targetPop)
    else:
        targPops, offTargPops = targetPop, specPops

    npoints = 5
    ticks = np.full([npoints], None)
    affScan = np.logspace(affDL[0], affDL[1], npoints)
    ticks[0], ticks[-1] = "1e" + str(9 - affDL[0]), "1e" + str(9 - affDL[1])
    bounds = (cBnd, (np.log(1e-15), np.log(1e-9)), (0.99, 1), (0, 1), (np.log(1e2), np.log(1e10)), (np.log(1e2), np.log(1e10)))
    xnot = np.array([np.log(1e-9), np.log(1e-9), 1, 1, np.log(10e8), np.log(10e8)])
    ratioDF = pd.DataFrame(columns=affScan, index=affScan)

    for ii, aff1 in enumerate(affScan):
        sampMeans = np.zeros(npoints)
        for jj, aff2 in enumerate(np.flip(affScan)):
            optimized = minimize(minSelecFuncDL, xnot, bounds=np.array(bounds), args=(targPops, offTargPops, fDL, [aff1, aff2]))
            assert optimized.success
            sampMeans[jj] = len(offTargPops) / optimized.fun
        ratioDF[ratioDF.columns[ii]] = sampMeans

    Cbar = True
    sns.heatmap(ratioDF, ax=ax, xticklabels=ticks, yticklabels=np.flip(ticks), vmin=0, vmax=16, cbar=Cbar, cbar_kws={'label': 'Selectivity Ratio w Antagonist', 'aspect': 40}, annot=True)
    ax.set(xlabel="Antagonist $R_1$ Affinity ($K_d$, in nM)", ylabel="Antagonist $R_2$ Affinity ($K_d$, in nM)")

    if specPops:
        ax.set_title("Agonist bound with antagonist, " + targetPop[0] + " vs " + specPops[0], fontsize=8)
    else:
        ax.set_title("Agonist bound with antagonist, " + targetPop[0] + " vs all", fontsize=8)

    maxindices = ratioDF.to_numpy()
    (i, j) = np.unravel_index(maxindices.argmax(), maxindices.shape)
    maxaff1 = affScan[j]
    maxaff2 = affScan[npoints - 1 - i]
    optimized = minimize(minSelecFuncDL, xnot, bounds=np.array(bounds), args=(targPops, offTargPops, fDL, [maxaff1, maxaff2]))
    assert optimized.success

    return optimized.x, np.array([maxaff1, maxaff2])


def heatmapDL(ax, L0, KxStar, Kav, Comp, Cplx=None, vrange=(-2, 4), title="", cbar=True, dead=False, jTherap=False, highlight=[], lowlight=[]):
    "Makes a heatmap with modified cell population abundances according to Antagonist binding"
    nAbdPts = 70
    abunds = np.array(list(cellPopulations.values()))[:, 0:2]
    abundRange = (np.amin(abunds) - 1, np.amax(abunds) + 1)
    abundScan = np.logspace(abundRange[0], abundRange[1], nAbdPts)
    if dead:
        func = np.vectorize(lambda abund1, abund2: polyc(L0, KxStar, [abund1, abund2], Cplx, Comp, Kav)[0][0])
        title = "Antagonist Bound"
    else:
        func = np.vectorize(lambda abund1, abund2: polyc(L0, KxStar, [abund1, abund2], Cplx, Comp, Kav)[0][1])
        if jTherap:
            title = "Agonist Bound without Antagonist"
        else:
            title = "Agonist Bound with Antagonist"

    X, Y = np.meshgrid(abundScan, abundScan)
    logZ = np.log(func(X, Y))
    contours = ax.contour(X, Y, logZ, levels=np.arange(-20, 20, 1.0), colors="black", linewidths=0.5)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title(title)
    plt.clabel(contours, inline=True, fontsize=6)
    ax.pcolor(X, Y, logZ, cmap='RdYlGn', vmin=vrange[0], vmax=vrange[1])
    norm = plt.Normalize(vmin=vrange[0], vmax=vrange[1])
    ax.set(xlabel="Receptor 1 Abundance (#/cell)", ylabel='Receptor 2 Abundance (#/cell)')
    if cbar:
        cbar = ax.figure.colorbar(cm.ScalarMappable(norm=norm, cmap='RdYlGn'), ax=ax, aspect=40)
        cbar.set_label("Log Ligand Bound")
    overlapCellPopulation(ax, abundRange, data=cellPopulations, highlight=highlight, lowlight=lowlight)


def minSelecFuncDLVal(x, targPop, offTpops, fDL):
    """ Provides the function to be minimized to get optimal selectivity with addition of dead ligand. """
    offTargetBound, targetBound = 0, 0
    for tPop in targPop:
        targetBound += sum(sigmaPopC(tPop, np.exp(x[0]), np.exp(x[1]), [[fDL, 0], [0, x[2]]], [0.5, 0.5], np.array([[np.exp(x[5]), np.exp(x[6])], [np.exp(x[3]), np.exp(x[4])]]), quantity=0))
    for offTPop in offTpops:
        offTargetBound += sum(sigmaPopC(offTPop, np.exp(x[0]), np.exp(x[1]), [[fDL, 0], [0, x[2]]], [0.5, 0.5], np.array([[np.exp(x[5]), np.exp(x[6])], [np.exp(x[3]), np.exp(x[4])]]), quantity=0))
    return offTargetBound / targetBound


def valScanOpt(ax, targetPop, specPops=False):
    """Scans through antagonist valencies and finds best specificity and affinities"""
    vals = np.linspace(2, 8, num=7)
    resultDF = pd.DataFrame(columns=["Valency", "Specificity"])
    AffDF = pd.DataFrame(columns=["Agonist Mix", "Receptor", "Affinity", "Ligand Type"])

    if not specPops:
        targPops, offTargPops = genOnevsAll(targetPop)
    else:
        targPops, offTargPops = targetPop, specPops

    bounds = (cBnd, (np.log(1e-15), np.log(1e-9)), (0.99, 1), (np.log(1e2), np.log(1e10)), (np.log(1e2), np.log(1e10)), (np.log(1e2), np.log(1e10)), (np.log(1e2), np.log(1e10)))
    xnot = np.array([np.log(1e-9), np.log(1e-9), 1, np.log(10e4), np.log(10e8), np.log(10e4), np.log(10e8)])

    for ii, val in enumerate(vals):
        optimized = minimize(minSelecFuncDLVal, xnot, bounds=np.array(bounds), args=(targPops, offTargPops, val))
        assert optimized.success
        selec = len(offTargPops) / optimized.fun
        optimArr = np.exp(optimized.x[3:7])
        optimArr = np.log10(optimArr)
        optimArr = optimArr * -1 + 9
        resultDF = resultDF.append(pd.DataFrame({"Valency": [val], "Specificity": selec}))
        AffDF = AffDF.append(pd.DataFrame({"Valency": [val], "Receptor": r"$R_{1} Affinity$", "Affinity": optimArr[0], "Ligand Type": "Agonist"}))
        AffDF = AffDF.append(pd.DataFrame({"Valency": [val], "Receptor": r"$R_{2} Affinity$", "Affinity": optimArr[1], "Ligand Type": "Agonist"}))
        AffDF = AffDF.append(pd.DataFrame({"Valency": [val], "Receptor": r"$R_{1} Affinity$", "Affinity": optimArr[2], "Ligand Type": "Antagonist"}))
        AffDF = AffDF.append(pd.DataFrame({"Valency": [val], "Receptor": r"$R_{2} Affinity$", "Affinity": optimArr[3], "Ligand Type": "Antagonist"}))

    sns.lineplot(data=resultDF, x="Valency", y="Specificity", ax=ax[0], palette='k')
    ax[0].set(xlabel="Antagonist Valency", ylabel="Optimal Specificty", ylim=(0, 30))
    ax[0].set_xticks(np.linspace(2, 8, num=4))
    sns.lineplot(data=AffDF, x="Valency", y="Affinity", hue="Receptor", style="Ligand Type", ax=ax[1])
    ax[1].set(xlabel="Antagonist Valency", ylabel=r"$K_d$ ($log_{10}$(nM))", title="Agonist Optimal Affinity", ylim=((-2, 8)))
    ax[1].set_xticks(np.linspace(2, 8, num=4))


def minSelecFuncDLMix(x, targPop, offTpops, antMix):
    """ Provides the function to be minimized to get optimal selectivity with addition of dead ligand. """
    offTargetBound, targetBound = 0, 0
    totC = 1e-9 + antMix
    mix = [1e-9 / totC, antMix / totC]
    for tPop in targPop:
        targetBound += sum(sigmaPopC(tPop, totC, np.exp(x[1]), [[4, 0], [0, x[2]]], mix, np.array([[np.exp(x[5]), np.exp(x[6])], [np.exp(x[3]), np.exp(x[4])]]), quantity=0))
    for offTPop in offTpops:
        offTargetBound += sum(sigmaPopC(offTPop, totC, np.exp(x[1]), [[4, 0], [0, x[2]]], mix,
                              np.array([[np.exp(x[5]), np.exp(x[6])], [np.exp(x[3]), np.exp(x[4])]]), quantity=0))
    return offTargetBound / targetBound


def mixScanOpt(ax, targetPop, specPops=False):
    """Scans through antagonist valencies and finds best specificity and affinities"""
    mixs = np.logspace(-12, -6, num=7)
    resultDF = pd.DataFrame(columns=["Agonist Mix", "Specificity"])
    AffDF = pd.DataFrame(columns=["Agonist Mix", "Receptor", "Affinity", "Ligand Type"])
    if not specPops:
        targPops, offTargPops = genOnevsAll(targetPop)
    else:
        targPops, offTargPops = targetPop, specPops

    bounds = (cBnd, (np.log(1e-15), np.log(1e-9)), (0.99, 1), (np.log(1e2), np.log(1e10)), (np.log(1e2), np.log(1e10)), (np.log(1e2), np.log(1e10)), (np.log(1e2), np.log(1e10)))
    xnot = np.array([np.log(1e-9), np.log(1e-9), 1, np.log(1e4), np.log(1e8), np.log(1e4), np.log(1e8)])

    for ii, mix in enumerate(mixs):
        optimized = minimize(minSelecFuncDLMix, xnot, bounds=np.array(bounds), args=(targPops, offTargPops, mix))
        assert optimized.success
        selec = len(offTargPops) / optimized.fun
        optimArr = np.exp(optimized.x[3:7])
        optimArr = np.log10(optimArr)
        optimArr = optimArr * -1 + 9
        resultDF = resultDF.append(pd.DataFrame({"Agonist Mix": [mix/1e-9], "Specificity": selec}))
        AffDF = AffDF.append(pd.DataFrame({"Agonist Mix": [mix/1e-9], "Receptor": r"$R_{1} Affinity$", "Affinity": optimArr[0], "Ligand Type": "Agonist"}))
        AffDF = AffDF.append(pd.DataFrame({"Agonist Mix": [mix/1e-9], "Receptor": r"$R_{2} Affinity$", "Affinity": optimArr[1], "Ligand Type": "Agonist"}))
        AffDF = AffDF.append(pd.DataFrame({"Agonist Mix": [mix/1e-9], "Receptor": r"$R_{1} Affinity$", "Affinity": optimArr[2], "Ligand Type": "Antagonist"}))
        AffDF = AffDF.append(pd.DataFrame({"Agonist Mix": [mix/1e-9], "Receptor": r"$R_{2} Affinity$", "Affinity": optimArr[3], "Ligand Type": "Antagonist"}))

    sns.lineplot(data=resultDF, x="Agonist Mix", y="Specificity", ax=ax[0], palette='k')
    ax[0].set(xlabel="Antagonist Concentration (nM)", ylabel="Optimal Specificity", xlim=(1e-3, 1e3), ylim=(0, 25),
              xscale="log")
    ax[0].set_xticks(np.logspace(-3, 3, num=4))
    sns.lineplot(data=AffDF, x="Agonist Mix", y="Affinity", hue="Receptor", style="Ligand Type", ax=ax[1])
    ax[1].set(xlabel="Antagonist Concentration (nM)", ylabel=r"$K_d$ ($log_{10}$(nM))", title="Ligand Optimal Affinities",
              xlim=(1e-3, 1e3), ylim=((-2, 8)), xscale="log")
    ax[1].set_xticks(np.logspace(-3, 3, num=4))
