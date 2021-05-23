"""
Functions for reimplementing Csizmar et al.
"""
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.optimize import minimize
from sklearn.linear_model import LinearRegression
from valentbind import polyfc


Kav = np.array([[5.88e7], [9.09e5], [0]])   # [C5, B22, NT]
Recep = {"MDA": 5.2e4, "SK": 2.2e5, "LNCaP": 2.8e6, "MCF": 3.8e6}


def model_predict(df, KxStarP, LigC, slopeP, Kav1, abund, valencies=False):
    "Gathers predicted and measured fluorescent intensities for a given population"
    predicted, measured = [], []

    for _, row in df.iterrows():
        if not valencies:
            res = polyfc(row.monomer * 1e-9 / 8, KxStarP, 8, abund, np.array(LigC) * row.valency / 8 + [0, 0, 1 - sum(np.array(LigC) * row.valency / 8)], Kav1)
        else:
            ind = int(row.valency)
            res = polyfc(row.monomer * 1e-9 / 8, KxStarP, 8, abund, np.array(LigC) * valencies[ind] / 8 + [0, 0, 1 - sum(np.array(LigC) * valencies[ind] / 8)], Kav1)
            assert(np.array(np.array(LigC) * valencies[ind] / 8 + [0, 0, 1 - sum(np.array(LigC) * valencies[ind] / 8)]).all()
                   == np.array(np.array(LigC) * row.valency / 8 + [0, 0, 1 - sum(np.array(LigC) * row.valency / 8)]).all())

        Lbound, _ = res[0] * slopeP, res[1]
        predicted.append(Lbound)
        measured.append(row.intensity)

    return np.array(predicted), np.array(measured)


def fit_slope(ax, KxStarF, slopeC5, slopeB22, Kav2, abund, valencies=False):
    "Outputs predicted vs. Experimental fluorescent intensities for C5 and B22 binding"
    df1 = pd.read_csv("selecv/data/csizmar_s4a.csv")
    X1, Y1 = model_predict(df1, KxStarF, [1, 0, 0], slopeC5, Kav2, abund, valencies)
    df1["Predicted"] = X1
    df1["data"] = "C5"

    df2 = pd.read_csv("selecv/data/csizmar_s4b.csv")
    X2, Y2 = model_predict(df2, KxStarF, [0, 1, 0], slopeB22, Kav2, abund, valencies)
    df2["Predicted"] = X2
    df2["data"] = "B22"
    df = pd.concat([df1, df2])
    df = df.rename(columns={"valency": "Valency", "data": "Clone", "intensity": "Measured Fluorescent Intensity"})

    lr = LinearRegression(fit_intercept=False)
    X11, Y11 = np.array(X1).reshape(-1, 1), np.array(Y1)
    lr.fit(X11, Y11)
    C5_score = lr.score(X11, Y11)
    X22, Y22 = np.array(X2).reshape(-1, 1), np.array(Y2)
    lr.fit(X22, Y22)
    B22_score = lr.score(X22, Y22)
    print(lr.score(np.append(X11, X22).reshape(-1, 1), np.append(Y11, Y22)))

    sns.lineplot(x="Predicted", y="Measured Fluorescent Intensity", hue="Clone", style="Valency", markers=True, data=df, ax=ax)

    return C5_score, B22_score


KxStar = 10 ** -14.714
slope = 0.008677777424519703

KxStar_C5 = 10 ** -14.693
KxStar_B22 = 10 ** -12.734

slope_C5 = 0.008514426941736077
slope_B22 = 0.012855332053729724

ligandDict = {"[8, 0, 0]": "Octovalent C5", "[4, 0, 4]": "Tetravalent C5", "[0, 8, 0]": "Octovalent B22", "[0, 4, 4]": "Tetravalent B22"}


def discrim2(ax, KxStarD, slopeC5, slopeB22, KavD, valencies=False):
    "Returns predicted fluorescent values over a range of abundances with unique slopes for C5 and B22"
    df = pd.DataFrame(columns=["Ligand", "Receptor", "value"])
    if not valencies:
        for lig in [[8, 0, 0], [4, 0, 4]]:
            for rec in Recep.values():
                res = polyfc(50 * 1e-9, KxStarD, 8, [rec], lig, KavD)
                df = df.append({"Ligand": ligandDict[str(lig)], "Recep": rec, "value": res[0] * slopeC5}, ignore_index=True)  # * (lig[0] + lig[1])
        for lig in [[0, 8, 0], [0, 4, 4]]:
            for rec in Recep.values():
                res = polyfc(50 * 1e-9, KxStarD, 8, [rec], lig, KavD)
                df = df.append({"Ligand": ligandDict[str(lig)], "Recep": rec, "value": res[0] * slopeB22}, ignore_index=True)  # * (lig[0] + lig[1])

    else:
        for lig in [[valencies[0], 0, 8 - valencies[0]], [valencies[1], 0, 8 - valencies[1]]]:
            for rec in Recep.values():
                res = polyfc(50 * 1e-9, KxStarD, 8, [rec], lig, KavD)
                df = df.append({"Ligand": ligandDict[str(lig)], "Recep": rec, "value": res[0] * slopeC5}, ignore_index=True)  # * (lig[0] + lig[1])
        for lig in [[0, valencies[0], 8 - valencies[0]], [0, valencies[1], 8 - valencies[1]]]:
            for rec in Recep.values():
                res = polyfc(50 * 1e-9, KxStarD, 8, [rec], lig, KavD)
                df = df.append({"Ligand": ligandDict[str(lig)], "Recep": rec, "value": res[0] * slopeB22}, ignore_index=True)  # * (lig[0] + lig[1])
    sns.lineplot(x="Recep", y="value", hue="Ligand", style="Ligand", markers=True, data=df, ax=ax)
    ax.set(xlabel="Receptor Abundance", ylabel="Ligand Bound")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set(xlim=(1e4, 1e7), ylim=(10, 1e5))
    return ax


def xeno(ax, KxStarX, KavX):
    "Plots Xenograft targeting ratios"
    df = pd.DataFrame(columns=["Ligand", "ratio"])
    for lig in [[8, 0, 0], [4, 0, 4], [0, 8, 0], [0, 4, 4]]:
        mcf = polyfc(50 * 1e-9, KxStarX, 8, [Recep["MCF"]], lig, KavX)[0]
        mda = polyfc(50 * 1e-9, KxStarX, 8, [Recep["MDA"]], lig, KavX)[0]
        df = df.append({"Ligand": ligandDict[str(lig)], "ratio": (mcf / mda)}, ignore_index=True)
    sns.barplot(x="Ligand", y="ratio", data=df, ax=ax)
    ax.set(xlabel="", ylabel="Binding Ratio")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=25, horizontalalignment='right')
    ax.set(ylim=(0, 200))
    return ax


def resids(x):
    "Least squares residual function"
    valpack = False  # np.array([x[6], x[7], x[8], x[9]])

    df1 = pd.read_csv("selecv/data/csizmar_s4a.csv")
    X1, Y1 = model_predict(df1, np.exp(x[0]), [1, 0, 0], x[1], [[np.exp(x[3])], [np.exp(x[4])], [0]], np.exp(x[5]), valpack)
    df2 = pd.read_csv("selecv/data/csizmar_s4b.csv")
    X2, Y2 = model_predict(df2, np.exp(x[0]), [0, 1, 0], x[2], [[np.exp(x[3])], [np.exp(x[4])], [0]], np.exp(x[5]), valpack)
    return np.linalg.norm(X2 - Y2) + np.linalg.norm(X1 - Y1)


def fitfunc():
    "Runs least squares fitting for various model parameters, and returns the minimizers"
    x0 = np.array([np.log(10 ** -14.714), 0.01, 0.01, np.log(Kav[0])[0], np.log(Kav[1])[0], np.log(3.8e6), 1, 2, 4, 8])  # KXSTAR, slopeC5, slopeB22, KA C5, KA, B22, receps MH-7
    bnds = ((None, None), (None, None), (None, None), (None, None), (None, None), (np.log(3.8e6) * 0.9999, np.log(3.8e6) * 1.0001), (1, 1.01), (2, 2.01), (4, 4.01), (8, 8.01))
    parampredicts = minimize(resids, x0, jac="3-point", bounds=bnds)
    assert parampredicts.success
    return parampredicts.x
