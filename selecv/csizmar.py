"""
Functions for reimplementing Csizmar et al.
"""
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.optimize import minimize
from sklearn.linear_model import LinearRegression
from selecv.model import polyfc


Kav = np.array([[5.88e7], [9.09e5], [0]])   # [C5, B22, NT]
Recep = {"MDA": 5.2e4, "SK": 2.2e5, "LNCaP": 2.8e6, "MCF": 3.8e6}


def model_predict(df, KxStarP, LigC, slopeP, Kav1, abund):
    "Gathers predicted and measured fluorescent intensities for a given population"
    predicted, measured = [], []
    for _, row in df.iterrows():
        res = polyfc(row.monomer * 1e-9 / 8, KxStarP, 8, abund, np.array(LigC) * row.valency / 8 + [0, 0, 1 - sum(np.array(LigC) * row.valency / 8)], Kav1)
        Lbound, _ = res[0] * slopeP, res[1]
        predicted.append(Lbound)
        measured.append(row.intensity)
    return predicted, measured


def fit_slope(ax, KxStarF, slopeC5, slopeB22, Kav2, abund):
    "Outputs predicted vs. Experimental fluorescent intensities for C5 and B22 binding"
    df1 = pd.read_csv("selecv/data/csizmar_s4a.csv")
    X1, Y1 = model_predict(df1, KxStarF, [1, 0, 0], slopeC5, Kav2, abund)
    df1['predicted'] = X1
    df1['data'] = "C5"

    df2 = pd.read_csv("selecv/data/csizmar_s4b.csv")
    X2, Y2 = model_predict(df2, KxStarF, [0, 1, 0], slopeB22, Kav2, abund)
    df2['predicted'] = X2
    df2['data'] = "B22"
    df = pd.concat([df1, df2])

    lr = LinearRegression(fit_intercept=False)
    X, Y = np.array(X1).reshape(-1, 1), np.array(Y1)
    lr.fit(X, Y)
    C5_score = lr.score(X, Y)
    X, Y = np.array(X2).reshape(-1, 1), np.array(Y2)
    lr.fit(X, Y)
    B22_score = lr.score(X, Y)

    sns.lineplot(x='predicted', y='intensity', hue='data', style='valency', markers=True, data=df, ax=ax)

    return C5_score, B22_score


KxStar = 10**-14.714
slope = 0.008677777424519703

KxStar_C5 = 10**-14.693
KxStar_B22 = 10**-12.734

slope_C5 = 0.008514426941736077
slope_B22 = 0.012855332053729724


def discrim():
    "Returns predicted fluorescent values over a range of abundances"
    df = pd.DataFrame(columns=['Lig', 'Recep', 'value'])
    for lig in [[8, 0, 0], [4, 0, 4], [0, 8, 0], [0, 4, 4]]:
        for rec in Recep.values():
            res = polyfc(50 * 1e-9, KxStar, 8, [rec], lig, Kav)
            Lbound, _ = res[0], res[1]
            df = df.append({'Lig': str(lig), 'Recep': rec, 'value': Lbound * slope * (lig[0] + lig[1])}, ignore_index=True)
    ax = sns.lineplot(x='Recep', y='value', hue='Lig', style='Lig', markers=True, data=df)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set(xlim=(1e4, 1e7), ylim=(10, 1e5))
    return ax


def discrim2(ax, KxStarD, slopeC5, slopeB22):
    "Returns predicted fluorescent values over a range of abundances with unique slopes for C5 and B22"
    df = pd.DataFrame(columns=['Lig', 'Recep', 'value'])
    for lig in [[8, 0, 0], [4, 0, 4]]:
        for rec in Recep.values():
            res = polyfc(50 * 1e-9, KxStarD, 8, [rec], lig, Kav)
            df = df.append({'Lig': str(lig), 'Recep': rec, 'value': res[0] * slopeC5}, ignore_index=True)  # * (lig[0] + lig[1])
    for lig in [[0, 8, 0], [0, 4, 4]]:
        for rec in Recep.values():
            res = polyfc(50 * 1e-9, KxStarD, 8, [rec], lig, Kav)
            df = df.append({'Lig': str(lig), 'Recep': rec, 'value': res[0] * slopeB22}, ignore_index=True)  # * (lig[0] + lig[1])
    sns.lineplot(x='Recep', y='value', hue='Lig', style='Lig', markers=True, data=df, ax=ax)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set(xlim=(1e4, 1e7), ylim=(10, 1e5))
    return ax


def xeno(ax, KxStarX, KavX):
    "Plots Xenograft targeting ratios"
    df = pd.DataFrame(columns=['Lig', 'ratio'])
    for lig in [[8, 0, 0], [4, 0, 4], [0, 8, 0], [0, 4, 4]]:
        mcf = polyfc(50 * 1e-9, KxStarX, 8, [Recep["MCF"]], lig, KavX)[0]
        mda = polyfc(50 * 1e-9, KxStarX, 8, [Recep["MDA"]], lig, KavX)[0]
        df = df.append({'Lig': str(lig), 'ratio': (mcf / mda)}, ignore_index=True)
    sns.barplot(x="Lig", y="ratio", data=df, ax=ax)
    #ax.set(ylim=(0, 100))
    return ax


def resids(x):
    "Least squares residual function"
    df1 = pd.read_csv("selecv/data/csizmar_s4a.csv")
    X1, Y1 = model_predict(df1, np.exp(x[0]), [1, 0, 0], x[1], [[np.exp(x[3])], [np.exp(x[4])], [0]], np.exp(x[5]))
    df2 = pd.read_csv("selecv/data/csizmar_s4b.csv")
    X2, Y2 = model_predict(df2, np.exp(x[0]), [0, 1, 0], x[2], [[np.exp(x[3])], [np.exp(x[4])], [0]], np.exp(x[5]))
    X1 = np.asarray(X1)
    X2 = np.asarray(X2)
    Y1 = np.asarray(Y1)
    Y2 = np.asarray(Y2)
    return sum(np.square(X2 - Y2) + np.square(X1 - Y1))


def fitfunc():
    "Runs least squares fitting for various model parameters, and returns the minimizers"
    x0 = np.array([np.log(10**-14.714), 0.01, 0.01, np.log(Kav[0])[0], np.log(Kav[1])[0], np.log(3.8e6)])  # KXSTAR, slopeC5, B22, KA C5, B22, receps MH-7
    bnds = ((None, None), (None, None), (None, None), (None, None), (None, None), (np.log(3.8e6) * 0.999, np.log(3.8e6) * 1.001))
    parampredicts = minimize(resids, x0, bounds=bnds, method='L-BFGS-B')
    return parampredicts.x
