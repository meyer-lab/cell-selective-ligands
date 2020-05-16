import numpy as np
import pandas as pd
import seaborn as sns
from selecv.model import polyfc
from sklearn.linear_model import LinearRegression


Kav = np.array([[5.88e7], [9.09e5], [0]])   # [C5, B22, NT]
Recep = {"MDA": 5.2e4, "SK": 2.2e5, "LNCaP": 2.8e6, "MCF": 3.8e6}


def model_predict(df, KxStar, LigC):
    predicted, measured = [], []
    for _, row in df.iterrows():
        res = polyfc(row.monomer * 1e-9 / 8, KxStar, row.valency, [Recep["MCF"]], LigC, Kav)
        Lbound, Rbound = res[0], res[1]
        predicted.append(Lbound)
        measured.append(row.intensity / row.valency)
    return predicted, measured


def fit_slope(KxStar):
    df1 = pd.read_csv("selecv/data/csizmar_s4a.csv")
    X1, Y1 = model_predict(df1, KxStar, [1, 0, 0])
    df1['predicted'] = X1
    df1['adjusted intensity'] = Y1
    df1['data'] = "C5"
    df2 = pd.read_csv("selecv/data/csizmar_s4b.csv")
    X2, Y2 = model_predict(df2, KxStar, [0, 1, 0])
    df2['predicted'] = X2
    df2['adjusted intensity'] = Y2
    df2['data'] = "B22"
    df = pd.concat([df1, df2])
    X, Y = np.array(X1 + X2).reshape(-1, 1), np.array(Y1 + Y2)
    lr = LinearRegression(fit_intercept=False)
    lr.fit(X, Y)
    ax = sns.lineplot(x='predicted', y='adjusted intensity', hue='data', style='valency', markers=True, data=df)
    return lr.score(X, Y), lr.coef_


KxStar = 10**-14.714
slope = 0.008677777424519703

KxStar_C5 = 10**-14.693
KxStar_B22 = 10**-12.734

slope_C5 = 0.008514426941736077
slope_B22 = 0.012855332053729724


def discrim():
    df = pd.DataFrame(columns=['Lig', 'Recep', 'value'])
    for lig in [[8, 0, 0], [4, 0, 4], [0, 8, 0], [0, 4, 4]]:
        for rec in Recep.values():
            res = polyfc(50 * 1e-9, KxStar, 8, [rec], lig, Kav)
            Lbound, Rbound = res[0], res[1]
            df = df.append({'Lig': str(lig), 'Recep': rec, 'value': Lbound * slope * (lig[0] + lig[1])}, ignore_index=True)
    ax = sns.lineplot(x='Recep', y='value', hue='Lig', style='Lig', markers=True, data=df)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set(xlim=(1e4, 1e7), ylim=(10, 1e5))
    return ax


def discrim2():
    df = pd.DataFrame(columns=['Lig', 'Recep', 'value'])
    for lig in [[8, 0, 0], [4, 0, 4]]:
        for rec in Recep.values():
            res = polyfc(50 * 1e-9, KxStar_C5, 8, [rec], lig, Kav)
            df = df.append({'Lig': str(lig), 'Recep': rec, 'value': res[0] * slope_C5 * (lig[0] + lig[1])}, ignore_index=True)
    for lig in [[0, 8, 0], [0, 4, 4]]:
        for rec in Recep.values():
            res = polyfc(50 * 1e-9, KxStar_B22, 8, [rec], lig, Kav)
            df = df.append({'Lig': str(lig), 'Recep': rec, 'value': res[0] * slope_B22 * (lig[0] + lig[1])}, ignore_index=True)
    ax = sns.lineplot(x='Recep', y='value', hue='Lig', style='Lig', markers=True, data=df)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set(xlim=(1e4, 1e7), ylim=(10, 1e5))
    return ax


def xeno(KxStar):
    df = pd.DataFrame(columns=['Lig', 'ratio'])
    for lig in [[8, 0, 0], [4, 0, 4], [0, 8, 0], [0, 4, 4]]:
        mcf = polyfc(50 * 1e-9, KxStar, 8, [Recep["MCF"]], lig, Kav)[0]
        mda = polyfc(50 * 1e-9, KxStar, 8, [Recep["MDA"]], lig, Kav)[0]
        df = df.append({'Lig': str(lig), 'ratio': (mcf / mda)}, ignore_index=True)
    print(df)
    ax = sns.barplot(x="Lig", y="ratio", data=df)
    ax.set(ylim=(0, 100))
    return ax
