import numpy as np
import pandas as pd
import seaborn as sns
from selecv.model import polyfc, polyc
from matplotlib.pyplot import plot
from sklearn.linear_model import LinearRegression


Kav = np.array([[5.88e7], [9.09e5], [0]])   # [C5, B22, NT]
Recep = {"MDA": 5.2e4, "SK": 2.2e5, "LNCaP": 2.8e6, "MCF": 3.8e6}


def model_predict(df, KxStar, LigC, polyfcM):
    predicted, measured = [], []
    for _, row in df.iterrows():
        if polyfcM:
            res = polyfc(row.monomer * 1e-9 / 8, KxStar, row.valency, [Recep["MCF"]], LigC, Kav)
        else:
            if LigC == [1, 0, 0]:
                res = polyc(row.monomer * 1e-9 / 8, KxStar, [Recep["MCF"]], [[row.valency, 0, 8 - row.valency]], [1], Kav)
            elif LigC == [0, 1, 0]:
                res = polyc(row.monomer * 1e-9 / 8, KxStar, [Recep["MCF"]], [[0, row.valency, 8 - row.valency]], [1], Kav)
        Lbound, Rbound = res[0], res[1]
        predicted.append(Rbound)
        measured.append(row.intensity)
    return predicted, measured


def fit_slope(KxStar, polyfcM=True):
    df1 = pd.read_csv("selecv/data/csizmar_s4a.csv")
    X1, Y1 = model_predict(df1, KxStar, [1, 0, 0], polyfcM)
    df1['predicted'] = X1
    df1['data'] = "C5"
    df2 = pd.read_csv("selecv/data/csizmar_s4b.csv")
    X2, Y2 = model_predict(df2, KxStar, [0, 1, 0], polyfcM)
    df2['predicted'] = X2
    df2['data'] = "B22"
    df = pd.concat([df1, df2])

    X, Y = np.array(X1 + X2).reshape(-1, 1), np.array(Y1 + Y2)
    lr = LinearRegression(fit_intercept=False)
    lr.fit(X, Y)
    print(lr.score(X, Y))
    ax = sns.lineplot(x='predicted', y='intensity', hue='data', style='valency', markers=True, data=df)
    return ax


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
            df = df.append({'Lig': str(lig), 'Recep': rec, 'value': res[1] * slope}, ignore_index=True)
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
            df = df.append({'Lig': str(lig), 'Recep': rec, 'value': res[1] * slope_C5}, ignore_index=True)
    for lig in [[0, 8, 0], [0, 4, 4]]:
        for rec in Recep.values():
            res = polyfc(50 * 1e-9, KxStar_B22, 8, [rec], lig, Kav)
            df = df.append({'Lig': str(lig), 'Recep': rec, 'value': res[1] * slope_B22}, ignore_index=True)
    ax = sns.lineplot(x='Recep', y='value', hue='Lig', style='Lig', markers=True, data=df)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set(xlim=(1e4, 1e7), ylim=(10, 1e5))
    return ax
