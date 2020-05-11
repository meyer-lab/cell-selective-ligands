import numpy as np
import pandas as pd
from selecv.model import polyfc, polyc
from matplotlib.pyplot import plot
from sklearn.linear_model import LinearRegression


Kav = np.array([[5.88e7], [9.09e5], [0]])   # [C5, B22, NT]
Recep = {"MDA": 5.2e4, "SK": 2.2e5, "LNCaP": 2.8e6, "MCF": 3.8e6}


def predict(df, KxStar, LigC, polyfcM):
    predicted, measured = [], []
    for _, row in df.iterrows():
        if polyfcM:
            res = polyfc(row.monomer * 1e-9 / 8, KxStar, row.valency, [Recep["MCF"]], LigC, Kav)
        else:
            if LigC ==[1, 0, 0]:
                res = polyc(row.monomer * 1e-9 / 8, KxStar, [Recep["MCF"]], [[row.valency, 0, 8 - row.valency]], [1], Kav)
            elif LigC == [0, 1, 0]:
                res = polyc(row.monomer * 1e-9 / 8, KxStar, [Recep["MCF"]], [[0, row.valency, 8 - row.valency]], [1], Kav)
        Lbound, Rbound = res[0], res[1]
        predicted.append(Rbound)
        measured.append(row.intensity)
    return predicted, measured


def fit_slope(KxStar, polyfcM=False):
    df = pd.read_csv("selecv/data/csizmar_s4a.csv")
    X1, Y1 = predict(df, KxStar, [1, 0, 0], polyfcM)
    df = pd.read_csv("selecv/data/csizmar_s4b.csv")
    X2, Y2 = predict(df, KxStar, [0, 1, 0], polyfcM)
    X, Y = np.array(X1 + X2).reshape(-1, 1), np.array(Y1 + Y2)
    lr = LinearRegression(fit_intercept=False)
    lr.fit(X, Y)
    print(lr.score(X, Y))
    plt = plot(X1, Y1, 'or')
    plot(X2, Y2, 'ob')
    return plt#plt#lr.coef_[0]#plt#


KxStar = 10**-14.714
