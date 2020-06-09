import numpy as np
from selecv.model import *


L0 = 1e-9
KxStar = 1e-12
Rtot1 = [1e4, 1e4]
Rtot2 = [1e4, 1e2]
Cplx = np.array([[1, 1]])
Ctheta = [1.0]
Kav = np.array([[1e6, 5e4], [5e4, 1e6]])

for KxStar in np.logspace(-15, -9, num = 7):
    Lbound1, _ = polyc(L0, KxStar, Rtot1, Cplx, Ctheta, Kav)
    Lbound2, _ = polyc(L0, KxStar, Rtot2, Cplx, Ctheta, Kav)
    ratio = Lbound1[0] / Lbound2[0]
    print(KxStar, ratio)
