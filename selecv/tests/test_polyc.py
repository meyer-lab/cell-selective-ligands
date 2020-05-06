import unittest
import numpy as np
from scipy.special import binom
from ..model import polyc, polyfc


def genPerm(len, sum):
    if len <= 1:
        yield [sum]
    else:
        for i in range(sum + 1):
            for sub in genPerm(len - 1, sum - i):
                yield sub + [i]


def multinomial(params):
    if len(params) == 1:
        return 1
    return binom(sum(params), params[-1]) * multinomial(params[:-1])


def polyfc2(L0, KxStar, f, Rtot, LigC, Kav):
    """ This function should give the same result as polyfc() but less efficient.
    This function is used for testing only. Use polyfc() for random complexes calculation"""
    LigC = np.array(LigC)
    assert LigC.ndim == 1
    LigC = LigC / np.sum(LigC)

    Cplx = np.array(list(genPerm(LigC.size, f)))
    Ctheta = np.exp(np.dot(Cplx, np.log(LigC).reshape(-1, 1))).flatten()
    Ctheta *= np.array([multinomial(Cplx[i, :]) for i in range(Cplx.shape[0])])
    assert abs(sum(Ctheta) - 1.0) < 1e-12

    return polyc(L0, KxStar, Rtot, Cplx, Ctheta, Kav)


class TestPolyc(unittest.TestCase):
    def test_equivalence(self):
        L0 = np.random.rand() * 10.0 ** np.random.randint(-15, -5)
        KxStar = np.random.rand() * 10.0 ** np.random.randint(-15, -5)
        f = np.random.randint(1, 10)
        nl = np.random.randint(1, 10)
        nr = np.random.randint(1, 10)
        Rtot = np.floor(100. + np.random.rand(nr) * (10. ** np.random.randint(4, 6, size=nr)))
        LigC = np.random.rand(nl) * (10. ** np.random.randint(1, 2, size=nl))
        Kav = np.random.rand(nl, nr) * (10. ** np.random.randint(3, 7, size=(nl, nr)))

        res = polyfc(L0, KxStar, f, Rtot, LigC, Kav)
        res2 = polyfc2(L0, KxStar, f, Rtot, LigC, Kav)

        self.assertTrue(abs(res[0] - res2[0]) < res[0] * 1e-7)
        self.assertTrue(abs(res[1] - res2[1]) < res[1] * 1e-7)
        self.assertAlmostEqual(np.sum(res[2]), res[0])


if __name__ == '__main__':
    unittest.main()
