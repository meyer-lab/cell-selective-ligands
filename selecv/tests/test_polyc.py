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
        Rtot = np.floor(100.0 + np.random.rand(nr) * (10.0 ** np.random.randint(4, 6, size=nr)))
        LigC = np.random.rand(nl) * (10.0 ** np.random.randint(1, 2, size=nl))
        Kav = np.random.rand(nl, nr) * (10.0 ** np.random.randint(3, 7, size=(nl, nr)))

        res = polyfc(L0, KxStar, f, Rtot, LigC, Kav)
        res2 = polyfc2(L0, KxStar, f, Rtot, LigC, Kav)
        res20 = np.sum(res2[0])
        res21 = np.sum(res2[1])

        self.assertTrue(abs(res[0] - res20) < res[0] * 1e-7)
        self.assertTrue(abs(res[1] - res21) < res[1] * 1e-7)
        self.assertTrue(abs(np.sum(res[2]) - res[0]) < res[1] * 1e-3)

    def test_null_monomer(self):
        # [3 0 0] should be equivalent to [3 0 5] if the last ligand has affinity 0
        L0 = np.random.rand() * 10.0 ** np.random.randint(-15, -5)
        KxStar = np.random.rand() * 10.0 ** np.random.randint(-15, -5)
        Rtot = [1e5]
        Kav = [[2e7], [3e5], [0]]

        res11 = polyc(L0, KxStar, Rtot, [[3, 0, 0]], [1], Kav)
        res12 = polyc(L0, KxStar, Rtot, [[3, 0, 5]], [1], Kav)
        res21 = polyc(L0, KxStar, Rtot, [[0, 6, 0]], [1], Kav)
        res22 = polyc(L0, KxStar, Rtot, [[0, 6, 3]], [1], Kav)
        res31 = polyc(L0, KxStar, Rtot, [[2, 4, 0]], [1], Kav)
        res32 = polyc(L0, KxStar, Rtot, [[2, 4, 5]], [1], Kav)

        for i in range(2):
            self.assertEqual(res11[i], res12[i])
            self.assertEqual(res21[i], res22[i])
            self.assertEqual(res31[i], res32[i])

    def test_Lfbnd(self):
        L0 = np.random.rand() * 10.0 ** np.random.randint(-15, -5)
        KxStar = np.random.rand() * 10.0 ** np.random.randint(-15, -5)
        nl = 4
        nr = np.random.randint(1, 10)
        Rtot = np.floor(100.0 + np.random.rand(nr) * (10.0 ** np.random.randint(4, 6, size=nr)))
        Kav = np.random.rand(nl, nr) * (10.0 ** np.random.randint(3, 7, size=(nl, nr)))
        Cplx = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
        Ctheta = np.random.rand(4)
        Ctheta = Ctheta / sum(Ctheta)

        res = polyc(L0, KxStar, Rtot, Cplx, Ctheta, Kav)
        self.assertAlmostEqual(np.sum(res[0]), np.sum(res[2]))
        for i in range(len(res[0])):
            self.assertAlmostEqual(res[0][i], np.sum(res[1], axis=1)[i])


if __name__ == "__main__":
    unittest.main()
