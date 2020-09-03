import unittest
from selecv.figures.figure6 import *

class TestPolyc(unittest.TestCase):
    def test_equivalence(self):
        pass

        targPops, offTargPops = genOnevsAll(targetPop)
        targMeans, offTargMeans = genPopMeans(targPops), genPopMeans(offTargPops)

        minSelecFunc(x, tPops, offTPops)
        minSigmaVar(x, tPops, offTPops)

        self.assertEqual(res11[i], res12[i])