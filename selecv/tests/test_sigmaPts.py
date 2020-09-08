import unittest
import numpy as np
from selecv.sampling import cellPopulations
from selecv.figures.figure6 import genOnevsAll, minSelecFunc, minSigmaVar

class TestSigmaPts(unittest.TestCase):
    def test_hAsZero(self):
        for targetPop in cellPopulations.keys():
            targPops, offTargPops = genOnevsAll([targetPop])

            x = np.array([np.log(1e-9), np.log(1e-13), 5, 0.8, np.log(1e7), np.log(1e5)])
            sel1 = minSelecFunc(x, targPops, offTargPops)
            sel2 = minSigmaVar(x, targPops, offTargPops, h=0)

            self.assertAlmostEqual(sel1, sel2)
