import pytest
import numpy as np
from selecv.sampling import cellPopulations
from selecv.figures.figure6 import genOnevsAll, minSelecFunc, minSigmaVar


@pytest.mark.parametrize("targetPop", cellPopulations.keys())
def test_hAsZero(targetPop):
    targPops, offTargPops = genOnevsAll([targetPop])
    x = np.array([np.log(1e-9), np.log(1e-13), 5, 0.8, np.log(1e7), np.log(1e5), np.log(1e7), np.log(1e5)])
    sel1 = minSelecFunc(x, targPops, offTargPops)
    sel2 = minSigmaVar(x, targPops, offTargPops, h=0)
    np.testing.assert_almost_equal(sel1, sel2)
