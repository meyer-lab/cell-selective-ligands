#!/usr/bin/env python3

from selecv.figures.figureCommon import overlayCartoon
import sys
from logging import basicConfig, INFO, info
from time import time
import matplotlib
matplotlib.use('AGG')

fdir = './output/'

basicConfig(format='%(levelname)s:%(message)s', level=INFO)

if __name__ == '__main__':
    start = time()
    nameOut = 'figure' + sys.argv[1]

    exec('from selecv.figures import ' + nameOut)
    ff = eval(nameOut + '.makeFigure()')
    ff.savefig(fdir + nameOut + '.svg', dpi=ff.dpi, bbox_inches='tight', pad_inches=0)

    info('%s is done after %s seconds.', nameOut, time() - start)
