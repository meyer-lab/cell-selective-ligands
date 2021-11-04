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

    if sys.argv[1] == '1':
        # Overlay Figure 1 cartoon
        overlayCartoon(fdir + 'figure1.svg',
                       './selecv/graphics/figure_1a.svg', 10, 15, scalee=0.02, scale_x=0.45, scale_y=0.45)
        overlayCartoon(fdir + 'figure1.svg',
                       './selecv/graphics/figure_1b.svg', 5, 280, scalee=0.12, scale_x=0.3, scale_y=0.3)
    if sys.argv[1] == '3':
        overlayCartoon(fdir + 'figure3.svg',
                       './selecv/graphics/figure_3i.svg', 480, 490, scalee=0.15, scale_x=1, scale_y=1)

    info('%s is done after %s seconds.', nameOut, time() - start)
