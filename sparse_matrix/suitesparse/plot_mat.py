import numpy as np
import scipy.sparse as sp
import pyfancyplot.plot as plot
from scipy.io import mmread
import glob

files = glob.glob("*.mtx")
for filename in files:
    A = mmread(filename)
    plot.plt.spy(A, markersize=.3, rasterized=True)
    plot.set_xticks([0, A.shape[1]],[0, A.shape[1]])
    plot.set_yticks([0, A.shape[0]],[0, A.shape[0]])
    plot.save_plot("%s.pdf"%filename)
