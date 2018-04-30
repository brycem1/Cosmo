import pickle
import numpy as np
import matplotlib as plt
import pylab
from IPython import embed

cm = pickle.load(open("coadded_maps.pkl","rb"))
embed()
cm['all'].make_plot(cm['all'].I)
pylab.show()

