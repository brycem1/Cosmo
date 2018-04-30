import astropy
import matplotlib.pyplot as plt
import numpy as np
from IPython import embed
import pickle
import pdb
import healpy as hp

#data = whatever data goes here (Planck)

'''
This out map specifically is going from Planck to PB
#out = np.zeros(flatsky)

Planck Pix keeps track of what has been hit
#hitout = np.zeros(hitmap)

right in essence we have something like

<.> = (1/N)*np.sum(X)

Where X is the associated planck map value for that pixel region?
and N is the number of hits for each pixel to Polar bear.

More specifically 

for some Ra Dec on the healpy map---> We know it's PB PIX as HP PIX (For now using the same specs as the large patch map)

From that we set

X ----> out[PBPIX#] += data[HPPIX#]
N ----> hitout[HPPIX#] += 1

This is to avoid divide zero errors

out = out/hitout | hitout > 0 





