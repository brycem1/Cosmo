import astropy
import matplotlib.pyplot as plt
import numpy as np
from IPython import embed
import pickle
import pdb
import healpy as hp

fname = './Planck2015_base_plikHM_TT_lowTEB_lensing_r0p025_lensedtotCls.dat'

#returns C_l's from l(l+1)C^XY/(2pi) [micro-K^2]
ell, TT, EE, BB, TE = np.loadtxt(fname, unpack = 'True')


embed()
