import astropy
import matplotlib.pyplot as plt
from scipy import optimize
import numpy as np
from IPython import embed
import numpy.fft as fft
import pickle
import pdb
import healpy as hp
import h5py
import AnalysisBackend.whwp.maprep_whwp as maprep

m = maprep.MapMakingVectors.load('realmap.hdf5')
mm = m.mapinfo

mplk===

mplk_1sthalf
mplk_2ndhal

#mm.pixel_size = 0.0005817764173314432


I = mm.view2d(m.I)
ft_I = fft.fft2(I)
ft_plk = fft.fft2(mplk)

mod_ft_I_sq = abs(ft_I)**2
mod_ft_Icross = abs(ft_I*ft__plk**2

                #read beams
                    bplk
                    bpb

#ft_I.shape = (2401, 2401)
freq = fft.fftfreq(2401)
freq = 2*np.pi*freq

ell = np.zeros((2401,2401))
for i in range(2401):
    for j in range(2401):
	ell[i][j]=np.sqrt(freq[i]**2+freq[j]**2)
dl2d = mod_ft_i_sq * ell * (ell+1)/(2*np.pi)
#After doing the fftfreq, this takes into account the pixel size
ell = ell/0.0005817764173314432
ell_min = np.arange(200)*30
ell_max = np.arange(200)*30+30
ell_cen = 0.5*(ell_min+ell_max)
n_modes = len(ell_cen)

top_bin = np.zeros(n_modes)
                    bot_bin = np.zeros(n_modes)
                    
for i in range(n_modes):
    bot_bin[i] = np.mean(dl2d_bot[np.logical_and(ell >= ell_min[i],ell < ell_max[i])] * beam stuff)
                    top_bin[i] = np.mean(dl2d_top[np.logical_and(ell >= ell_min[i],ell < ell_max[i])])
#embed()

                ratios = top_bin / bot_bin
#I don't think this is right as for Planck got an array of C_l, so I think there should be a list of k values per bin
d_b = np.sum(mod_ft_I_sq_bin)/n_modes
embed()
