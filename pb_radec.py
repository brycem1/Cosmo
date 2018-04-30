import astropy
import matplotlib.pyplot as plt
import numpy as np
from IPython import embed
import pickle
import pdb
import healpy as hp
import weave
from astropy.visualization import astropy_mpl_style
plt.style.use(astropy_mpl_style)
from astropy.io import fits
from astropy.utils.data import download_file
from astropy.coordinates import SkyCoord
from healpy.rotator import Rotator
import healpy
from AnalysisBackend.hdf5 import window_hdf5

from scipy import optimize

import AnalysisBackend.quaternionarray as qa
from AnalysisBackend.pointing.quat_pointing import euler_quatz, euler_quatx, euler_quaty
from AnalysisBackend.mapping import projection, flatmap

import astropy
import matplotlib.pyplot as plt
from astropy.visualization import astropy_mpl_style
plt.style.use(astropy_mpl_style)





from astropy.coordinates import SkyCoord
from healpy.rotator import Rotator



def root_finder(array, value):
    index = np.argmin(abs(array-value))
    return index

def double_root_finder(array1, array2, value1, value2, tol):
    indexes = []
    for i in range(len(array1)):
	if (abs(array1[i]-value1) < tol) & (abs(array2[i]-value2) < tol):
	    indexes.append(i)
    indexes = np.array(indexes)
    return indexes

def offset_radecpa_makequat(ra,dec,racen,deccen):
	# Can eliminate temporaries here
	qra = euler_quatz(ra)

	qdec = euler_quaty(-dec)
	#qpa = euler_quatx(-pa)

	qracen = euler_quatz(-racen)
	qdeccen = euler_quaty(deccen)

	
	q = qa.mult(qra,qdec)
	q = qa.mult(qracen,q)
	q = qa.mult(qdeccen,q)

	return q

def quat_to_radecpa_python(seq):
	q1,q2,q3,q0 = seq.T
	phi = np.arctan2(2*(q0*q1+q2*q3),1-2*(q1*q1+q2*q2))
	theta = np.arcsin(2*(q0*q2-q3*q1))
	psi = np.arctan2(2*(q0*q3+q1*q2),1-2*(q2*q2+q3*q3))
	return phi,theta,psi	    


radec = pickle.load(open("radec.pkl","rb"))
ra = radec['ra']
dec = radec['dec']

ra_rad = np.deg2rad(ra)
dec_rad = np.deg2rad(dec)
'''
bicep_ra = root_finder(ra, 0)
bicep_dec = root_finder(dec, -57.5)
'''
'''
radec_ind = double_root_finder(ra_rad, dec_rad, np.deg2rad(0.0), np.deg2rad(-57.5), np.deg2rad(45))

ras = ra[radec_ind]
decs = dec[radec_ind]
'''
hope = offset_radecpa_makequat(ra_rad,dec_rad,np.deg2rad(0.0),np.deg2rad(-57.5))

phi, theta, psi = quat_to_radecpa_python(hope)

pb_ra = phi
pb_dec = -theta
pb_pa = -psi

l_ra, l_dec = projection.LamCyl(pb_ra,pb_dec)

fmap = flatmap.FlatMapInfo(xmin = -0.6981317007977318, xmax = 0.6981317007977318, ymin = -0.6981317007977318, ymax = 0.6981317007977318, pixel_size = 0.0005817764173314432)
vec = fmap.mapindex1d(l_ra,l_dec)

fits_file = fits.open('./COM_CMB_IQU-smica_1024_R2.02_full.fits')
#mapp = healpy.fitsfunc.read_map(fits_file[1], field = (0,1,2,3,4,5))
#hope = healpy.sphtfunc.map2alm(mapp)
#m143 = healpy.fitsfunc.read_map(fits_file[1], field = (0,1,2))
I,Q,U,tmask,pmask = healpy.fitsfunc.read_map(fits_file[1], field = (0,1,2,3,4), nest = True)

I = I*10**6
Q = Q*10**6
U = U*10**6


embed()
'''
radec_1024 = { 'ra' : ra_f, 'dec' : dec_f }
with open("radec.pkl", "wb") as infile:
    pickle.dump(radec_1024, infile) 

'''

