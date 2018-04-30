#! /opt/polarbear/bin/python

from AnalysisBackend.whwp import maprep_whwp
from matplotlib.pyplot import *
import numpy as np
from scipy import ndimage, fftpack

radToDeg = 57.2958
filename = '/scratch/ngoecknerwald/largepatch2/mapmaking/epsilon_fiducial_BICEP/coadd/realmap.hdf5'
m = maprep_whwp.MapMakingVectors.load(filename)
sigma_convolve_arcmin = 5
hpf = 200

#read the patch description from the mapmaker
pixel_size = m.mapinfo.pixel_size
pixel_area = (pixel_size* radToDeg*60)**2
pixel_arcmin = pixel_size* radToDeg*60

#read the information from the hdf5 file
Q = m.mapinfo.view2d(m.Q) *10**6
U = m.mapinfo.view2d(m.U) *10**6
w_QU = m.mapinfo.view2d(m.w4)
#Normalize weights
norm_w_QU = w_QU / np.max(w_QU)

#Convolve with a Gaussian
if sigma_convolve_arcmin is not None:
	Q = ndimage.filters.gaussian_filter(Q, sigma=(sigma_convolve_arcmin / pixel_arcmin))
	U = ndimage.filters.gaussian_filter(U, sigma=(sigma_convolve_arcmin / pixel_arcmin))

#Now, we want to HPF at some certain ell
if hpf is not None:
	kx1 = fftpack.fftfreq(Q.shape[0],d=pixel_size)*2.0*np.pi
	ky1 = fftpack.fftfreq(Q.shape[1],d=pixel_size)*2.0*np.pi
	kx,ky = np.meshgrid(kx1,ky1)
	lmap = np.sqrt(kx**2+ky**2)

	Q_fft = fftpack.fft2(Q*norm_w_QU)
	Q_fft[lmap < hpf] = 0.
	Q_apod = np.real(fftpack.ifft2(Q_fft))

        U_fft = fftpack.fft2(U*norm_w_QU)	
        U_fft[lmap < hpf] = 0.
        U_apod = np.real(fftpack.ifft2(U_fft))
else:
	Q_apod = Q * norm_w_QU
        U_apod = U * norm_w_QU

def clip(array, minthresh=1, maxthresh=99):
	minval = np.nanpercentile(array, minthresh)
	maxval = np.nanpercentile(array, maxthresh)
	return np.minimum(np.maximum(array, minval), maxval)

#Normalize weights
Q_apod[norm_w_QU <= 0.1] = np.nan
U_apod[norm_w_QU <= 0.1] = np.nan

#Map area
half_width = (Q.shape[0] / 2.) *(pixel_size*radToDeg)
map_extent=[-half_width, half_width, -half_width, half_width]

#Define the weight conversion
depth = np.zeros_like(w_QU)
depth[norm_w_QU > 0] = np.sqrt(pixel_area / w_QU[norm_w_QU > 0]) * (10**6)
depth[norm_w_QU <= 0.1] = np.nan

#figure()
#imshow(clip(depth), cmap=cm.jet, extent=map_extent, origin='lower', aspect='equal')
#xlabel('dRA, relative to patch center, $^\circ$')
#ylabel('dDec, relative to patch center, $^\circ$')
#xlim((-30, 30))
#ylim((-20, 20))
#title('%s 4f Map Depth, $\mu K \cdot amin$'%name)
#gca().grid(True)

figure()
imshow(clip(Q_apod), cmap=cm.jet, extent=map_extent, origin='lower', aspect='equal')
xlabel('dRA, relative to patch center, $^\circ$')
ylabel('dDec, relative to patch center, $^\circ$')
xlim((-30, 30))
ylim((-20, 20))
title('3 year apodized Q map, 5\' smoothed, HPF $\ell=200$')
gca().grid(True)

figure()
imshow(clip(U_apod), cmap=cm.jet, extent=map_extent, origin='lower', aspect='equal')
xlabel('dRA, relative to patch center, $^\circ$')
ylabel('dDec, relative to patch center, $^\circ$')
xlim((-30, 30))
ylim((-20, 20))
title('3 year apodized U map, 5\' smoothed, HPF $\ell=200$')
gca().grid(True)

show()
