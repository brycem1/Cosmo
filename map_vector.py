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
import weave
import scipy

#testing only
from AnalysisBackend.mapping import wiener_filter
from AnalysisBackend.pointsource import at20g
from AnalysisBackend.powerspectrum import psestimator
from AnalysisBackend.mapping import polwindow
from AnalysisBackend.mapping.flatmap import FlatMapInfo

from AnalysisBackend.hdf5 import window_hdf5
from AnalysisBackend.hdf5.window_hdf5 import denan,get_iqu,pointsource_mask,get_cross_weight,standard_temp_mask,standard_pol_mask
#testing libraries
from scipy import ndimage, fftpack

def get_ell_array(n,pixel_size):
	'''Calculates and returns a map with all the rotation angles needed to go from Q/U to E/B in flat sky case
	
	@type n: int
	@param n: pixel width of map
	@type pixel_size: float
	@param pixel_size:  width of pixel in radians
	@rtype: (numpy.ndarray,numpy.ndarray)
	@return: Values of ell in each pixel for the fourier transformed maps along with the angles for each 2d-frequency
	'''
	u_freq = np.fft.fftfreq(n,pixel_size)
	u,v = np.meshgrid(u_freq,u_freq)
	ell = 2*np.pi * np.sqrt(u**2 + v**2)
	chi = np.arctan2(v,u)
	
	return ell,chi

def normalization_fft_mask(mask):
	'''Calculcates the normalization needed for Fourier space values. Normalization factors for both the pixel numbers and 
	the mask are calculated.
	
	@type mask: numpy.ndarray
	@param mask: mask that we use to calculate normalizations
	@type: float
	@return: normalizaton value for Fourier space'''

	fftnorm = 1.0 / np.prod(mask.shape) #this is due to the definition of the FFT in numpy
	masknorm = np.sqrt(1.0 / np.mean(mask*mask)) #this keeps everything the same amplitude if we zero-pad the maps (and mask is constant)

	return fftnorm*masknorm

def extend_mask_fft(in_mask,extend_ratio=1.3):
	'''
	Zero pad a mask or a map to eliminate the periodic boundary condition
	Roughly center the old map in the new map
	Use zeropad module to pick an efficient size for FFTs that is at least extend_ratio larger than input
	'''
	n = in_mask.shape[0]
	nover = zeropad.pick_full_size(n,extend_ratio=extend_ratio)
	return extend_mask(in_mask, nover=nover)

def calc_window_derivs(w,pixel_size,do_fft=False):
	'''Calculate 1st and 2nd derivates of the window function. Uses finite differencing. Assumes window is 0 at edge of the map so 
	that when I use np.roll it doesn't screw up derivatives at the edges (they should be 0)
	
	@type w: numpy.ndarray
	@param w: window function that we want the derivates for
	@type pixel_size: float
	@param pixel_size: the size of the pixel along an edge in radians
	@rtype: numpy.ndarray
	@return: The 1st and 2nd derivatives of the window function
	'''
	
	if do_fft:
		print "Calculating window derivatives with FFT rather than by finite differences. See calc_window_derivs in powerspectrum/psestimator.py"
		#derivatives just bring down a factor of i*k_j in Fourier space
		n = w.shape[0]
		w_fft = np.fft.fft2(w)
		ell,chi = get_ell_array(n,pixel_size)
		ell_x = ell*np.cos(chi)
		ell_y = ell*np.sin(chi)
		dwdy = np.fft.ifft2(w_fft*1j*ell_y).real
		dwdx = np.fft.ifft2(w_fft*1j*ell_x).real
		d2wdy2 = np.fft.ifft2(-w_fft*ell_y**2).real
		d2wdx2 = np.fft.ifft2(-w_fft*ell_x**2).real
		d2wdxdy = np.fft.ifft2(-w_fft*ell_x*ell_y).real
	else:
		print "Calculating window derivatives by finite differences rathter than with FFT. See calc_window_derivs in powerspectrum/psestimator.py"
		#maps have square pixels with size pixel_size so space between pixels is pixel_size
		dx = pixel_size
		dy = pixel_size

		axisx = 1
		axisy = 0

		#Use finite differencing formulas to calculate derivatives
		#Uses the central differencing formulas
		#which axis is x and which axis is y must be the same as the calculation of chi (phi_l), cos(chi) is az direction
		#so x is az direction (which happens to be axis=1 in this code) (i.e. chi[0,:] = 0.0 or pi)
		dwdy = (np.roll(w,-1,axis=axisy)-np.roll(w,1,axis=axisy)) / (2*dy)
		dwdx = (np.roll(w,-1,axis=axisx)-np.roll(w,1,axis=axisx)) / (2*dx)
		d2wdy2 = (np.roll(w,-1,axis=axisy) - 2*w + np.roll(w,1,axis=axisy)) / dy**2
		d2wdx2 = (np.roll(w,-1,axis=axisx) - 2*w + np.roll(w,1,axis=axisx)) / dx**2
		d2wdxdy = (np.roll(dwdx,-1,axis=axisy) - np.roll(dwdx,1,axis=axisy)) / (2*dy)

	return dwdx,dwdy,d2wdx2,d2wdy2,d2wdxdy


def qu2eb(Q,U,mask,pixel_size,pure=True,window_derivs=None,trig=None):
	'''Q/U to E/B. This will add the counter terms for the estimator from Kendrick Smith if requested
	
	@type Q: numpy.ndarray
	@param Q: Q map
	@type U: numpy.ndarray
	@param U: U map
	@type mask: numpy.ndarray
	@param mask: weights for each pixel in the map. The window function
	@type pure: boolean, optional
	@param pure: whether to calculate the pure B multipoles or not
	@type window_derivs: numpy.ndarray
	@param window_derivs: Precomputed window_derivs from calc_window_derivs() if available
	@type trig: numpy.ndarray
	@param trig: Precomputed c2chi,s2chi,c1chi,s1chi if available
	@rtype: numpy.ndarray
	@return: The E/B multipole maps
	'''

	'''
	from AnalysisBackend.misc import util

	fn_map = 'sim_tteebb/simmap.npy'
	fn_desc = 'sim_tteebb/simdesc.pkl'

	mapdesc = util.pickle_load(fn_desc)
	simmaps = np.load(fn_map)

	n = mapdesc['n']
	pixel_size = mapdesc['pixel_size']
	I,Q,U = simmaps[0,:,:]

	I = I.reshape((n,n))
	Q = Q.reshape((n,n))
	U = U.reshape((n,n))
	
	mask = np.ones_like(Q)
	'''

	assert Q.shape[0] == Q.shape[1]
	n = Q.shape[0]
	ell,chi = get_ell_array(n,pixel_size)
	#Get E spectrum
	Qmap_ft = np.fft.fft2(Q*mask)
	Umap_ft = np.fft.fft2(U*mask)

	if not trig:
		trig = np.cos(2*chi),np.sin(2*chi),np.cos(chi),np.sin(chi)
	c2chi,s2chi,cchi,schi = trig
	#rotate Q/U into E/B
	Emap_ft = Qmap_ft*c2chi + Umap_ft*s2chi
	Bmap_ft = -s2chi*Qmap_ft + c2chi*Umap_ft
	
	#If calculating pure multipoles, then calculate the counter terms and add them
	if pure:
		#Get the various derivatives of the window
		if not window_derivs:
			window_derivs = calc_window_derivs(mask,pixel_size)
		dWdx,dWdy,d2Wdx2,d2Wdy2,d2Wdxdy = window_derivs
		#make lots of ffts of Q/U times window and its derivs
		QdWdx = np.fft.fft2(Q*dWdx)
		QdWdy = np.fft.fft2(Q*dWdy)
		UdWdy = np.fft.fft2(U*dWdy)
		UdWdx = np.fft.fft2(U*dWdx)

		#don't divide by zero. Just cancel out contribution from this term by making it extremely large
		ell[ell == 0] = 1e10 

		#Calculate the counter term and apply it to Bmap_ft
		#a bunch of algebra has reduced to these simple terms
		counter_term = - (2.0j / ell) * (schi*QdWdx + cchi*QdWdy + schi*UdWdy - cchi*UdWdx)
		counter_term += (1.0 / ell**2) * np.fft.fft2(2*Q*d2Wdxdy + U*(d2Wdy2-d2Wdx2))

		Bmap_ft += counter_term

	norm = normalization_fft_mask(mask)

	return Emap_ft*norm,Bmap_ft*norm

def get_temp_weight(m):
	if hasattr(m,'w0'):
		tweight = m.mapinfo.view2d(m.w0)
	else:
		tweight = m.mapinfo.view2d(m.w)
	denan(tweight)
	return tweight

def get_pol_weight(m):
	if hasattr(m,'w4'):
		pweight = m.mapinfo.view2d(m.w4)
	else:
		cc = m.mapinfo.view2d(m.cc)
		cs = m.mapinfo.view2d(m.cs)
		ss = m.mapinfo.view2d(m.ss)
		cn = m.condnum()
		denan(cn)
		cn = m.mapinfo.view2d(cn)

		ok = cn < 10

		pweight = polwindow.qu_weight_mineig(cc,cs,ss)
		pweight *= ok
	denan(pweight)
	return pweight

'''
def standard_temp_mask(tmap,tweight,ps=None):
	denan(tweight)
#this is fine
	mask = wiener_temp(tweight)
	if ps is None:
		ps = pointsource_mask(tmap)
	mask *= ps
	mask = psestimator.prepare_mask(mask)
	return mask
'''

def get_maskweight_whwp(m,ps=None,tpweights=None):
	tweight = get_temp_weight(m)
	pweight = get_pol_weight(m)

#	m.mapinfo.source = source
	m.mapinfo.I_weight = tweight

	if ps is None:
		m.mapinfo.I_weight = tweight
		ps = pointsource_mask(m)


	if tpweights:
		tweight1,pweight1 = tpweights
		tweight = get_cross_weight(tweight,tweight1)
		pweight = get_cross_weight(pweight,pweight1)
		m.mapinfo.I_weight = tweight
	
	#somewhere here, it goes from (2401,2401) to (3456,3456)
	tmask = standard_temp_mask(m,tweight,ps)
	pmask = standard_pol_mask(m,pweight,ps)

	tweight1 = np.sum(tweight)
	pweight1 = np.sum(pweight)

	return tmask,pmask,tweight1,pweight1

def pol_mask(pmap,mask):
    pixel_size = pmap.mapinfo.pixel_size
    edge_taper = 1.0*(np.pi/180.0)	
    pixel_taper = int(edge_taper / pixel_size)
    ps = pointsource_mask(pmap)
    mask = psestimator.prepare_polmask(mask,pixel_taper)
    # PS mask already uses the nice taper, so apply it later
    mask *= ps
    mask = psestimator.extend_mask_fft(mask)
    return mask

def auto_spectra(mapf,pixel_size,masks,bins,weights):
	n = mapf.shape[0]
	l,chi = get_ell_array(n,pixel_size)
	ak_to_dl = psestimator.fourier_calibrate(l,n,pixel_size)

		
	psd = (mapf*np.conj(mapf)).real
	psd *= ak_to_dl
	
	dl = psestimator.binpsd(l,psd,bins)

	return dl


m = maprep.MapMakingVectors.load('realmap.hdf5')
mm = m.mapinfo

#pdb.set_trace()
tmask, pmask, tweight, pweight = get_maskweight_whwp(m)
'''
window_hdf5.get_temp_weight = get_temp_weight
window_hdf5.get_pol_weight = get_pol_weight
window_hdf5.get_maskweight = get_maskweight_whwp
'''
#mm.pixel_size = 0.0005817764173314432



I = mm.view2d(m.I)*10**6
Q = mm.view2d(m.Q)*10**6
U = mm.view2d(m.U)*10**6
I = psestimator.extend_mask_fft(I)
Q = psestimator.extend_mask_fft(Q)
U = psestimator.extend_mask_fft(U)

#Q_gauss = scipy.ndimage.filters.gaussian_filter(Q,sigma=(5.0/(mm.pixel_size*57.2958*60)))
#U_gauss = scipy.ndimage.filters.gaussian_filter(U,sigma=(5.0/(mm.pixel_size*57.2958*60)))

w0 = mm.view2d(m.w0)
w4 = mm.view2d(m.w4)

ell, chi = psestimator.get_ell_array(3456,mm.pixel_size)

X = 0.7E10

Y_test = np.arange(18)*0.05E10
'''
Y = 0.6*X
array = w4>X

array1 = 0.9*X<w4
array2 = 0.8*X<w4
array3 = 0.7*X<w4
array4 = 0.6*X<w4
array5 = 0.5*X<w4
test = w4[array]
test1 = w4[array1]
test2 = w4[array2]
test3 = w4[array3]
test4 = w4[array4]
test5 = w4[array5]

var_X = np.var(test)
var_X1 = np.var(test1)
var_X2 = np.var(test2)
var_X3 = np.var(test3)
var_X4 = np.var(test4)
var_X5 = np.var(test5)
'''
'''
mod_arc = np.zeros((2401,2401))
for i in range(2401):
    for j in range(2401):
        #convert index to arcminute scale with pixel size 0.5 and flat map from -150'<=x,y<=150'
        arcx = (i -1200)*2.0
        arcy = (j -1200)*2.0
        mod_arc = np.sqrt(arcx**2+arcy**2)
        #if mod_arc <30:
        #    mask[j][i] = 1
	#else:
	#    mask[j][i] = exp(-((mod_arc-30)/5)**2)
'''	
'''
mask = np.ones((2401,2401))
mask1 = np.ones((2401,2401))
mask2 = np.ones((2401,2401))
mask3 = np.ones((2401,2401))
mask4 = np.ones((2401,2401))
mask5 = np.ones((2401,2401))
mask6 = np.ones((2401,2401))
mask7 = np.ones((2401,2401))
mask8 = np.ones((2401,2401))
mask9 = np.ones((2401,2401))
mask10 = np.ones((2401,2401))
mask11 = np.ones((2401,2401))
mask12 = np.ones((2401,2401))
mask13 = np.ones((2401,2401))
mask14 = np.ones((2401,2401))
mask15 = np.ones((2401,2401))
mask16 = np.ones((2401,2401))
mask17 = np.ones((2401,2401))
#dist = psestimator.boundary_distance(mask)
for i in range(2401):
    for j in range(2401):
	if w4[i,j]<Y:
	    mask[i,j] = w4[i,j]/Y[0]
	    mask1[i,j] = w4[i,j]/Y[1]
	    mask2[i,j] = w4[i,j]/Y[2]
            mask3[i,j] = w4[i,j]/Y[3]
	    mask4[i,j] = w4[i,j]/Y[4]
	    mask5[i,j] = w4[i,j]/Y[5]
	    mask6[i,j] = w4[i,j]/Y[6]
            mask7[i,j] = w4[i,j]/Y[7]
	    mask8[i,j] = w4[i,j]/Y[8]
	    mask9[i,j] = w4[i,j]/Y[9]
	    mask10[i,j] = w4[i,j]/Y[10]
            mask11[i,j] = w4[i,j]/Y[11]
	    mask12[i,j] = w4[i,j]/Y[12]
	    mask13[i,j] = w4[i,j]/Y[13]
	    mask14[i,j] = w4[i,j]/Y[14]
            mask15[i,j] = w4[i,j]/Y[15]
	    mask16[i,j] = w4[i,j]/Y[16]
            mask17[i,j] = w4[i,j]/Y[17]
            #mask1[i,j] = (1.0/2.0)*(1.0+np.cos(distance/(1.0*np.pi/180)))	
#dist = psestimator.boundary_distance(mask)           
#hope = pol_mask(m,mask)

#I_norm = normalization_fft_mask(tmask)
#I_fft = np.fft.fft2(I*tmask)*I_norm
E_fft,B_fft = qu2eb(Q,U,mask,mm.pixel_size)
E_fft1,B_fft1 = qu2eb(Q,U,mask1,mm.pixel_size)
E_fft2,B_fft2 = qu2eb(Q,U,mask2,mm.pixel_size)
E_fft3,B_fft3 = qu2eb(Q,U,mask3,mm.pixel_size)
E_fft4,B_fft4 = qu2eb(Q,U,mask4,mm.pixel_size)
E_fft5,B_fft5 = qu2eb(Q,U,mask5,mm.pixel_size)
E_fft6,B_fft6 = qu2eb(Q,U,mask6,mm.pixel_size)
E_fft7,B_fft7 = qu2eb(Q,U,mask7,mm.pixel_size)
E_fft8,B_fft8 = qu2eb(Q,U,mask8,mm.pixel_size)
E_fft9,B_fft9 = qu2eb(Q,U,mask9,mm.pixel_size)
E_fft10,B_fft10 = qu2eb(Q,U,mask10,mm.pixel_size)
E_fft11,B_fft11 = qu2eb(Q,U,mask11,mm.pixel_size)
E_fft12,B_fft12 = qu2eb(Q,U,mask12,mm.pixel_size)
E_fft13,B_fft13 = qu2eb(Q,U,mask13,mm.pixel_size)
E_fft14,B_fft14 = qu2eb(Q,U,mask14,mm.pixel_size)
E_fft15,B_fft15 = qu2eb(Q,U,mask15,mm.pixel_size)
E_fft16,B_fft16 = qu2eb(Q,U,mask16,mm.pixel_size)
E_fft17,B_fft17 = qu2eb(Q,U,mask17,mm.pixel_size)

E_norm = 1.0/np.prod(Q.shape)
E = np.fft.ifft2(E_fft)/E_norm
'''
E_fft,B_fft = qu2eb(Q,U,pmask,mm.pixel_size)
#White noise at mu = 0.0;sigma = 1.0
'''
mu = 0.0;sigma = 1.0
noise = sigma*np.random.randn(3456,3456)+mu


E_gauss_real = scipy.ndimage.filters.gaussian_filter(np.real(E),sigma = (5/(mm.pixel_size*57.2958*60)))
E_gauss_imag = scipy.ndimage.filters.gaussian_filter(np.imag(E),sigma = (5/(mm.pixel_size*57.2958*60)))
E_gauss = (E_gauss_real+E_gauss_imag*1j)

out = E_gauss + noise

out_fft = np.fft.fft2(out)*E_norm
'''	
'''
norm_w_I = w0/np.max(w0)
norm_w_QU = w4/np.max(w4)

radToDeg = 57.2958
pixel_size = m.mapinfo.pixel_size
pixel_area = (pixel_size* radToDeg*60)**2
pixel_arcmin = pixel_size* radToDeg*60
sigma_convolve_arcmin = 5
'''
'''
if sigma_convolve_arcmin is not None:
        I = ndimage.filters.gaussian_filter(I, sigma=(sigma_convolve_arcmin / pixel_arcmin))
	Q = ndimage.filters.gaussian_filter(Q, sigma=(sigma_convolve_arcmin / pixel_arcmin))
	U = ndimage.filters.gaussian_filter(U, sigma=(sigma_convolve_arcmin / pixel_arcmin))

norm_fft = 1.0/(2401**2)
I_fft = fftpack.fft2(I)*norm_fft
Q_fft = fftpack.fft2(Q)*norm_fft
U_fft = fftpack.fft2(U)*norm_fft
'''


'''
#pow_I = np.abs(I_fft)**2
pow_E = np.abs(E_fft)**2
pow_E1 = np.abs(E_fft1)**2
ell_min = np.arange(120)*25
ell_max = np.arange(120)*25+25
ell_cen = 0.5*(ell_min+ell_max)
n_modes = len(ell_cen)

#Cl_TT = np.zeros(n_modes)
Cl_EE = np.zeros(n_modes)
Cl_EE1 = np.zeros(n_modes)

for i in range(n_modes):
    #Cl_TT[i] = np.mean(pow_I[np.logical_and(ell >= ell_min[i],ell < ell_max[i])])
    Cl_EE[i] = np.mean(pow_E[np.logical_and(ell >= ell_min[i],ell < ell_max[i])])
    Cl_EE1[i] = np.mean(pow_E1[np.logical_and(ell >= ell_min[i],ell < ell_max[i])])
'''
'''
rms_EE = np.sqrt(np.mean(Cl_EE**2))
3.5612415469573124e-05

test = np.zeros((2401,2401))

test=1.0/w4
test[w4 == 0]=0
np.std(test)

rms_noise = np.sqrt(np.nanmean(test**2))
3.0774357997769571e-07
'''

'''
array1 = np.logical_and(0.9*X<w4,w4<X)
array2 = np.logical_and(0.8*X<w4,w4<0.9*X)
array3 = np.logical_and(0.7*X<w4,w4<0.8*X)
array4 = np.logical_and(0.6*X<w4,w4<0.7*X)
array5 = np.logical_and(0.5*X<w4,w4<0.6*X)

X = 0.6E10
array = w4>X

array1 = 0.9*X<w4
array2 = 0.8*X<w4
array3 = 0.7*X<w4
array4 = 0.6*X<w4
array5 = 0.5*X<w4
test = w4[array]
test1 = w4[array1]
test2 = w4[array2]
test3 = w4[array3]
test4 = w4[array4]
test5 = w4[array5]

var_X = np.var(test)
var_X1 = np.var(test1)
var_X2 = np.var(test2)
var_X3 = np.var(test3)
var_X4 = np.var(test4)
var_X5 = np.var(test5)

Y = 0.6*X

mask = np.ones((2401,2401))
for i in range(2401):
    for j in range(2401):
	if w4[i,j]<Y:
	    mask[i,j] = w4[i,j]/Y

'''	

#var_E = np.var(E)
bins = psestimator.binspec(config='dl=20')
embed()
Dl = auto_spectra(E_fft, mm.pixel_size, pmask, bins , w4)
#Dl_TT = ell_cen*(ell_cen+1)*Cl_TT/(2*np.pi)
#Dl_EE = ell_cen*(ell_cen+1)*Cl_EE/(2*np.pi)
#Dl_EE1 = ell_cen*(ell_cen+1)*Cl_EE1/(2*np.pi)

plt.figure(1)
#plt.plot(ell_cen,Dl_TT)
#plt.xlabel('ell')
#plt.ylabel('Dl_TT')
#plt.figure(2)
plt.plot(bins[:,0],Dl,label = 'dl=20 from l = 0')
plt.plot(bins[:,1],Dl,label = 'dl=20 from l = 10')
plt.xlabel('ell')
plt.ylabel('Dl_EE')
plt.legend(title = 'dl = 20', loc = 'lower right')
#plt.figure(2)
#plt.plot(ell_cen,Dl_EE1)
#plt.xlabel('ell')
#plt.ylabel('Dl_EE1')
plt.show()

'''
ft_I = fft.fft2(I)
mod_ft_I_sq = abs(ft_I)**2
#ft_I.shape = (2401, 2401)
freq = fft.fftfreq(2401)
freq = 2*np.pi*freq

ell = np.zeros((2401,2401))
for i in range(2401):
    for j in range(2401):
	ell[i][j]=np.sqrt(freq[i]**2+freq[j]**2)
#After doing the fftfreq, this takes into account the pixel size
ell = ell/0.0005817764173314432
ell_min = np.arange(200)*30
ell_max = np.arange(200)*30+30
ell_cen = 0.5*(ell_min+ell_max)
n_modes = len(ell_cen)

mod_ft_I_sq_bin = np.zeros(n_modes)
for i in range(n_modes):
    mod_ft_I_sq_bin[i] = np.mean(mod_ft_I_sq[np.logical_and(ell >= ell_min[i],ell < ell_max[i])])*((ell_cen[i]*ell_cen[i]+1)/(2*np.pi))
#embed()

#I don't think this is right as for Planck got an array of C_l, so I think there should be a list of k values per bin
d_b = np.sum(mod_ft_I_sq_bin)/n_modes
'''
embed()
