import pickle	
import numpy as np
import matplotlib.pyplot as plt
import pylab
from IPython import embed
from scipy import fftpack
from AnalysisBackend.powerspectrum import psestimator

def view2d(flatvec,must_share=True):
		'''
		Return a 2d view into a 1d flat map vector.
		The view can be modified to modify the flat vector if the vector needs to be manipulated in 2d.
		First axis indexes x and second axis indexes y like you expect for an imshow picture.
		The guard pixel is not included in the view.
		'''
		map2d = flatvec.reshape((cm['n'],cm['n'])).T
		if must_share: assert np.may_share_memory(flatvec,map2d)
		return map2d
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

cm = pickle.load(open("fl_simmap_noBB/simdesc.pkl","rb"))
cm0 = pickle.load(open("fl_simmap_noBB/sim000/simdesc.pkl","rb"))
cm1 = pickle.load(open("fl_simmap_noBB/sim001/simdesc.pkl","rb"))
cm2 = pickle.load(open("fl_simmap_noBB/sim002/simdesc.pkl","rb"))
cm3 = pickle.load(open("fl_simmap_noBB/sim003/simdesc.pkl","rb"))
cm4 = pickle.load(open("fl_simmap_noBB/sim004/simdesc.pkl","rb"))
cm5 = pickle.load(open("fl_simmap_noBB/sim005/simdesc.pkl","rb"))
cm6 = pickle.load(open("fl_simmap_noBB/sim006/simdesc.pkl","rb"))
cm7 = pickle.load(open("fl_simmap_noBB/sim007/simdesc.pkl","rb"))

map0 = np.load('fl_simmap_noBB/sim000/simmap.npy')
map0 = map0[0]
map1 = np.load('fl_simmap_noBB/sim001/simmap.npy')
map1 = map1[0]
map2 = np.load('fl_simmap_noBB/sim002/simmap.npy')
map2 = map2[0]
map3 = np.load('fl_simmap_noBB/sim003/simmap.npy')
map3 = map3[0]
map4 = np.load('fl_simmap_noBB/sim004/simmap.npy')
map4 = map4[0]
map5 = np.load('fl_simmap_noBB/sim005/simmap.npy')
map5 = map5[0]
map6 = np.load('fl_simmap_noBB/sim006/simmap.npy')
map6 = map6[0]
map7 = np.load('fl_simmap_noBB/sim007/simmap.npy')
map7 = map7[0]

I0 = view2d(map0[0])
Q0 = view2d(map0[1])
U0 = view2d(map0[2])
I1 = view2d(map1[0])
Q1 = view2d(map1[1])
U1 = view2d(map1[2])
I2 = view2d(map2[0])
Q2 = view2d(map2[1])
U2 = view2d(map2[2])
I3 = view2d(map3[0])
Q3 = view2d(map3[1])
U3 = view2d(map3[2])
I4 = view2d(map4[0])
Q4 = view2d(map4[1])
U4 = view2d(map4[2])
I5 = view2d(map5[0])
Q5 = view2d(map5[1])
U5 = view2d(map5[2])
I6 = view2d(map6[0])
Q6 = view2d(map6[1])
U6 = view2d(map6[2])
I7 = view2d(map7[0])
Q7 = view2d(map7[1])
U7 = view2d(map7[2])
print("View 2d done!")

def qu2eb(Q,U,pixel_size,mask=None,pure=False,window_derivs=None,trig=None):
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
	if mask == None:
    		mask = np.ones((Q.shape[0],Q.shape[1]))


	assert Q.shape[0] == Q.shape[1]
	n = Q.shape[0]
	ell,chi = psestimator.get_ell_array(n,pixel_size)
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
			window_derivs = psestimator.calc_window_derivs(mask,pixel_size)
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

	norm = psestimator.normalization_fft_mask(mask)

	return Emap_ft*norm,Bmap_ft*norm

norm_fft = (1.0)/(cm['n']**2)

#fft_I0 = np.fft.fft2(I0)
#fft_Q0 = np.fft.fft2(Q0/2)
print("fft0 done!")
#fft_U0 = np.fft.fft2(U0)
#fft_I1 = np.fft.fft2(I1)
#fft_Q1 = np.fft.fft2(Q1/2)
#fft_U1 = np.fft.fft2(U1)
print("fft1 done!")
fft_I2 = np.fft.fft2(I2)*norm_fft
#fft_Q2 = np.fft.fft2(Q2/2)
#fft_U2 = np.fft.fft2(U2/2)
print("fft2 done!")
#fft_I3 = np.fft.fft2(I3)
#fft_Q3 = np.fft.fft2(Q3/2)
#fft_U3 = np.fft.fft2(U3)
print("fft3 done!")
#fft_I4 = np.fft.fft2(I4)
#fft_Q4 = np.fft.fft2(Q4/2)
#fft_U4 = np.fft.fft2(U4)
print("fft4 done!")
'''
fft_I5 = np.fft.fft2(I5)*norm_fft
fft_Q5 = np.fft.fft2(Q5/2)*norm_fft
fft_U5 = np.fft.fft2(U5/2)*norm_fft
'''
print("fft5 done!")
#fft_I6 = np.fft.fft2(I6)
#fft_Q6 = np.fft.fft2(Q6/2)
#fft_U6 = np.fft.fft2(U6)
print("fft6 done!")
#fft_I7 = np.fft.fft2(I7)
#fft_Q7 = np.fft.fft2(Q7/2)
#fft_U7 = np.fft.fft2(U7)
print("ffts done!")
#fft=fft*norm_pix*norm_fft

E_fft, B_fft = qu2eb(Q2,U2,cm['pixel_size'])
#freq = 2*np.pi*np.fft.fftfreq(cm['n'],cm['pixel_size'])
#x, y = np.meshgrid(freq, freq, sparse = True)
ell, chi = get_ell_array(cm['n'],cm['pixel_size'])

'''
mod0_alm2_EE = np.abs(fft_Q5)**2
mod1_alm2_EE = np.abs(fft_U5)**2
mod2_alm2_EE = np.abs(fft_Q2)**2
mod3_alm2_EE = np.abs(fft_Q3)**2
mod4_alm2_EE = np.abs(fft_Q4)**2
mod5_alm2_EE = np.abs(fft_Q5)**2
mod6_alm2_EE = np.abs(fft_Q6)**2
mod7_alm2_EE = np.abs(fft_Q7)**2
'''
pow_I = np.abs(fft_I2)**2
pow_E = np.abs(E_fft)**2
pow_TE = np.real(fft_I2*np.conj(E_fft))
pow_TB = np.real(fft_I2*np.conj(B_fft))
pow_EB = np.real(E_fft*np.conj(B_fft))

ell_min = np.arange(120)*25
ell_max = np.arange(120)*25+25
ell_cen = 0.5*(ell_min+ell_max)
n_modes = len(ell_cen)

Cl_TT = np.zeros(n_modes)

Cl_TE = np.zeros(n_modes)
Cl_EE = np.zeros(n_modes)
Cl_TB = np.zeros(n_modes)
Cl_EB = np.zeros(n_modes)
'''
Cl1_EE = np.zeros(n_modes)

Cl2_EE = np.zeros(n_modes)
Cl3_EE = np.zeros(n_modes)
Cl4_EE = np.zeros(n_modes)
Cl5_EE = np.zeros(n_modes)
Cl6_EE = np.zeros(n_modes)
Cl7_EE = np.zeros(n_modes)

'''
'''
pixelspacingrad = float(pixelsizearcmin)*arcmins2radians
kx = 2.0*pi*fftfreq(pixelnumber, pixelspacingrad)
kx2d = tile(kx, (pixelnumber, 1))
ky = 2.0*pi*fftfreq(pixelnumber, pixelspacingrad)
ky2d = transpose(tile(ky, (pixelnumber, 1)))
cl2d = zeros((pixelnumber, pixelnumber))
k2d = sqrt(kx2d*kx2d + ky2d*ky2d)
'''
print("n_modes")

for i in range(n_modes):
    Cl_TT[i] = np.mean(pow_I[np.logical_and(ell >= ell_min[i],ell < ell_max[i])])	
    Cl_TE[i] = np.mean(pow_TE[np.logical_and(ell >= ell_min[i],ell < ell_max[i])])
    Cl_EE[i] = np.mean(pow_E[np.logical_and(ell >= ell_min[i],ell < ell_max[i])])
    Cl_TB[i] = np.mean(pow_TB[np.logical_and(ell >= ell_min[i],ell < ell_max[i])])	
    Cl_EB[i] = np.mean(pow_EB[np.logical_and(ell >= ell_min[i],ell < ell_max[i])])
    '''
    Cl1_EE[i] = np.mean(mod1_alm2_EE[np.logical_and(ell >= ell_min[i],ell < ell_max[i])])
    
    Cl2_EE[i] = np.mean(mod2_alm2_EE[np.logical_and(ell >= ell_min[i],ell < ell_max[i])])
    Cl3_EE[i] = np.mean(mod3_alm2_EE[np.logical_and(ell >= ell_min[i],ell < ell_max[i])])
    Cl4_EE[i] = np.mean(mod4_alm2_EE[np.logical_and(ell >= ell_min[i],ell < ell_max[i])])
    Cl5_EE[i] = np.mean(mod5_alm2_EE[np.logical_and(ell >= ell_min[i],ell < ell_max[i])])
    Cl6_EE[i] = np.mean(mod6_alm2_EE[np.logical_and(ell >= ell_min[i],ell < ell_max[i])])
    Cl7_EE[i] = np.mean(mod7_alm2_EE[np.logical_and(ell >= ell_min[i],ell < ell_max[i])])
    '''
    print(i)

#Cl_EE = (Cl0_EE+Cl1_EE)/2
Dl_TT = ell_cen*(ell_cen+1)*Cl_TT/(2*np.pi)
Dl_TE = ell_cen*(ell_cen+1)*Cl_TE/(2*np.pi)
Dl_EE = ell_cen*(ell_cen+1)*Cl_EE/(2*np.pi)
Dl_TB = ell_cen*(ell_cen+1)*Cl_TB/(2*np.pi)
Dl_EB = ell_cen*(ell_cen+1)*Cl_EB/(2*np.pi)
#Dl0_EE = ell_cen*(ell_cen+1)*Cl0_EE/(2*np.pi) 
#Dl1_EE = ell_cen*(ell_cen+1)*Cl1_EE/(2*np.pi)

map_avg = (map0+map1+map2+map3+map4+map5+map6+map7)/8

plt.figure()
plt.plot(ell_cen,Dl_TT)
plt.xlabel('ell')
plt.ylabel('Dl_TT')

plt.figure()
plt.plot(ell_cen,Dl_TE)
plt.xlabel('ell')
plt.ylabel('Dl_TE')

plt.figure()
plt.plot(ell_cen,Dl_TB)
plt.xlabel('ell')
plt.ylabel('Dl_TB')

plt.figure()
plt.plot(ell_cen,Dl_EE)
plt.xlabel('ell')
plt.ylabel('Dl_EE ')

plt.figure()
plt.plot(ell_cen,Dl_EB)
plt.xlabel('ell')
plt.ylabel('Dl_EB ')
'''
plt.figure()
plt.plot(ell_cen,Dl1_EE)
plt.xlabel('ell')
plt.ylabel('Dl_EE (U/2)')
'''
plt.show()

embed()
#cm['all'].make_plot(cm['all'].I)
#pylab.show()

                                
