#http://camb.readthedocs.io/en/latest/CAMBdemo.html

#%matplotlib inline
import sys, platform, os
import matplotlib.pyplot as plt
import numpy as np
from IPython import embed
print('Using CAMB installed at %s'%(os.path.realpath(os.path.join(os.getcwd(),'..'))))
#uncomment this if you are running remotely and want to keep in synch with repo changes
#if platform.system()!='Windows':
#    !cd $HOME/git/camb; git pull github master; git log -1
sys.path.insert(0,os.path.realpath(os.path.join(os.getcwd(),'..')))
import camb
from camb import model, initialpower
print('CAMB version: %s '%camb.__version__)

#Set up a new set of parameters for CAMB
pars = camb.CAMBparams()
#This function sets up CosmoMC-like settings, with one massive neutrino and helium set using BBN consistency
pars.set_cosmology(H0=67.5, ombh2=0.022, omch2=0.122, mnu=0.06, omk=0, tau=0.06)
pars.InitPower.set_params(ns=0.965, r=0)
pars.set_for_lmax(2500, lens_potential_accuracy=0)

#calculate results for these parameters
results = camb.get_results(pars)

#get dictionary of CAMB power spectra
powers =results.get_cmb_power_spectra(pars, CMB_unit='muK')
for name in powers: print(name)

#plot the total lensed CMB power spectra versus unlensed, and fractional difference
totCL=powers['total']
lensedCL=powers['lensed_scalar']
lens_pot = powers['lens_potential']
print(totCL.shape)

#Python CL arrays are all zero based (starting at L=0), Note L=0,1 entries will be zero by default.
#The different CL are always in the order TT, EE, BB, TE (with BB=0 for unlensed scalar results).
PP_lenspot = lens_pot[:,0]

TT_totCL = totCL[:,0]
EE_totCL = totCL[:,1]
BB_totCL = totCL[:,2]
TE_totCL = totCL[:,3]
TT_lensedCL = lensedCL[:,0]
EE_lensedCL = lensedCL[:,1]
BB_lensedCL = lensedCL[:,2]
TE_lensedCL = lensedCL[:,3]

ells = np.arange(totCL.shape[0])

Cl_TT = np.nan_to_num((TT_lensedCL * 2*np.pi)/(ells*(ells+1)))
Cl_EE = np.nan_to_num((EE_lensedCL * 2*np.pi)/(ells*(ells+1)))
Cl_BB = np.nan_to_num((BB_lensedCL * 2*np.pi)/(ells*(ells+1)))
Cl_TE = np.nan_to_num((TE_lensedCL * 2*np.pi)/(ells*(ells+1)))
Cl_PP = np.nan_to_num((PP_lenspot  * 2*np.pi)/(ells*(ells+1)))**2


#Now assuming a fully polarised detector

#https://arxiv.org/pdf/1702.01871.pdf

#SPT-3G
#Calculates the noise component of the power spectrum using SPT-3G estimates
'''
noise_uK_arcmin = 3.0
l_knee = 200
fwhm_arcmin = 1.2
'''

#PLANCK
'''
noise_uK_arcmin = 56
l_knee = 200
fwhm_arcmin = 7
'''

#Exp1
'''
noise_uK_arcmin = 9.6
l_knee = 200
fwhm_arcmin = 8
'''

#CMBPol
'''
noise_uK_arcmin = np.sqrt(2)
l_knee = 200
fwhm_arcmin = 4
'''

def CL_noise_temp(noise_uK_arcmin_pol, fwhm_arcmin):
    del_x = (noise_uK_arcmin_pol * np.pi/180./60./np.sqrt(2))
    CL_n = (del_x**2) * np.exp((ells*(ells+1) * (fwhm_arcmin * np.pi/180./60.)**2)/(8. * np.log(2)))
    return CL_n

def CL_noise_pol(noise_uK_arcmin_pol, fwhm_arcmin):
    del_x = (noise_uK_arcmin_pol * np.pi/180./60.)
    CL_n = (del_x**2) * np.exp((ells*(ells+1) * (fwhm_arcmin * np.pi/180./60.)**2)/(8. * np.log(2)))
    return CL_n

#Calculating the noise hat power spectrum where Cl_EE and Cl_BB is the scalar lensed power spectrum with size (2551,) or binned ells as 0<=ell<=2550
#SPT-3G
CL_TT_hat1 = Cl_TT + CL_noise_temp(3.0,1.2)
CL_EE_hat1 = Cl_EE + CL_noise_pol(3.0,1.2)
CL_BB_hat1 = Cl_BB + CL_noise_pol(3.0,1.2)

#PLANCK
CL_TT_hat2 = Cl_TT + CL_noise_temp(56,7)
CL_EE_hat2 = Cl_EE + CL_noise_pol(56,7)
CL_BB_hat2 = Cl_BB + CL_noise_pol(56,7)

#Exp1
CL_TT_hat3 = Cl_TT + CL_noise_temp(9.6,8)
CL_EE_hat3 = Cl_EE + CL_noise_pol(9.6,8)
CL_BB_hat3 = Cl_BB + CL_noise_pol(9.6,8)
#CMBPol
CL_TT_hat4 = Cl_TT + CL_noise_temp(np.sqrt(2),4)
CL_EE_hat4 = Cl_EE + CL_noise_pol(np.sqrt(2),4)
CL_BB_hat4 = Cl_BB + CL_noise_pol(np.sqrt(2),4)
CL_PP_hat4 = Cl_PP + CL_noise_temp(np.sqrt(2),4)

#So what the np.interp function does in this context is it takes np.interp(x,xp,fp) = y where the parameters are defined in more detail below. So in effect it takes an array of x values and and an array of xp values where those xp values are used as the reference for x to interpolate from. From there, it is possible to just take the yp value as the new interpolated value for the x input values using fp. So below, taking the magnitude of Lx and Ly coordinate or l_x and l_y
'''
Parameters
----------
x : array_like
    The x-coordinates of the interpolated values.

xp : 1-D sequence of floats
    The x-coordinates of the data points, must be increasing if argument
    `period` is not specified. Otherwise, `xp` is internally sorted after
    normalizing the periodic boundaries with ``xp = xp % period``.

fp : 1-D sequence of floats
    The y-coordinates of the data points, same length as `xp`.

Returns
-------
y : float or ndarray
    The interpolated values, same shape as `x`.
'''
'''
C_L_EE= np.interp(L,ells,Cl_EE)
C_L_BB= np.interp(L,ells,Cl_BB)
C_L_EE_hat = np.interp(L,ells,CL_EE_hat)
C_L_BB_hat = np.interp(L,ells,CL_BB_hat)
'''

#http://background.uchicago.edu/~whu/Papers/recon.pdf

#embed()
def A_TB(L_x,del_l,CL_TT_hat,CL_BB_hat):
    lx = np.arange(-3000,3000,del_l)
    ly=np.copy(lx)
    Lx,Ly = np.meshgrid(lx,ly)
    L = np.sqrt(Lx**2+Ly**2)
    phi = np.nan_to_num(np.arctan2(Ly,Lx))




    l1_x = Lx
    l1_y = Ly
    l2_x = L_x - l1_x
    l2_y = -l1_y
    phi_2 = np.nan_to_num(np.arctan2(l2_y,l2_x))
    phi_12 = phi-phi_2
    abs_l1 = np.abs(np.around(L)).astype(int)
    abs_l2 = np.abs(np.around(np.sqrt(l2_x**2+l2_y**2))).astype(int)



    integrand = []
    
    for i in range(len(l1_x[0])):
        for j in range(len(l1_y[:,0])):
            if np.logical_and(np.logical_and(2<abs_l2[i,j], abs_l2[i,j]<2500),np.logical_and(2<abs_l1[i,j], abs_l1[i,j]<2500))==True:
                #print(i,j)
             
                f_TB = 2.0*(Cl_TE[abs_l1[i,j]])*(np.cos(2*phi_12[i,j]))
                F_TB = f_TB/(CL_TT_hat[abs_l1[i,j]]*CL_BB_hat[abs_l2[i,j]])
                integrand.append(f_TB*F_TB)
    return (np.sum(np.array(integrand))*(del_l**2)/(2*np.pi)**2)**(-1) 

def A_EB(L_x,del_l,CL_EE_hat,CL_BB_hat):
    lx = np.arange(-3000,3000,del_l)
    ly=np.copy(lx)
    Lx,Ly = np.meshgrid(lx,ly)
    L = np.sqrt(Lx**2+Ly**2)
    phi = np.nan_to_num(np.arctan2(Ly,Lx))




    l1_x = Lx
    l1_y = Ly
    l2_x = L_x - l1_x
    l2_y = -l1_y
    phi_2 = np.nan_to_num(np.arctan2(l2_y,l2_x))
    phi_12 = phi-phi_2
    abs_l1 = np.abs(np.around(L)).astype(int)
    abs_l2 = np.abs(np.around(np.sqrt(l2_x**2+l2_y**2))).astype(int)

    integrand = []
    
    for i in range(len(l1_x[0])):
        for j in range(len(l1_y[:,0])):
            if np.logical_and(np.logical_and(2<abs_l2[i,j], abs_l2[i,j]<2500),np.logical_and(2<abs_l1[i,j], abs_l1[i,j]<2500))==True:
                #print(i,j)
             
                f_EB = 2.0*(Cl_EE[abs_l1[i,j]]-Cl_BB[abs_l2[i,j]])*(np.cos(2*phi_12[i,j]))
                F_EB = f_EB/(CL_EE_hat[abs_l1[i,j]]*CL_BB_hat[abs_l2[i,j]])
                integrand.append(f_EB*F_EB)
    return (np.sum(np.array(integrand))*(del_l**2)/(2*np.pi)**2)**(-1) 

def A_EE(L_x,del_l,CL_EE_hat):
    lx = np.arange(-3000,3000,del_l)
    ly=np.copy(lx)
    Lx,Ly = np.meshgrid(lx,ly)
    L = np.sqrt(Lx**2+Ly**2)
    phi = np.nan_to_num(np.arctan2(Ly,Lx))




    l1_x = Lx
    l1_y = Ly
    l2_x = L_x - l1_x
    l2_y = -l1_y
    phi_2 = np.nan_to_num(np.arctan2(l2_y,l2_x))
    phi_12 = phi-phi_2
    abs_l1 = np.abs(np.around(L)).astype(int)
    abs_l2 = np.abs(np.around(np.sqrt(l2_x**2+l2_y**2))).astype(int)



    integrand = []
    
    for i in range(len(l1_x[0])):
        for j in range(len(l1_y[:,0])):
            if np.logical_and(np.logical_and(2<abs_l2[i,j], abs_l2[i,j]<2500),np.logical_and(2<abs_l1[i,j], abs_l1[i,j]<2500))==True:
                #print(i,j)
             
                f_EE = 2.0*(Cl_EE[abs_l1[i,j]]-Cl_EE[abs_l2[i,j]])*(np.sin(2*phi_12[i,j]))
                F_EE = f_EE/(CL_EE_hat[abs_l1[i,j]]*CL_EE_hat[abs_l2[i,j]])
                integrand.append(f_EE*F_EE)
    return (np.sum(np.array(integrand))*(del_l**2)/(2*np.pi)**2)**(-1) 

def A_BB(L_x,del_l,CL_BB_hat):
    lx = np.arange(-3000,3000,del_l)
    ly=np.copy(lx)
    Lx,Ly = np.meshgrid(lx,ly)
    L = np.sqrt(Lx**2+Ly**2)
    phi = np.nan_to_num(np.arctan2(Ly,Lx))




    l1_x = Lx
    l1_y = Ly
    l2_x = L_x - l1_x
    l2_y = -l1_y
    phi_2 = np.nan_to_num(np.arctan2(l2_y,l2_x))
    phi_12 = phi-phi_2
    abs_l1 = np.abs(np.around(L)).astype(int)
    abs_l2 = np.abs(np.around(np.sqrt(l2_x**2+l2_y**2))).astype(int)



    integrand = []
    
    for i in range(len(l1_x[0])):
        for j in range(len(l1_y[:,0])):
            if np.logical_and(np.logical_and(2<abs_l2[i,j], abs_l2[i,j]<2500),np.logical_and(2<abs_l1[i,j], abs_l1[i,j]<2500))==True:
                #print(i,j)
             
                f_BB = (Cl_BB[abs_l1[i,j]]+Cl_BB[abs_l2[i,j]])*(np.sin(2*phi_12[i,j]))
                F_BB = f_BB/(CL_BB_hat[abs_l1[i,j]]*CL_BB_hat[abs_l2[i,j]])
                integrand.append(f_BB*F_BB)
    return (np.sum(np.array(integrand))*(del_l**2)/(2*np.pi)**2)**(-1) 

A_TB1 = []
A_EB1 = []
A_TB2 = []
A_EB2 = []
A_TB3 = []
A_EB3 = []
A_TB4 = []
A_EB4 = []

L = np.arange(2549)+1

for i in range(2549):
   
    A_TB1.append(A_TB(L[i],30,CL_TT_hat1,CL_BB_hat1))
    A_EB1.append(A_EB(L[i],30,CL_EE_hat1,CL_BB_hat1))
    A_TB2.append(A_TB(L[i],30,CL_TT_hat2,CL_BB_hat2))
    A_EB2.append(A_EB(L[i],30,CL_EE_hat2,CL_BB_hat2))
    A_TB3.append(A_TB(L[i],30,CL_TT_hat3,CL_BB_hat3))
    A_EB3.append(A_EB(L[i],30,CL_EE_hat3,CL_BB_hat3))
    A_TB4.append(A_TB(L[i],30,CL_TT_hat4,CL_BB_hat4))
    A_EB4.append(A_EB(L[i],30,CL_EE_hat4,CL_BB_hat4))

    #A_EE1.append(A_EE(L[i],10,CL_EE_hat4))
    #A_BB1.append(A_BB(L[i],10,CL_BB_hat4))
    #A_EB1.append(A_EB(L[i],10,CL_EE_hat4,CL_BB_hat4))
    
    

    print(i)


embed()

A_TB1 = np.array(A_TB1)
A_EB1 = np.array(A_EB1)
A_TB2 = np.array(A_TB2)
A_EB2 = np.array(A_EB2)
A_TB3 = np.array(A_TB3)
A_EB3 = np.array(A_EB3)
A_TB4 = np.array(A_TB4)
A_EB4 = np.array(A_EB4)

D_TB2 =np.sqrt((L**2)*A_TB2/(2*np.pi))
D_EB2 =np.sqrt((L**2)*A_EB2/(2*np.pi))
D_TB3 =np.sqrt((L**2)*A_TB3/(2*np.pi))
D_EB3 =np.sqrt((L**2)*A_EB3/(2*np.pi))
D_TB4 =np.sqrt((L**2)*A_TB4/(2*np.pi))
D_EB4 =np.sqrt((L**2)*A_EB4/(2*np.pi))



plt.plot(L,np.sqrt((L**2)*A_TB2/(2*np.pi))/np.radians(1),ls = '--',c='b')
plt.plot(L,np.sqrt((L**2)*A_EB2/(2*np.pi))/np.radians(1),ls = '--',c='r')
plt.plot(L,np.sqrt((L**2)*A_TB3/(2*np.pi))/np.radians(1),ls = ':',c='b')
plt.plot(L,np.sqrt((L**2)*A_EB3/(2*np.pi))/np.radians(1),ls = ':',c='r')
plt.plot(L,np.sqrt((L**2)*A_TB4/(2*np.pi))/np.radians(1),ls = '-.',c='b')
plt.plot(L,np.sqrt((L**2)*A_EB4/(2*np.pi))/np.radians(1),ls = '-.',c='r')
plt.xlabel('L')
plt.ylabel(r'$[L^{2}N(L)/ 2\pi]^{1/2}$')
plt.xscale('log')
plt.yscale('log')
plt.ylim(1e-3, 1e1)
plt.show()	
embed()

data = { 'L' : L, 'A_TB1' : A_TB1, 'A_EB1' : A_EB1, 'A_TB2' : A_TB2, 'A_EB2' : A_EB2, 'A_TB3' : A_TB3, 'A_EB3' : A_EB3, 'A_TB4' : A_TB4, 'A_EB4' : A_EB4 }
with open("N_AA.pkl", "wb") as infile:
    pickle.dump(data, infile) 
  
