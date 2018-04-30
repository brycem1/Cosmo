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
print(totCL.shape)

#Python CL arrays are all zero based (starting at L=0), Note L=0,1 entries will be zero by default.
#The different CL are always in the order TT, EE, BB, TE (with BB=0 for unlensed scalar results).

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

def delensed_fraction(lensed_power_spectrum,modes_lensed,frac):
    unlensed_power_spectrum = np.append((1-frac)*Cl_BB[0:modes_lensed-1],Cl_BB[modes_lensed:len(lensed_power_spectrum)])
    return unlensed_power_spectrum
     



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
             
                f_EB = 2*(Cl_EE[abs_l1[i,j]]-Cl_BB[abs_l2[i,j]])*(np.cos(2*phi_12[i,j]))
                F_EB = f_EB/(CL_EE_hat[abs_l1[i,j]]*CL_BB_hat[abs_l2[i,j]])
                integrand.append(f_EB*F_EB)
    return (np.sum(np.array(integrand))*(del_l**2)/(2*np.pi)**2)**(-1) 


f = np.linspace(0.0,1.0,1000)

all_modes=[]
L_500_modes = []

for i in range(len(f)):
    all_m = (1-f[i])*Cl_BB
    L_500 = delensed_fraction(Cl_BB,500,f[i])
    all_modes.append(A_EB(1,10,Cl_EE,all_m))
    L_500_modes.append(A_EB(1,10,Cl_EE,L_500))
    print(i)

all_modes = np.array(all_modes)
L_500_modes = np.array(L_500_modes)

embed()

plt.xlabel('fraction of de-lensing f')
plt.ylabel(r'$[N(L=1)/ 2\pi]^{1/2}$ deg')
plt.plot(f,np.sqrt(all_modes/(2*np.pi))/np.radians(1), c='r')
plt.plot(f,np.sqrt(L_500_modes/(2*np.pi))/np.radians(1), ls= '--',c='k')




