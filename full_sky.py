import sys, platform, os
import matplotlib.pyplot as plt
import numpy as np
import pickle

import unittest, wignerpy._wignerpy as wp
import numpy as np
import commands, os, time, math

import camb
from IPython import embed
#from sympy.physics.wigner import wigner_3j
#import sympy
#from sage.all import ZZ
#from sage.functions.wigner import wigner_3j
#https://stackoverflow.com/questions/15724000/using-sage-math-library-within-python

'''
print('Using CAMB installed at %s'%(os.path.realpath(os.path.join(os.getcwd(),'..'))))
#uncomment this if you are running remotely and want to keep in synch with repo changes
#if platform.system()!='Windows':
#    !cd $HOME/git/camb; git pull github master; git log -1
sys.path.insert(0,os.path.realpath(os.path.join(os.getcwd(),'..')))
#import camb
#from camb import model, initialpower
print('CAMB version: %s '%camb.__version__)

#Set up a new set of parameters for CAMB
pars = camb.CAMBparams()
#This function sets up CosmoMC-like settings, with one massive neutrino and helium set using BBN consistency
pars.set_cosmology(H0=67.5, ombh2=0.022, omch2=0.122, mnu=0.06, omk=0, tau=0.06)
pars.InitPower.set_params(ns=0.965, r=0)
pars.set_for_lmax(2500, lens_potential_accuracy=4)
 
#calculate results for these parameters
results = camb.get_results(pars)
#cl_camb=results.get_lens_potential_cls(2500)
#get dictionary of CAMB power spectra
powers =results.get_cmb_power_spectra(pars, CMB_unit='muK')
for name in powers: print(name)

#plot the total lensed CMB power spectra versus unlensed, and fractional difference
totCL=powers['total']
lens_pot = powers['lens_potential']
lensedCL=powers['lensed_scalar']
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
Cl_PP = np.nan_to_num((PP_lenspot  * 2*np.pi)/(ells*(ells+1)))
'''
N_L = pickle.load(open("N(L).pkl","rb"))

L1 = N_L["L"][1:len(N_L["L"])]
L2 = N_L["L"][1:len(N_L["L"])]
L3 = N_L["L"][1:len(N_L["L"])]

#sympy.N(wigner_3j(l1,l2,l3,-s,s,0))
#def F_s_l1l2l3(s,l1,l2,l3):
#    return wp.wigner3j(l1,l2,l3,-s,s,0)

def F_s_l1l2l3(s,l2,l3):
    return camb.bispectrum.threej(l2,l3,s,0)

#def f_EB_l1l2l(l1,l2,l):
#    return ((-l1*(l1+1)+l2*(l2+1)+l*(l+1))*np.sqrt((2*l1+1)*(2*l2+1)*(2*l+1)/(16.0*np.pi))*(F_s_l1l2l3(-2,l1,l2,l)-F_s_l1l2l3(2,l1,l2,l)))/(2.0j)

def f_EB_l2l(l2,l):
    return (F_s_l1l2l3(-2,l2,l)-F_s_l1l2l3(2,l2,l))/(2.0j)

def f_EB_l1l2l(l1,l2,l):
    return (-l1*(l1+1)+l2*(l2+1)+l*(l+1))*np.sqrt((2*l1+1)*(2*l2+1)*(2*l+1)/(16.0*np.pi))*f_EB_l2l(l2,l)

def ell_one(l2,l):
    return np.arange(max(abs(l2-l),abs(2)),l2+l+1)

def ell_three(l1,l2):
    return np.arange(max(abs(l2-l),abs(2)),l2+l1+1)

#l1 = np.arange(max(abs(l2-l3),abs(2))
#(-l1*(l1+1)+l2*(l2+1)+l3*(l3+1))*np.sqrt((2*l1+1)*(2*l2+1)*(2*l3+1)/(16*np.pi)))

Cl1_Bres = []

ps = pickle.load(open("PS.pkl","rb"))
N_XX = pickle.load(open("A_XX.pkl","rb"))

#N_PP = N_XX['A_PP1'][1:len(N_XX['A_PP1'])]
N_EE = N_XX['A_EE1'][1:len(N_XX['A_EE1'])]
N_BB = N_XX['A_BB1'][1:len(N_XX['A_BB1'])]
N_EB = N_XX['A_EB1'][1:len(N_XX['A_EB1'])]


#embed()
Cl_TT = ps['Cl_TT']
Cl_EE = ps['Cl_EE']
Cl_BB = ps['Cl_BB']
Cl_TE = ps['Cl_TE']
Cl_PP = ps['Cl_PP']

Cl_TT = Cl_TT[2:len(Cl_TT)]
Cl_EE = Cl_EE[2:len(Cl_EE)]
Cl_BB = Cl_BB[2:len(Cl_BB)]
Cl_TE = Cl_TE[2:len(Cl_TE)]
Cl_PP = Cl_PP[2:len(Cl_PP)]



'''
i=0

for i in range(len(L1)):
    summand = []
    for j in range(len(L2)):
        for k in range(len(L3)):
            #embed()
            f_abs2 = np.real(f_EB_l1l2l(L1[i],L2[j],L3[k])*np.conj(f_EB_l1l2l(L1[i],L2[j],L3[k])))
            if f_abs2 !=0.0:
                summand.append(f_abs2*(Cl_EE[j]*Cl_PP[k]-((Cl_EE[j])**2/(Cl_EE[j]+N_EE[j]))*((Cl_PP[k])**2/(Cl_PP[k]+N_PP[k]))))
            #print(f_abs2)
            print(i,j,k)
    #embed()
    Cl1_Bres.append(np.sum(summand)/(2.0*L1[i]+1.0))

'''


Cl1_Bres = np.zeros(2050)
'''
for j in range(len(L2)):
    for k in range(len(L3)):
       
        f_abs2 = np.real(f_EB_l1l2l(ell_one(L2[j],L3[k]),L2[j],L3[k])*np.conj(f_EB_l1l2l(ell_one(L2[j],L3[k]),L2[j],L3[k])))        
	summand = f_abs2*(Cl_EE[j]*Cl_PP[k]-((Cl_EE[j])**2/(Cl_EE[j]+N_EE[j]))*((Cl_PP[k])**2/(Cl_PP[k]+N_PP[k])))
        Cl1_Bres[max(abs(L2[j]-L3[k]),abs(2)):L2[j]+L3[k]+1] += summand
	print(j,k)
'''
for j in range(len(L2)):
    for k in range(len(L3)):
       
        f_abs2 = np.real(f_EB_l1l2l(ell_one(L2[j],L3[k]),L2[j],L3[k])*np.conj(f_EB_l1l2l(ell_one(L2[j],L3[k]),L2[j],L3[k])))        
	#summand = f_abs2*(Cl_EE[j]*Cl_PP[k]-((Cl_EE[j])**2/(Cl_EE[j]+N_EE[j]))*((Cl_PP[k])**2/(Cl_PP[k]+0.0)))
        summand = f_abs2*(Cl_EE[j]*Cl_PP[k])
        Cl1_Bres[max(abs(L2[j]-L3[k]),abs(2)):L2[j]+L3[k]+1] += summand
	print(j,k)
Cl1_Bres_test = Cl1_Bres[2:1001]/(2.0*L1+1)

plt.plot(L1,Cl1_Bres_test)
plt.xscale('log')
plt.yscale('log')
plt.title('Full sky')
plt.xlabel('L')
plt.ylabel(r'$C_{\ell_1}^{B_{res}}$')
plt.figure(2)
plt.plot(L1,Cl1_Bres_test/(Cl_BB[2:len(N_XX['A_PP1'])+1]))
plt.xscale('log')
plt.yscale('log')
plt.title('full-CAMB')
plt.xlabel('L')
plt.ylabel(r'$C_{\ell_1}^{B_{lens}}$')
plt.show()

plt.plot(L1,Cl_PP[2:len(N_XX['A_PP1'])+1])
plt.xscale('log')
plt.yscale('log')
plt.title('full-CAMB')
plt.xlabel('L')
plt.ylabel(r'$C_{\ell_1}^{PP}$')
plt.show()


embed()
