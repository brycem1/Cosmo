import sys, platform, os
import matplotlib.pyplot as plt
import numpy as np
import pickle

import unittest, wignerpy._wignerpy as wp
import numpy as np
import commands, os, time, math

import numba

import camb
from IPython import embed

ps = pickle.load(open("PS.pkl","rb"))
N_AA_data = pickle.load(open("N_AA.pkl","rb"))
N_EE_data = pickle.load(open("N_EE.pkl","rb"))

L1 = N_AA_data["L"][1:len(N_AA_data["L"])]
L2 = N_AA_data["L"][1:len(N_AA_data["L"])]
L3 = N_AA_data["L"][1:len(N_AA_data["L"])]

#A_TB1 = N_AA_data['A_TB1'][1:len(N_AA_data['A_TB1'])]
#A_EB1 = N_AA_data['A_EB1'][1:len(N_AA_data['A_EB1'])]
A_TB2 = N_AA_data['A_TB2'][1:len(N_AA_data['A_TB2'])]
A_EB2 = N_AA_data['A_EB2'][1:len(N_AA_data['A_EB2'])]
A_TB3 = N_AA_data['A_TB3'][1:len(N_AA_data['A_TB3'])]
A_EB3 = N_AA_data['A_EB3'][1:len(N_AA_data['A_EB3'])]
A_TB4 = N_AA_data['A_TB4'][1:len(N_AA_data['A_TB4'])]
A_EB4 = N_AA_data['A_EB4'][1:len(N_AA_data['A_EB4'])]

Cl_TT = ps['Cl_TT']
Cl_EE = ps['Cl_EE']
Cl_BB = ps['Cl_BB']
Cl_TE = ps['Cl_TE']
Cl_PP = ps['Cl_PP']
Cl_AA = ps['Cl_AA']

Cl_TT = Cl_TT[2:len(Cl_TT)]
Cl_EE = Cl_EE[2:len(Cl_EE)]
Cl_BB = Cl_BB[2:len(Cl_BB)]
Cl_TE = Cl_TE[2:len(Cl_TE)]
Cl_PP = Cl_PP[2:len(Cl_PP)]
Cl_AA = Cl_AA[2:len(Cl_AA)]

N_EE1 = N_EE_data['N_EE1'][2:len(N_EE_data['N_EE1'])]
N_EE2 = N_EE_data['N_EE2'][2:len(N_EE_data['N_EE2'])]
N_EE3 = N_EE_data['N_EE3'][2:len(N_EE_data['N_EE3'])]
N_EE4 = N_EE_data['N_EE4'][2:len(N_EE_data['N_EE4'])]

N_EE = N_EE4
N_AA = A_EB4
#embed()
'''
plt.plot(L1,N_EE[0:999],label='N_EE',c='b')
plt.plot(L1,N_AA,label='N_AA',c='r')
plt.legend('lower right')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('L')
plt.ylabel('N(L)')
plt.show()
'''
@numba.jit

def Cl_BB_AA_lensf(L_x,del_l):
    lx = np.arange(-3000,3000,del_l)
    ly=np.copy(lx)
    Lx,Ly = np.meshgrid(lx,ly)
    L = np.sqrt(Lx**2+Ly**2)
    phi = np.nan_to_num(np.arctan2(Ly,Lx))




    l1_x = Lx
    l1_y = Ly
    l2_x = L_x - l1_x
    l2_y = -l1_y
    L_y = 0.0
    phi_2 = np.nan_to_num(np.arctan2(l2_y,l2_x))
    phi_12 = phi-phi_2
    abs_l1 = np.abs(np.around(L)).astype(int)
    abs_l2 = np.abs(np.around(np.sqrt(l2_x**2+l2_y**2))).astype(int)
    

    phi_L = np.nan_to_num(np.arctan2(L_y,L_x))
    phi_L1 = phi_L-phi
    W_Ll1 = 2.*np.cos(2.*phi_L1)
    
    integrand = []
    
    for i in range(len(l1_x[0])):
        for j in range(len(l1_y[:,0])):
            if np.logical_and(np.logical_and(2<=abs_l2[i,j], abs_l2[i,j]<=2500),np.logical_and(2<=abs_l1[i,j], abs_l1[i,j]<=2500))==True:
                #print(i,j)
                #sq_br = (1.0-(Cl_EE[abs_l1[i,j]-2]/(Cl_EE[abs_l1[i,j]-2]+N_EE[abs_l1[i,j]-2]))*(Cl_AA[abs_l2[i,j]-2]/(Cl_AA[abs_l2[i,j]-2]+N_AA[abs_l2[i,j]-2])))
                integrand.append(W_Ll1[i,j]**2*Cl_EE[abs_l1[i,j]-2]*Cl_AA[abs_l2[i,j]-2])
    return (np.sum(np.array(integrand))*(del_l**2)/(2*np.pi)**2)

def Cl_BB_AA_resf(L_x,del_l):
    lx = np.arange(-3000,3000,del_l)
    ly=np.copy(lx)
    Lx,Ly = np.meshgrid(lx,ly)
    L = np.sqrt(Lx**2+Ly**2)
    phi = np.nan_to_num(np.arctan2(Ly,Lx))




    l1_x = Lx
    l1_y = Ly
    l2_x = L_x - l1_x
    l2_y = -l1_y
    L_y = 0.0
    phi_2 = np.nan_to_num(np.arctan2(l2_y,l2_x))
    phi_12 = phi-phi_2
    abs_l1 = np.abs(np.around(L)).astype(int)
    abs_l2 = np.abs(np.around(np.sqrt(l2_x**2+l2_y**2))).astype(int)
    

    phi_L = np.nan_to_num(np.arctan2(L_y,L_x))
    phi_L1 = phi_L-phi
    W_Ll1 = 2.*np.cos(2.*phi_L1)
    
    integrand = []
    
    for i in range(len(l1_x[0])):
        for j in range(len(l1_y[:,0])):
            if np.logical_and(np.logical_and(2<=abs_l2[i,j], abs_l2[i,j]<=2500),np.logical_and(2<=abs_l1[i,j], abs_l1[i,j]<=2500))==True:
                #print(i,j)
                sq_br = (1.0-(Cl_EE[abs_l1[i,j]-2]/(Cl_EE[abs_l1[i,j]-2]+N_EE[abs_l1[i,j]-2]))*(Cl_AA[abs_l2[i,j]-2]/(Cl_AA[abs_l2[i,j]-2]+N_AA[abs_l2[i,j]-2])))
                integrand.append(W_Ll1[i,j]**2*Cl_EE[abs_l1[i,j]-2]*Cl_AA[abs_l2[i,j]-2]*sq_br)
    return (np.sum(np.array(integrand))*(del_l**2)/(2*np.pi)**2)

Cl_BB_AA_lens = []
Cl_BB_AA_res = []

for k in range(999):
    Cl_BB_AA_lens.append(Cl_BB_AA_lensf(L1[k],30))
    Cl_BB_AA_res.append(Cl_BB_AA_resf(L1[k],30))
    print(L1[k])

Cl_BB_AA_lens=np.array(Cl_BB_AA_lens)
Cl_BB_AA_res=np.array(Cl_BB_AA_res)
embed()
plt.figure(1)
plt.plot(L1[0:999],Cl_BB_AA_res,label = 'Cl_BB_rot_res',c='b')
plt.plot(L1[0:999],Cl_BB[0:999],label = 'Cl_BB_lens_camb',c='r')
plt.xscale('log')
plt.yscale('log')
plt.legend(loc='lower left')
plt.title('Lens rot res')
plt.xlabel('L')
plt.ylabel(r'$C_{\ell_1}^{B_{res}}$')
plt.figure(2)
plt.plot(L1[0:999],Cl_BB_AA_lens,label = 'Cl_BB_rot_lens',c='b')
plt.plot(L1[0:999],Cl_BB[0:999],label = 'Cl_BB_lens_camb',c='r')
plt.xscale('log')
plt.yscale('log')
plt.legend(loc='lower left')
plt.title('Lens rot')
plt.xlabel('L')
plt.ylabel(r'$C_{\ell_1}^{B_{lens}}$')
plt.show()
embed()



