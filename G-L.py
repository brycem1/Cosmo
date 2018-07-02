import sys, platform, os
import matplotlib.pyplot as plt
import numpy as np
import pickle

import unittest, wignerpy._wignerpy as wp
import numpy as np
import commands, os, time, math

from quicklens.math.wignerd import gauss_legendre_quadrature as glq

import camb
from IPython import embed


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

#l1 = np.arange(max(abs(l2-l3),abs(2))
#(-l1*(l1+1)+l2*(l2+1)+l3*(l3+1))*np.sqrt((2*l1+1)*(2*l2+1)*(2*l3+1)/(16*np.pi)))

Cl1_Bres = []

ps = pickle.load(open("PS.pkl","rb"))
N_XX = pickle.load(open("A_XX.pkl","rb"))

N_PP = N_XX['A_PP1'][1:len(N_XX['A_PP1'])]
N_EE = N_XX['A_EE1'][1:len(N_XX['A_EE1'])]
N_BB = N_XX['A_BB1'][1:len(N_XX['A_BB1'])]
N_EB = N_XX['A_EB1'][1:len(N_XX['A_EB1'])]



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

#temp
Cl_TT = Cl_TT[2:len(N_XX['A_PP1'])+1]
Cl_EE = Cl_EE[2:len(N_XX['A_PP1'])+1]
Cl_BB = Cl_BB[2:len(N_XX['A_PP1'])+1]
Cl_TE = Cl_TE[2:len(N_XX['A_PP1'])+1]
Cl_PP = Cl_PP[2:len(N_XX['A_PP1'])+1]


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

#embed()

points = glq(3*1000+1/2.0)
cf3m3E_arg = ((2.0*L1+1)/(4.0*np.pi))*(L1-2)*(L1+3)*((Cl_EE**2)/(Cl_EE+N_EE))
cf3m3E = glq.cf_from_cl(points,3, -3, cf3m3E_arg)
cf33E = glq.cf_from_cl(points,3, 3, cf3m3E_arg)
cf3m1E_arg = ((2.0*L1+1)/(4.0*np.pi))*np.sqrt((L1-1)*(L1+2)*(L1-2)*(L1+3))*((Cl_EE**2)/(Cl_EE+N_EE))
cf3m1E = glq.cf_from_cl(points,3, -1, cf3m1E_arg)
cf31E = glq.cf_from_cl(points,3, 1, cf3m1E_arg)
cf1m1E_arg = ((2.0*L1+1)/(4.0*np.pi))*(L1-1)*(L1+2)*((Cl_EE**2)/(Cl_EE+N_EE))
cf1m1E = glq.cf_from_cl(points,1, -1, cf1m1E_arg)
cf11E = glq.cf_from_cl(points,1, 1, cf1m1E_arg)
cfmB_arg = ((2.0*L1+1)/(4.0*np.pi))*(1.0/(Cl_BB+N_BB))
cfmB = glq.cf_from_cl(points,2, -2, cfmB_arg)
cfB = glq.cf_from_cl(points,2, 2, cfmB_arg)

N_PP_int1_arg = (cf33E*cfB-2.0*cf3m1E*cfmB+cf11E*cfB)
N_PP_int1 = glq.cl_from_cf(points,1000, 1, 1, N_PP_int1_arg)

N_PP_int2_arg = (cf3m3E*cfmB-2.0*cf31E*cfB+cf1m1E*cfmB)
N_PP_int2 = glq.cl_from_cf(points,1000, 1, 1, N_PP_int2_arg)



embed()
'''
for j in range(len(L2)):
    for k in range(len(L3)):
       
        f_abs2 = np.real(f_EB_l1l2l(ell_one(L2[j],L3[k]),L2[j],L3[k])*np.conj(f_EB_l1l2l(ell_one(L2[j],L3[k]),L2[j],L3[k])))        
	summand = f_abs2*(Cl_EE[j]*Cl_PP[k]-((Cl_EE[j])**2/(Cl_EE[j]+N_EE[j]))*((Cl_PP[k])**2/(Cl_PP[k]+N_PP[k])))
        Cl1_Bres[max(abs(L2[j]-L3[k]),abs(2)):L2[j]+L3[k]+1] += summand
	print(j,k)
Cl1_Bres_test = Cl1_Bres[2:1001]/(2.0*L1+1)

plt.plot(L1,Cl1_Bres_test)
plt.xscale('log')
plt.yscale('log')
plt.xlabel('L')
plt.ylabel(r'$C_{\ell_1}^{B_{res}}$')
plt.show()
'''




