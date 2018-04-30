import numpy as np
import matplotlib.pyplot as plt
from IPython import embed

sq_pix_array = np.zeros((2401,2401))

'''
for i in range(2401):
    for j in range(2401):
	sq_pix_array[i][j] = [2*i,2*j]
'''
nx, ny = (2401,2401)

x = np.linspace(0,4800,nx)
y = np.linspace(0,4800,ny)
xv,yv = np.meshgrid(x,y)


n_x,n_y = (9,5)

xx = np.linspace(0,6,n_x)
yy = np.linspace(0,6, n_y)
xc,yc = np.meshgrid(xx,yy)
#embed()


plt.figure(1)
plt.xlabel('x')
plt.ylabel('y')
#plt.plot(xv,yv,marker = '.',color ='k', linestyle = 'none')
plt.plot(xc,yc,marker = '.', color = 'm', linestyle = 'none')
plt.show()

