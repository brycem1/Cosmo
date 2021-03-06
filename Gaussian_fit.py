import astropy
import matplotlib.pyplot as plt
from scipy import optimize
from numpy import *
from IPython import embed
import pickle


def gaussian(p, x, y):
    height, center_x, center_y, width_x, width_y = p
    return height*exp(-(((center_x-x)/width_x)**2+((center_y-y)/width_y)**2)/2)

def moments(data):
    total = sum(data)
    X, Y = indices(data.shape)
    center_x = sum(X*data)/total
    center_y = sum(Y*data)/total
    row = data[int(center_x),:]
    col = data[:, int(center_y)]
    width_x = sum(sqrt(abs((arange(col.size)-center_y)**2*col))/sum(col))
    width_y = sum(sqrt(abs((arange(row.size)-center_x)**2*row))/sum(row))
    height = amax(data)
    return height, center_x, center_y, width_x, width_y

def errorfunction(p, x, y, data):
    #embed()
    err =  gaussian(p,x,y) - data
    return err.flatten()

def fitgaussian(data):
    params = moments(data)
    X, Y = indices(data.shape)
    x = X
    y = Y
    #embed()
    p, success = optimize.leastsq(errorfunction, params, args = (x, y, data))
    return p

cm = pickle.load(open("coadded_maps.pkl","rb"))
data = cm['all'].I[250:351,250:351]
X, Y = indices(data.shape)
parameters = fitgaussian(data)
fit = gaussian(parameters, X, Y)

plt.figure()
plt.imshow(data)
plt.colorbar()

plt.show()






################################################
#import pickle 
#import numpy as np
#import matplotlib as plt

#cm = pickle.load(open("coadded_maps.pkl","rb"))
#cm['all'].save_plot(cm['all'].I,"all.png")
################################################
