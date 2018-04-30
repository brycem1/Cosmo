import astropy
import matplotlib.pyplot as plt
from astropy.visualization import astropy_mpl_style
plt.style.use(astropy_mpl_style)

from scipy import optimize
import numpy as np
from IPython import embed
import numpy.fft as fft
import pickle
import pdb
from astropy.io import fits
from astropy.utils.data import download_file
import healpy
#matplotlib.rc('text', usetex=True)



import healpy as hp
import numpy as np
import matplotlib
matplotlib.rc('text', usetex=True)
import matplotlib.pyplot as plt
import astropy.units as u
from astropy.coordinates import SkyCoord
from healpy.rotator import Rotator


euler_transf_index = {
    'CG': 1,
    'GC': 2,
    'CE': 3,
    'EC': 4,
    'EG': 5,
    'GE': 6
}

def transform_coords_euler(lon, lat, coord_in, coord_out):
    '''
    Convert between astronomical coordinate systems.
    
    All inputs and outputs are in degrees.
    
    Valid values for coord_in and coord_out are
      'G' for 'Galactic'                (l, b)
      'C' for 'Celestial'/'Equatorial'  (ra, dec)
      'E' for 'Ecliptic'                (lon, lat)
    '''
    
    return hp.rotator.euler(lon, lat, euler_transf_index[coord_in + coord_out])


def transform_coords_rotator(lon, lat, coord_in, coord_out):
    '''
    Convert between astronomical coordinate systems.
    
    All inputs and outputs are in degrees.
    
    Valid values for coord_in and coord_out are
      'G' for 'Galactic'                (l, b)
      'C' for 'Celestial'/'Equatorial'  (ra, dec)
      'E' for 'Ecliptic'                (lon, lat)
    '''
    
    rot = hp.rotator.Rotator(coord=[coord_in, coord_out])
    
    t_in = np.radians(90. - lat)
    p_in = np.radians(lon)
    
    t_out, p_out = rot(t_in, p_in)
    lon_out = np.degrees(p_out)
    lat_out = 90. - np.degrees(t_out)
    
    return lon_out, lat_out


def rand_lon_lat(n):
    '''
    Generate n random longitudes and latitudes, drawn uniformly from the
    surface of a sphere.
    '''
    u = np.random.random(n)
    v = np.random.random(n)
    
    lon = 360. * u
    lat = 90. - np.degrees(np.arccos(2. * v - 1.))
    
    return lon, lat


def dtheta_hist(n):
    l,b = rand_lon_lat(n)
    
    ra0, dec0 = transform_coords_euler(l, b, 'G', 'C')
    ra1, dec1 = transform_coords_rotator(l, b, 'G', 'C')
    
    dir0 = np.vstack([ra0, dec0])
    dir1 = np.vstack([ra1, dec1])
    
    d = hp.rotator.angdist(dir0, dir1, lonlat=True)
    
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.hist(d*3600000., bins=25, normed=True)
    ax.set_xlabel(r'$\Delta \theta \ \left( \mathrm{mas} \right)$', fontsize=18)
    ax.set_ylabel(r'$\mathrm{frequency}$', fontsize=18)
    ax.set_title(r'$\mathrm{Error \ Histogram}$', fontsize=22)
    ax.set_ylim(0., ax.get_ylim()[1]*1.1)

def coord_trans(nside):
    npix = hp.pixelfunc.nside2npix(nside)
    t,p = hp.pixelfunc.pix2ang(nside, np.arange(npix))

    #Going to (theta, phi) to Galactic coordinates (l,b)

    l = np.degrees(p)
    b = 90. - np.degrees(t)
 
    #alpha is RA and delta is DEC in degrees
    delta = np.degrees(np.arcsin(np.cos(np.deg2rad(b))*np.sin(np.deg2rad(l-33))*np.sin(np.deg2rad(62.6))+np.sin(np.deg2rad(b))*np.cos(np.deg2rad(62.6))))
    alpha = np.degrees(np.arcsin((np.cos(np.deg2rad(b))*np.sin(np.deg2rad(l-33))*np.sin(np.deg2rad(62.6))-np.sin(np.deg2rad(b))*np.sin(np.deg2rad(62.6)))/np.cos(np.deg2rad(delta))))+282.25
    

def dtheta_map(nside):
    npix = hp.pixelfunc.nside2npix(nside)
    t,p = hp.pixelfunc.pix2ang(nside, np.arange(npix))
    
    # Galactic -> Celestial
    l = np.degrees(p)
    b = 90. - np.degrees(t)
    
    ra0, dec0 = transform_coords_euler(l, b, 'G', 'C')
    ra1, dec1 = transform_coords_rotator(l, b, 'G', 'C')
    
    dir0 = np.vstack([ra0, dec0])
    dir1 = np.vstack([ra1, dec1])
    
    d = hp.rotator.angdist(dir0, dir1, lonlat=True)
    
    hp.visufunc.mollview(3600000.*d, coord='G', unit='mas', title=r'$\Delta \theta$', format='%.2g')
    
    # Celestial -> Galactic
    ra = np.degrees(p)
    dec = 90. - np.degrees(t)
    
    l0, b0 = transform_coords_euler(ra, dec, 'C', 'G')
    l1, b1 = transform_coords_rotator(ra, dec, 'C', 'G')
    
    dir0 = np.vstack([l0, b0])
    dir1 = np.vstack([l1, b1])
    
    d = hp.rotator.angdist(dir0, dir1, lonlat=True)
    
    hp.visufunc.mollview(3600000.*d, coord='C', unit='mas', title=r'$\Delta \theta$', format='%.2g')

'''
def main():
    dtheta_hist(500000)
    dtheta_map(512)
    plt.show()
    
    return 0

if __name__ == '__main__':
    main()
'''
def G2C(nside):
    npix = hp.pixelfunc.nside2npix(nside)
    ipix =  np.arange(npix)
    t,p = hp.pixelfunc.pix2ang(nside,ipix, nest = True)

    rot = hp.rotator.Rotator(coord=['G', 'C'])
    #(lon,lat)=(l,b)
    l = np.degrees(p)
    b = 90. - np.degrees(t)
    lon = l
    lat = b
    t_in = np.radians(90. - lat)
    p_in = np.radians(lon)
    
 
    t_out, p_out = rot(t_in, p_in)
    
    lon_out = np.degrees(p_out)
    lat_out1 = 90. - np.degrees(t_out1)
    ra = lon_out1
    dec = lat_out1
    return ra, dec

def G2C2(nside):
    npix = hp.pixelfunc.nside2npix(nside)
    ipix =  np.arange(npix)
    t,p = hp.pixelfunc.pix2ang(nside,ipix, nest = True)
    rot = hp.rotator.Rotator(coord=['G', 'C'])
    #(lon,lat)=(l,b)
    l = np.degrees(p)
    b = 90. - np.degrees(t)
    lon = l
    lat = b
    t_in = np.radians(90. - lat)
    p_in = np.radians(lon)
    
 
    t_out1, p_out1 = rot(t_in[0:len(t_in)/2], p_in[0:len(p_in)/2])
    t_out2, p_out2 = rot(t_in[len(t_in)/2:len(t_in)], p_in[len(p_in)/2:len(p_in)])



    lon_out1 = np.degrees(p_out1)
    lon_out2 = np.degrees(p_out2)
    lat_out1 = 90. - np.degrees(t_out1)
    lat_out2 = 90. - np.degrees(t_out2)
    ra1 = lon_out1
    dec1 = lat_out1
    ra2 = lon_out2
    dec2 = lat_out2
    return ra1, dec1, ra2, dec2


#np.append

fits_file = fits.open('./COM_CMB_IQU-smica_1024_R2.02_full.fits')
#mapp = healpy.fitsfunc.read_map(fits_file[1], field = (0,1,2,3,4,5))
#hope = healpy.sphtfunc.map2alm(mapp)
#m143 = healpy.fitsfunc.read_map(fits_file[1], field = (0,1,2))
I,Q,U,tmask,pmask = healpy.fitsfunc.read_map(fits_file[1], field = (0,1,2,3,4), nest = True)

I = I*10**6
Q = Q*10**6
U = U*10**6

'''
ipix_200_ang = healpy.pix2ang(1024,200, nest = True)
nside = 16
npix = hp.pixelfunc.nside2npix(nside)
ipix =  np.arange(npix)
t,p = hp.pixelfunc.pix2ang(nside,ipix, nest = True)
#t,p = healpy.pix2ang(1024,200, nest = True)

#Going to (theta, phi) to Galactic coordinates (l,b)
rot = hp.rotator.Rotator(coord=['G', 'C'])
#(lon,lat)=(lb,b)
l = np.degrees(p)
b = 90. - np.degrees(t)
lon = l
lat = b
t_in = np.radians(90. - lat)
p_in = np.radians(lon)
    
 
t_out, p_out = rot(t_in, p_in)
lon_out = np.degrees(p_out)
lat_out = 90. - np.degrees(t_out)
ra = lon_out
dec = lat_out
'''
ra, dec = G2C(1024)

#ra_f1, dec_f1, ra_f2, dec_f2 = G2C(16384)

'''
radec_16384 = { 'ra' : ra_f, 'dec' : dec_f }
with open("radec_16384.pkl", "wb") as infile:
    pickle.dump(radec_16384, infile) 
'''




#Going from Galactic coordinates (l,b) to celestial coordinates (RA, DEC)
 


#The return object is too difficult to manipulate
'''
gc = SkyCoord(l=l*u.degree,b=b*u.degree,frame = 'galactic')
ec = gc.fk5
'''

#atm, this is bs
'''
#alpha is RA and delta is DEC
delta = np.degrees(np.arcsin(np.cos(np.deg2rad(b))*np.sin(np.deg2rad(l-33))*np.sin(np.deg2rad(62.6))+np.sin(np.deg2rad(b))*np.cos(np.deg2rad(62.6))))

for i in range(len(delta)):
    if delta[i] == 90:
        delta[i] = 90+1E-30
    elif delta[i] == 270:
        delta[i] = 270+1E-30

alpha2 = np.degrees(np.arccos((np.cos(np.deg2rad(b))*np.cos(np.deg2rad(l-33)))/np.cos(np.deg2rad(delta))))+282.25
#alpha = np.degrees(np.arcsin((np.cos(np.deg2rad(b))*np.sin(np.deg2rad(l-33))*np.cos(np.deg2rad(62.6))-np.sin(np.deg2rad(b))*np.sin(np.deg2rad(62.6)))/np.cos(np.deg2rad(delta))))+282.25
'''
embed()
'''
cl, alm= healpy.sphtfunc.anafast(m143, alm = True)
bl = healpy.sphtfunc.gauss_beam((5*np.pi)/(60*180),lmax=3*1024-1)

cl_sky_tt = cl[0]/(bl*bl)
cl_sky_tb = cl[1]/(bl*bl)

ell = np.arange(len(cl[0]))
Dl_tt = ell*(ell+1)*cl[0]/(2*np.pi*(1.0E-6)**2)
Dl_ee = ell*(ell+1)*cl[1]/(2*np.pi*(1.0E-6)**2)
Dl_bb = ell*(ell+1)*cl[2]/(2*np.pi*(1.0E-6)**2)
Dl_te = ell*(ell+1)*cl[3]/(2*np.pi*(1.0E-6)**2)
#embed()
plt.figure(1)
plt.plot(ell,Dl_tt)
plt.xlabel('ell'); plt.ylabel('d_l^tt'); plt.grid()

plt.figure(2)
plt.plot(ell,Dl_ee)
plt.xlabel('ell'); plt.ylabel('d_l^ee'); plt.grid()



plt.figure(3)
plt.plot(ell,Dl_te)
plt.xlabel('ell'); plt.ylabel('d_l^te'); plt.grid()

plt.figure(4)
plt.plot(ell,bl)
plt.xlabel('ell'); plt.ylabel('b_l'); plt.grid()

plt.show()


#embed()
'''

