from tables import *
from map_utils import *
from pylab import *
from mpl_toolkits import mplot3d, basemap
from matplotlib import cm

lon,lat,phe0,t = import_raster('phe0_mean','.')
lon,lat,vivax,type = import_raster('vivax_mean','.')
b = basemap.Basemap(lon.min(),lat.min(),lon.max(),lat.max())

d = csv2rec('/Volumes/Storage/generic-tests/dufvax/africa_only/jointdb-nosimp-input-data.csv')
d = d[np.where(d.datatype=='vivax')]
d_orig = d
cutoff = 30
d = d[np.where(d.vivax_pos+d.vivax_neg>cutoff)]

def d2p(d):
    return d.vivax_pos/(d.vivax_pos+d.vivax_neg)

def is_eastaf(rec):
    return rec.lon > 20 and rec.lat > -5 and rec.lon < 46 and rec.lat < 17
    
def is_madagascar(rec):
    return rec.lon > 41 and rec.lat > -25 and rec.lat < -10
    
def is_westaf(rec):
    return rec.lon < 11

def is_southaf(rec):
    return rec.lat < -7 and not is_madagascar(rec)
    
east, mad, west, south = map(lambda f, d=d: np.rec.fromrecords(filter(f, d), d.dtype), [is_eastaf, is_madagascar, is_westaf, is_southaf])
colors = ('r','g','b','w')


close('all')
figure(figsize=(12,6))
subplot(1,2,1)
b.imshow(grid_convert(phe0,'y-x+','y+x+'),cmap=cm.jet)
title('phe0, mean')
colorbar()


subplot(1,2,2)
b.imshow(grid_convert(vivax,'y-x+','y+x+'),cmap=cm.jet)
title('vivax, mean')
colorbar()
savefig('map-comparison.pdf',interpolation='nearest')

figure(figsize=(12,6))
subplot(1,2,1)
for l, col in zip([east,mad,west,south],colors):
    b.plot(l.lon, l.lat, col+'.', markersize=2, alpha=.5)
title('vivax data, n>%i'%cutoff)
b.drawcoastlines()

matplotlib.rcParams['xtick.labelsize']=6
matplotlib.rcParams['ytick.labelsize']=6
for i, l, t, c in zip([3,4,7,8],[east,mad,west,south],['East Africa','Madagascar','West Africa','Southern Africa'], colors):
    s=subplot(2,4,i)
    
    hist(d2p(l), color=c)
    title(t, fontsize=10)
savefig('vivax-data-locs.pdf',interpolation='nearest')

ax = mplot3d.Axes3D(pl.figure())
ax.scatter(d.lon, d.lat, d2p(d))
title('vivax data, n>%i'%cutoff)