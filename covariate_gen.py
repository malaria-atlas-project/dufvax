import tables as tb
import numpy as np
import map_utils
import os

# Parameters
modis_covariates = ['raw-data.elevation.geographic.world.version-5','daytime-land-temp.annual-amplitude.geographic.world.2001-to-2006','daytime-land-temp.triannual-amplitude.geographic.world.2001-to-2006','daytime-land-temp.biannual-amplitude.geographic.world.2001-to-2006']
glob_channels = [11,14,20,30,40,60,110,120,130,140,150,160,170,180,200]

modis_missing = 0
glob_missing = 210

modis_res = (21600, 43200)
lon = np.linspace(-180,180,modis_res[1])
lat = np.linspace(-90,90,modis_res[0])
lon_min, lon_max, lat_min, lat_max = (-19, 52, 0, 24)

# Subset the rasters
lon_min_i, lon_max_i = ((np.array([lon_min, lon_max])+180.)/360.*modis_res[1]).astype('int')
lat_min_i, lat_max_i = ((np.array([lat_min, lat_max])+90.)/180.*modis_res[0]).astype('int')

def subset_and_writeout(hf_in, fname, thin, maskval, binfn=lambda x:x):
    print 'Subsetting for %s'%fname
    hf_out = tb.openFile(fname.replace('-','_').replace('.','_')+'.hdf5','w')
    hf_out.createArray('/','lon',lon[lon_min_i:lon_max_i])
    hf_out.createArray('/','lat',lat[lat_min_i:lat_max_i])
    hf_out.createCArray('/','data',atom=tb.FloatAtom(),shape=(lat_max_i-lat_min_i,lon_max_i-lon_min_i),filters=tb.Filters(complevel=1,complib='zlib'))
    hf_out.createCArray('/','mask',atom=tb.BoolAtom(),shape=(lat_max_i-lat_min_i,lon_max_i-lon_min_i),filters=tb.Filters(complevel=1,complib='zlib'))
    hf_out.root.data.attrs.view = 'y-x+'
    
    d = hf_in.root.data[(hf_in.root.data.shape[0]-lat_max_i*thin):\
                        (hf_in.root.data.shape[0]-lat_min_i*thin):\
                        thin, 
                        lon_min_i*thin:\
                        lon_max_i*thin:\
                        thin]

    hf_out.root.data[:]=binfn(d)
    hf_out.root.mask[:]=d==maskval
    hf_out.close()

for m in modis_covariates:
    hf = tb.openFile('/Volumes/data/MODIS-hdf5/%s.hdf5'%m)
    subset_and_writeout(hf, '%s'%m, 1, modis_missing, lambda x:(x-x.min())/x.std())
    hf.close()

glob = tb.openFile('/Volumes/data/Globcover/Globcover.hdf5')
for c in glob_channels:
    subset_and_writeout(glob, 'globcover-channel-%i'%c, 3, glob_missing, lambda x:x==c)
glob.close()

# Reconcile the masks
print 'Finding the conservative mask'
el = tb.openFile('raw-data.elevation.geographic.world.version-5'.replace('-','_').replace('.','_')+'.hdf5')
c11 = tb.openFile('globcover-channel-11'.replace('-','_').replace('.','_')+'.hdf5')
conservative_mask = el.root.mask[:]+c11.root.mask[:]
el.close()
c11.close()

for fname in filter(lambda x: os.path.splitext(x)[1]=='.hdf5', os.listdir('.')):
    print 'Remasking %s'%fname
    hf = tb.openFile(fname,'a')
    hf.root.mask[:] = conservative_mask
    hf.close()