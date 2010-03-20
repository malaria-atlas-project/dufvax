import tables as tb
import numpy as np
import map_utils
import os
from mpl_toolkits import basemap
from dufvax import modis_covariates, glob_channels, cmph_covariates
modis_covariates = ['raw-data.elevation.geographic.world.version-5', 'daytime-land-temp.annual-amplitude.geographic.world.2001-to-2006', 'daytime-land-temp.mean.geographic.world.2001-to-2006', 'daytime-land-temp.biannual-amplitude.geographic.world.2001-to-2006', 'daytime-land-temp.triannual-amplitude.geographic.world.2001-to-2006']

for res in [1,2,5]:
    try:
        os.mkdir('%ik-covariates'%res)
    except OSError:
        pass

# Parameters

modis_missing = 0
glob_missing = 210

modis_res = (21600, 43200)
lon = np.linspace(-180,180,modis_res[1])
lat = np.linspace(-90,90,modis_res[0])
# lon_min, lon_max, lat_min, lat_max = (-19, 52, 0, 24)
# lon_min, lon_max, lat_min, lat_max = (-19, 52, 8, 37)
# lon_min, lon_max, lat_min, lat_max = (-19, 52, 5, 20)
lon_min, lon_max, lat_min, lat_max = (-19, 13, 42, 40)

# Subset the rasters
lon_min_i, lon_max_i = ((np.array([lon_min, lon_max])+180.)/360.*modis_res[1]).astype('int')
lat_min_i, lat_max_i = ((np.array([lat_min, lat_max])+90.)/180.*modis_res[0]).astype('int')

def subset_and_writeout(hf_in, fname, thin, maskval, binfn=lambda x:x):
    print 'Subsetting for %s'%fname
    for res in [1,2,5]:
        hf_out = tb.openFile(os.path.join('%ik-covariates'%res,fname.replace('-','_').replace('.','_')+'.hdf5'),'w')
        hf_out.createArray('/','lon',lon[lon_min_i:lon_max_i:res])
        hf_out.createArray('/','lat',lat[lat_min_i:lat_max_i:res])
    
        d = hf_in.root.data[(hf_in.root.data.shape[0]-lat_max_i*thin):\
                            (hf_in.root.data.shape[0]-lat_min_i*thin):\
                            thin, 
                            lon_min_i*thin:\
                            lon_max_i*thin:\
                            thin]
                            
        d = map_utils.grid_convert(map_utils.grid_convert(d,'y-x+','x+y+')[::res,::res], 'x+y+','y-x+')

        hf_out.createCArray('/','data',atom=tb.FloatAtom(),shape=d.shape,filters=tb.Filters(complevel=1,complib='zlib'))
        hf_out.createCArray('/','mask',atom=tb.BoolAtom(),shape=d.shape,filters=tb.Filters(complevel=1,complib='zlib'))
        hf_out.root.data.attrs.view = 'y-x+'


        hf_out.root.data[:]=binfn(d)
        hf_out.root.mask[:]=d==maskval

        hf_out.close()

for m in modis_covariates:
    hf = tb.openFile('/Volumes/data/MODIS-hdf5/%s.hdf5'%m)
    subset_and_writeout(hf, '%s'%m, 1, modis_missing, lambda x:(x-x.min())/x.std())
    hf.close()

for cmph in cmph_covariates:
    print 'Subsetting for %s'%cmph
    lon_,lat_,data = map_utils.CRU_extract('/Volumes/data','CMORPH/%s'%cmph, zip=False)
    lon_.sort()
    lat_.sort()
    # data = map_utils.interp_geodata(lon_, lat_, data, lon[lon_min_i:lon_max_i], lat[lon_min_i:lon_max_i])
    data = map_utils.grid_convert(basemap.interp(map_utils.grid_convert(data,'y-x+','y+x+'), lon_, lat_, *np.meshgrid(lon[lon_min_i:lon_max_i],lat[lat_min_i:lat_max_i])),'y+x+','x+y+')
    for res in [1,2,5]:
        hf_out = tb.openFile(os.path.join('%ik-covariates'%res,cmph.lower()+'.hdf5'),'w')
        hf_out.createArray('/','lon',lon[lon_min_i:lon_max_i][::res])
        hf_out.createArray('/','lat',lat[lat_min_i:lat_max_i][::res])

        d = map_utils.grid_convert(data[::res,::res], 'x+y+','y-x+')

        hf_out.createCArray('/','data',atom=tb.FloatAtom(),shape=d.shape,filters=tb.Filters(complevel=1,complib='zlib'))
        hf_out.createCArray('/','mask',atom=tb.BoolAtom(),shape=d.shape,filters=tb.Filters(complevel=1,complib='zlib'))
        hf_out.root.data.attrs.view = 'y-x+'

        hf_out.root.data[:]=d

        hf_out.close()

glob = tb.openFile('/Volumes/data/Globcover/Globcover.hdf5')
for c in glob_channels:
    subset_and_writeout(glob, 'globcover-channel-%i'%c, 3, glob_missing, lambda x:x==c)
glob.close()

# Reconcile the masks
print 'Finding the conservative mask'
for res in [1,2,5]:
    el = tb.openFile(os.path.join('%ik-covariates'%res,'raw-data.elevation.geographic.world.version-5'.replace('-','_').replace('.','_')+'.hdf5'))
    c11 = tb.openFile(os.path.join('%ik-covariates'%res,'globcover_channel_11'.replace('-','_').replace('.','_')+'.hdf5'))
    conservative_mask = el.root.mask[:]+c11.root.mask[:]
    el.close()
    c11.close()
    for fname in filter(lambda x: os.path.splitext(x)[1]=='.hdf5', os.listdir('%ik-covariates'%res)):
        print 'Remasking %s'%(os.path.join('%ik-covariates'%res, fname))
        hf = tb.openFile(os.path.join('%ik-covariates'%res, fname),'a')
        hf.root.mask[:] = conservative_mask
        hf.close()