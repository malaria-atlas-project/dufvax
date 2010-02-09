import tables as tb
import numpy as np
import map_utils
from pylab import csv2rec, rec2csv
import os
import sys

lon_min, lon_max, lat_min, lat_max = (-19, 52, 0, 24)

data_in = csv2rec(sys.argv[1])
data_box = data_in[np.where((data_in.lon>=lon_min)*(data_in.lon<=lon_max)*(data_in.lat>=lat_min)*(data_in.lat<=lat_max))]

cols = dict([(key,data_box[key]) for key in data_box.dtype.names])
cols.pop('urban')
cols.pop('rural')
for fname in filter(lambda x: os.path.splitext(x)[1]=='.hdf5', os.listdir('.')):
    print 'Evaluating %s'%fname
    colname = os.path.splitext(fname)[0]
    hf = tb.openFile(fname,'a')
    lon = np.linspace(-180,180,hf.root.data.shape[1])
    lat = np.linspace(-90,90,hf.root.data.shape[0])
    
    cols[colname] = map_utils.interp_geodata(lon,lat,hf.root.data[:],cols['lon'],cols['lat'],hf.root.mask[:],order=0)
    
    hf.close()
    
keys = cols.keys()
data_out = np.rec.fromarrays([cols[k] for k in keys], names=keys)
rec2csv(data_out, os.path.splitext(data_in)[0]+'_with_covariates.csv')