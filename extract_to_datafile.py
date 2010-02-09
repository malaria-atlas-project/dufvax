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

def mode(a):
    vals = list(set(a))
    counts = [(a==v).sum() for v in vals]
    return np.argmin(counts)

def nan_callback(lon_old, lat_old, data, lon_new, lat_new, order):
    lon_ind = np.argmin(np.abs(np.subtract.outer(lon_old, lon_new)), axis=0)
    lat_ind = np.argmin(np.abs(np.subtract.outer(lat_old, lat_new)), axis=0)
    out = lat_new*0
    for i in xrange(len(lon_new)):
        lai, loi = lat_ind[i], lon_ind[i]
        if data.mask[lai, loi]:
            for d in xrange(10):
                if True-np.all(data.mask[lai-d:lai+d,loi-d:loi+d]):
                    out[i] = mode(data.data[lai-d:lai+d,loi-d:loi+d][np.where(True-data.mask[lai-d:lai+d,loi-d:loi+d])])
                    break
        else:
            out[i] = data[lai,loi]
    if np.any(np.isnan(out)):
        raise ValueError
    return out

for fname in filter(lambda x: os.path.splitext(x)[1]=='.hdf5', os.listdir('.')):
    print 'Evaluating %s'%fname
    colname = os.path.splitext(fname)[0]
    hf = tb.openFile(fname,'a')
    
    cols[colname] = map_utils.interp_geodata(hf.root.lon[:],hf.root.lat[:],hf.root.data[:],cols['lon'],cols['lat'],hf.root.mask[:],order=0,nan_handler=nan_callback)
    
    hf.close()
    
keys = cols.keys()
data_out = np.rec.fromarrays([cols[k] for k in keys], names=keys)
rec2csv(data_out, os.path.splitext(sys.argv[1])[0]+'_with_covariates.csv')