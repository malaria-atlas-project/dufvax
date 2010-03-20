# Copyright (C) 2009 Anand Patil
# 
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

# FIXME: Need to extract urban, rural from asciis, Otherwise they'll be
# FIXME: NaN at some points.

import sys
from pylab import csv2rec, rec2csv
import numpy as np
import warnings

duffy_datafile, vivax_datafile = sys.argv[1:]
combined_datafile = duffy_datafile.split('.')[0]+'_+_'+vivax_datafile.split('.')[0]+'.csv'

duffy_data = csv2rec(duffy_datafile)
vivax_data = csv2rec(vivax_datafile)
n_duffy = len(duffy_data)
n_vivax = len(vivax_data)

duffy_nan = np.repeat(np.nan,n_duffy)
vivax_nan = np.repeat(np.nan,n_vivax)

tstart = vivax_data.yestart + (vivax_data.mostart-1)/12.
tend = vivax_data.yeend + (vivax_data.moend-1)/12.

weirdcols = ['lon','lat','t','vivax_pos','vivax_neg','n','datatype']
vivaxcols = ['lo_age','up_age','urban','rural',]
duffycols = ['genaa','genab','genbb','gen00','gena0','genb0','gena1','genb1','gen01','gen11',
            'pheab','phea','pheb','phe0','prom0','promab','aphea','aphe0','bpheb',
            'bphe0']

coldict = {}
coldict['t'] = np.concatenate((duffy_nan,(tstart+tend)/2.))
coldict['lon'] = np.concatenate((duffy_data.lon, vivax_data.lon))
coldict['lat'] = np.concatenate((duffy_data.lat, vivax_data.lat))
coldict['n'] = np.concatenate((duffy_data.n, vivax_data.pos+vivax_data.neg))
coldict['vivax_pos'] = np.concatenate((duffy_nan,vivax_data.pos))
coldict['vivax_neg'] = np.concatenate((duffy_nan,vivax_data.neg))
coldict['datatype'] = np.concatenate((duffy_data.datatype, np.repeat('vivax',n_vivax)))

for colname in vivaxcols:
    coldict[colname] = np.concatenate((duffy_nan, vivax_data[colname]))
    
for colname in duffycols:
    coldict[colname] = np.concatenate((duffy_data[colname], vivax_nan))

allcols = coldict.keys()
combined_data = np.rec.fromarrays([coldict[col] for col in allcols], names=allcols)

# FIXME: Do the Sahel instead.
def box_data(data, llcrnrlon, llcrnrlat, urcrnrlon, urcrnrlat):
    indicator = (data.lon>llcrnrlon)*(data.lon<urcrnrlon)*(data.lat>llcrnrlat)*(data.lat<urcrnrlat)
    return data[np.where(indicator)]
    

# Write out
# warnings.warn('Boxing')
# combined_data = combined_data[np.where((combined_data.lon>-19)*(combined_data.lon<54)*(combined_data.lat>0))]
# combined_data = box_data(combined_data, 31.5, 11.5, 64, 32)
rec2csv(combined_data, combined_datafile)