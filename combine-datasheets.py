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

import sys
from pylab import csv2rec, rec2csv
import numpy as np

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
vivaxcols = ['pos','neg','lo_age','up_age','urban','rural',]
duffycols = ['genaa','genab','genbb','gen00','gena0','genb0','gena1','genb1','gen01','gen11',
            'africa','pheab','phea','pheb','phe0','prom0','promab','aphea','aphe0','bpheb',
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
    coldict[colname] = np.concatenate((vivax_data[colname], duffy_nan))
    
for colname in duffycols:
    coldict[colname] = np.concatenate((vivax_nan, duffy_data[colname]))

allcols = coldict.keys()
combined_data = np.rec.fromarrays([coldict[col] for col in allcols], names=allcols)


# Checks
def testcol(predicate, col):
    where_fail = np.where(predicate(combined_data[col]))
    if len(where_fail[0])>0:
        raise ValueError, 'Test %s fails. %s \nFailure at rows %s'%(predicate.__name__, predicate.__doc__, where_fail[0]+1)
        
def loncheck(lon):
    """Makes sure longitudes are between -180 and 180."""
    return np.abs(lon)>180. + np.isnan(lon)
testcol(loncheck,'lon')
    
def latcheck(lat):
    """Makes sure latitudes are between -90 and 90."""
    return np.abs(lat)>180. + np.isnan(lat)
testcol(latcheck,'lat')

def duffytimecheck(t):
    """Makes sure times are between 1985 and 2010"""
    return True-((t[n_vivax:]>=1985) + (t[n_vivax:]<=2010))
testcol(duffytimecheck,'t')
    
def dtypecheck(datatype):
    """Makes sure all datatypes are recognized."""
    return True-((datatype=='gen')+(datatype=='prom')+(datatype=='aphe')+(datatype=='bphe')+(datatype=='phe')+(datatype=='vivax'))
testcol(dtypecheck,'datatype')

def ncheck(n):
    """Makes sure n>0 and not nan"""
    return (n==0)+np.isnan(n)
testcol(ncheck,'n')


# Write out
rec2csv(combined_data, combined_datafile)