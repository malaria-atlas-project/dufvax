# from mcmc import *
disttol = 0./6378.
ttol = 0./12

import tables as tb
import numpy as np
import agecorr

modis_covariates = ['raw_data_elevation_geographic_world_version_5','daytime_land_temp_mean_geographic_world_2001_to_2006','daytime_land_temp_annual_amplitude_geographic_world_2001_to_2006','daytime_land_temp_triannual_amplitude_geographic_world_2001_to_2006','daytime_land_temp_biannual_amplitude_geographic_world_2001_to_2006']
# glob_channels = [11,14,20,30,40,60,110,120,130,140,150,160,170,180,200]
glob_channels = []
cmph_covariates = ['CMPH50A%i'%i for i in range(4)]
# cmph_covariates = []
covariate_names = modis_covariates + map(lambda n: 'globcover_channel_%i'%n, glob_channels) + cmph_covariates+['africa']

a_pred = a_pred = np.hstack((np.arange(15), np.arange(15,75,5), [100]))
try:
    age_pr_file = tb.openFile('pr-vivax')
    age_dist_file = tb.openFile('age-dist-vivax')

    age_pr_trace = age_pr_file.root.chain0.PyMCsamples.cols
    age_dist_trace = age_dist_file.root.chain0.PyMCsamples.cols
    P_trace = age_pr_trace.P_pred[:]
    S_trace = age_dist_trace.S_pred[:]
    F_trace = age_pr_trace.F_pred[:]
    age_pr_file.close()
    age_dist_file.close()

    two_ten_factors = agecorr.two_ten_factors(10000, P_trace, S_trace, F_trace)
except IOError:
    print 'Could not open age-pr files'
    P_trace, S_trace, F_trace = [None]*3

from model import *
from generic_mbg import FieldStepper
from pymc import thread_partition_array
from pymc.gp import GPEvaluationGibbs
import dufvax
from postproc_utils import *
import pymc as pm
import os
root = os.path.split(dufvax.__file__)[0]
pm.gp.cov_funs.cov_utils.mod_search_path.append(root)


def check_data(input):
    
    # Make sure there are no 'nan's.
    required_columns = {'phe': ['pheab','phea','pheb','phe0'],
                        'prom': ['prom0', 'promab'],
                        'aphe': ['aphea', 'aphe0'],
                        'bphe': ['bpheb', 'bphe0'],
                        'gen': ['genaa', 'genab', 'gena0', 'gena1', 'genbb', 'genb0', 'genb1', 'gen00', 'gen01', 'gen11'],
                        'vivax': ['vivax_pos', 'vivax_neg', 't', 'lo_age', 'up_age']}
    for datatype in ['phe','prom','aphe','bphe','gen','vivax']:
        this_data = input[np.where(input.datatype==datatype)]
        for c in required_columns[datatype]:
            if np.any(np.isnan(this_data[c])) or np.any(this_data[c]<0):
                raise ValueError, 'Datatype %s has nans or negs in col %s'%(datatype,c)
                
    if np.any([np.isnan(input[k]) for k in ['lon','lat']]):
        raise ValueError, 'Some nans in %s'%k
    
    # Column-specific checks
    def testcol(predicate, col):
        where_fail = np.where(predicate(input[col]))
        if len(where_fail[0])>0:
            raise ValueError, 'Test %s fails. %s \nFailure at rows %s'%(predicate.__name__, predicate.__doc__, where_fail[0]+1)

    n_vivax = np.sum(input.datatype=='vivax')

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
        return True-((t[-n_vivax:]>=1985) + (t[-n_vivax:]<=2010))
    testcol(duffytimecheck,'t')

    def dtypecheck(datatype):
        """Makes sure all datatypes are recognized."""
        return True-((datatype=='gen')+(datatype=='prom')+(datatype=='aphe')+(datatype=='bphe')+(datatype=='phe')+(datatype=='vivax'))
    testcol(dtypecheck,'datatype')

    def ncheck(n):
        """Makes sure n>0 and not nan"""
        return (n<0)+np.isnan(n)
    testcol(ncheck,'n')
 
nugget_labels = {'sp_sub_b': 'V_b', 'sp_sub_0': 'V_0', 'sp_sub_v': 'V_v'}
obs_labels = {'sp_sub_b':'eps_p_fb','sp_sub_0':'eps_p_f0', 'sp_sub_v': 'eps_p_fv'}

def phe0(sp_sub_b, sp_sub_0, sp_sub_v, p1):
    cmin, cmax = thread_partition_array(sp_sub_b)
    out = sp_sub_b.copy('F')     
    pm.map_noreturn(phe0_postproc, [(out, sp_sub_0, p1, cmin[i], cmax[i]) for i in xrange(len(cmax))])
    return out

def gena(sp_sub_b, sp_sub_0, sp_sub_v, p1):
    cmin, cmax = thread_partition_array(sp_sub_b)        
    out = sp_sub_b.copy('F')         
    pm.map_noreturn(gena_postproc, [(out, sp_sub_0, p1, cmin[i], cmax[i]) for i in xrange(len(cmax))])
    return out
    
def genb(sp_sub_b, sp_sub_0, sp_sub_v):
    cmin, cmax = thread_partition_array(sp_sub_b)        
    out = sp_sub_b.copy('F')         
    pm.map_noreturn(genb_postproc, [(out, sp_sub_0, cmin[i], cmax[i]) for i in xrange(len(cmax))])
    return out
    
def gen0(sp_sub_b, sp_sub_0, sp_sub_v):
    cmin, cmax = thread_partition_array(sp_sub_b)        
    out = sp_sub_b.copy('F')         
    pm.map_noreturn(gen0_postproc, [(out, sp_sub_0, cmin[i], cmax[i]) for i in xrange(len(cmax))])
    return out
    
def vivax(sp_sub_b, sp_sub_0, sp_sub_v, p1):
    cmin, cmax = thread_partition_array(sp_sub_b)
    out = sp_sub_b.copy('F')     
    # ttf = two_ten_factors[np.random.randint(len(two_ten_factors))]
    ttf = 1
    pm.map_noreturn(vivax_postproc, [(out, sp_sub_0, sp_sub_v, p1, ttf, cmin[i], cmax[i]) for i in xrange(len(cmax))])
    # pm.map_noreturn(vivax_postproc, [(out, sp_sub_0, sp_sub_v, p1, 1, cmin[i], cmax[i]) for i in xrange(len(cmax))])
    return out
    
    
map_postproc = [phe0, gena, genb, gen0, vivax]
# map_postproc = [gen0]

def validate_postproc(**non_cov_columns):
    """
    Don't know what to do here yet.
    """
    raise NotImplementedError
    
metadata_keys = ['disttol','ttol']

def mcmc_init(M):
    for k in ['b','0']:
        M.use_step_method(GPEvaluationGibbs, M.sp_sub[k], M.V[k], M.eps_p_f_d[k], ti=duffy_ti)
    M.use_step_method(GPEvaluationGibbs, M.sp_sub['v'], M.V['v'], M.eps_p_f_d['v'], ti=vivax_ti)
    
    for k in ['b','0','v']:
        for epf in M.eps_p_f_d[k]:
            M.use_step_method(pm.AdaptiveMetropolis, epf)
            
    M.assign_step_methods()
    for sm in M.step_method_dict.itervalues():
        for smm in sm:
            smm._tuning_info=[]
            smm._state = []
            

non_cov_columns = { 'n': 'int',
                    'datatype': 'str',
                    'genaa': 'float',
                    'genab': 'float',
                    'genbb': 'float',
                    'gen00': 'float',
                    'gena0': 'float',
                    'genb0': 'float',
                    'gena1': 'float',
                    'genb1': 'float',
                    'gen01': 'float',
                    'gen11': 'float',
                    'pheab': 'float',
                    'phea': 'float',
                    'pheb': 'float',
                    'phe0': 'float',
                    'prom0': 'float',
                    'promab': 'float',
                    'aphea': 'float',
                    'aphe0': 'float',
                    'bpheb': 'float',
                    'bphe0': 'float',
                    'vivax_pos': 'float',
                    'vivax_neg': 'float',
                    'lo_age': 'float',
                    'up_age': 'float'}