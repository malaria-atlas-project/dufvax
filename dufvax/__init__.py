# from mcmc import *
from model import *
from generic_mbg import FieldStepper
from pymc import thread_partition_array
from pymc.gp import GPEvaluationGibbs
import dufvax
from postproc_utils import *
import pymc as pm
import numpy as np
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
 
f_labels = ['eps_p_fb', 'eps_p_f0', 'eps_p_fv']
fs_have_nugget = {'eps_p_fb': True, 'eps_p_f0': True, 'eps_p_fv': True}
nugget_labels = {'eps_p_fb': 'V_b', 'eps_p_f0': 'V_0', 'eps_p_fv': 'V_v'}
M_labels = {'eps_p_fb': 'M_b', 'eps_p_f0': 'M_0', 'eps_p_fv': 'M_v'}
C_labels = {'eps_p_fb': 'C_b', 'eps_p_f0': 'C_0', 'eps_p_fv': 'C_v'}
x_labels = {'eps_p_fb': 'data_mesh', 'eps_p_f0': 'data_mesh', 'eps_p_fv': 'data_mesh'}
diags_safe = {'eps_p_fb': True, 'eps_p_f0': True, 'eps_p_fv': True}

def phe0(eps_p_fb, eps_p_f0, eps_p_fv, p1):
    cmin, cmax = thread_partition_array(eps_p_fb)
    out = eps_p_fb.copy('F')     
    pm.map_noreturn(phe0_postproc, [(out, eps_p_f0, p1, cmin[i], cmax[i]) for i in xrange(len(cmax))])
    return out

def gena(eps_p_fb, eps_p_f0, eps_p_fv, p1):
    cmin, cmax = thread_partition_array(eps_p_fb)        
    out = eps_p_fb.copy('F')         
    pm.map_noreturn(gena_postproc, [(out, eps_p_f0, p1, cmin[i], cmax[i]) for i in xrange(len(cmax))])
    return out
    
def genb(eps_p_fb, eps_p_f0, eps_p_fv):
    cmin, cmax = thread_partition_array(eps_p_fb)        
    out = eps_p_fb.copy('F')         
    pm.map_noreturn(genb_postproc, [(out, eps_p_f0, cmin[i], cmax[i]) for i in xrange(len(cmax))])
    return out
    
def gen0(eps_p_fb, eps_p_f0, eps_p_fv):
    cmin, cmax = thread_partition_array(eps_p_fb)        
    out = eps_p_fb.copy('F')         
    pm.map_noreturn(gen0_postproc, [(out, eps_p_f0, cmin[i], cmax[i]) for i in xrange(len(cmax))])
    return out
    
map_postproc = [phe0, gena, genb, gen0]

def validate_postproc(**non_cov_columns):
    """
    Don't know what to do here yet.
    """
    raise NotImplementedError
    
metadata_keys = []

def mcmc_init(M):
    for k in ['b','0','v']:
        M.use_step_method(GPEvaluationGibbs, M.sp_sub[k], M.V[k], M.eps_p_f[k])
    
    for k in ['b','0','v']:
        for epf in M.eps_p_f_d[k]:
            M.use_step_method(pm.AdaptiveMetropolis, epf)
            

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