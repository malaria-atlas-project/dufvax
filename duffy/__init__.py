# from mcmc import *
from model import *
from cut_geographic import cut_geographic, hemisphere
import duffy
from postproc_utils import *
import pymc as pm
from pymc import thread_partition_array
import numpy as np
import os
root = os.path.split(duffy.__file__)[0]
pm.gp.cov_funs.cov_utils.mod_search_path.append(root)

cut_matern = pm.gp.cov_utils.covariance_wrapper('matern', 'pymc.gp.cov_funs.isotropic_cov_funs', {'diff_degree': 'The degree of differentiability of realizations.'}, 'cut_geographic', 'cg')
 
nugget_labels = {'sp_sub_b': 'V_b', 'sp_sub_0': 'V_0'}
obs_labels = {'sp_sub_b':'eps_p_fb','sp_sub_0':'eps_p_f0'}

def check_data(ra):
    pass

def phe0(sp_sub_b, sp_sub_0, p1):
    cmin, cmax = thread_partition_array(sp_sub_b)
    out = sp_sub_b.copy('F')     
    pm.map_noreturn(phe0_postproc, [(out, sp_sub_0, p1, cmin[i], cmax[i]) for i in xrange(len(cmax))])
    return out

def gena(sp_sub_b, sp_sub_0, p1):
    cmin, cmax = thread_partition_array(sp_sub_b)        
    out = sp_sub_b.copy('F')         
    pm.map_noreturn(gena_postproc, [(out, sp_sub_0, p1, cmin[i], cmax[i]) for i in xrange(len(cmax))])
    return out
    
def genb(sp_sub_b, sp_sub_0):
    cmin, cmax = thread_partition_array(sp_sub_b)        
    out = sp_sub_b.copy('F')         
    pm.map_noreturn(genb_postproc, [(out, sp_sub_0, cmin[i], cmax[i]) for i in xrange(len(cmax))])
    return out
    
def gen0(sp_sub_b, sp_sub_0):
    cmin, cmax = thread_partition_array(sp_sub_b)        
    out = sp_sub_b.copy('F')         
    pm.map_noreturn(gen0_postproc, [(out, sp_sub_0, cmin[i], cmax[i]) for i in xrange(len(cmax))])
    return out
    
map_postproc = [phe0, gena, genb, gen0]

def validate_postproc(**non_cov_columns):
    """
    Don't know what to do here yet.
    """
    raise NotImplementedError
    
metadata_keys = []

def mcmc_init(M):
    M.use_step_method(pm.gp.GPEvaluationGibbs, M.sp_sub_b, M.V_b, M.eps_p_fb)
    M.use_step_method(pm.gp.GPEvaluationGibbs, M.sp_sub_0, M.V_0, M.eps_p_f0)
    for tup in zip(M.eps_p_fb_d, M.eps_p_f0_d):
        M.use_step_method(pm.AdaptiveMetropolis, tup)
        # for v in tup:
        #     M.use_step_method(pm.Metropolis, v)
    # scalar_stochastics = []
    # for v in M.stochastics:
    #     if v not in M.eps_p_fb_d and v not in M.eps_p_f0_d and np.squeeze(v.value).shape == ():
    #         scalar_stochastics.append(v)
    # M.use_step_method(pm.AdaptiveMetropolis, scalar_stochastics)


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
                    'bphe0': 'float'}