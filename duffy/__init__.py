# from mcmc import *
from model import *
from generic_mbg import invlogit, FieldStepper, fast_inplace_mul, fast_inplace_square
from cut_geographic import cut_geographic, hemisphere
import duffy
import pymc as pm
import os
root = os.path.split(duffy.__file__)[0]
pm.gp.cov_funs.cov_utils.mod_search_path.append(root)

# Stuff mandated by the new map_utils standard
cut_matern = pm.gp.cov_utils.covariance_wrapper('matern', 'pymc.gp.cov_funs.isotropic_cov_funs', {'diff_degree': 'The degree of differentiability of realizations.'}, 'cut_geographic', 'cg')
 
diags_safe = {'eps_p_fb': True, 'eps_p_f0': True}
f_labels = ['eps_p_fb', 'eps_p_f0']
x_label = 'data_mesh'
fs_have_nugget = {'eps_p_fb': True, 'eps_p_f0': True}
nugget_labels = {'eps_p_fb': 'V_b', 'eps_p_f0': 'V_0'}
M_labels = {'eps_p_fb': 'M_b', 'eps_p_f0': 'M_0'}
C_labels = {'eps_p_fb': 'C_b', 'eps_p_f0': 'C_0'}
covariate_pertenencies = {'eps_p_fb': [], 'eps_p_f0': []}

def map_postproc(eps_p_fb, eps_p_f0):
    """
    Returns probability of Duffy negativity from two random fields giving mutation frequencies.
    Fast and threaded.
    """
    pb = invlogit(eps_p_fb)
    p0 = invlogit(eps_p_f0)
    pb = fast_inplace_mul(pb,p0)
    print 'OK!'
    pb = fast_inplace_square(pb)
    return pb

def validate_postproc(**non_cov_columns):
    """
    Don't know what to do here yet.
    """
    raise NotImplementedError
    
metadata_keys = ['fi','ti','ui']

def mcmc_init(M):
    M.use_step_method(FieldStepper, M.fb, M.V_b, M.C_eval_b, M.M_eval_b, M.logp_mesh, M.eps_p_fb, M.ti)
    M.use_step_method(FieldStepper, M.f0, M.V_0, M.C_eval_0, M.M_eval_0, M.logp_mesh, M.eps_p_f0, M.ti)

non_cov_columns = {'africa': 'int',
                    'n': 'int',
                    'datatype': 'str',
                    'genaa': 'float',
                    'genab': 'float',
                    'genbb': 'float',
                    'gen00': 'float',
                    'gena0': 'float',
                    'genb0': 'float',
                    'gfga': 'float',
                    'gfgb': 'float',
                    'gfg0': 'float',
                    'pheab': 'float',
                    'phea': 'float',
                    'pheb': 'float',
                    'phe0': 'float',
                    'pos0': 'float',
                    'negab': 'float',
                    'aphea': 'float',
                    'aphe0': 'float',
                    'gfpa': 'float',
                    'gfpb': 'float',
                    'gfp0': 'float',
                    'gfpb0': 'float'}