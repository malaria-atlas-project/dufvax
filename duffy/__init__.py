# from mcmc import *
from model import *
from generic_mbg import invlogit, FieldStepper
from cut_geographic import cut_geographic, hemisphere
import duffy
import pymc as pm
import os
root = os.path.split(duffy.__file__)[0]
pm.gp.cov_funs.cov_utils.mod_search_path.append(root)

 
# Stuff mandated by the new map_utils standard
 
cut_matern = pm.gp.cov_utils.covariance_wrapper('matern', 'pymc.gp.cov_funs.isotropic_cov_funs', {'diff_degree': 'The degree of differentiability of realizations.'}, 'cut_geographic', 'cg')
 
diag_safe = True
f_name = 'eps_p_f'
x_name = 'data_mesh'
f_has_nugget = True
nugget_name = 'V'
metadata_keys = ['fi','ti','ui']
postproc = invlogit
step_method_orders = {'f':(FieldStepper, )}
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