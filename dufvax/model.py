# Author: Anand Patil
# Date: 6 Feb 2009
# License: Creative Commons BY-NC-SA
####################################

# The idea: There are two mutations involved, one (Nr33) converting Fya to Fyb and one (Nr125) 
# silencing expression. The latter mutation tends to occur only with the former, but the converse 
# is not true.
# 
# The model for an individual chromosomal genotype is that the first mutation is a spatially-
# correlated random field, but the second occurs independently with some probability in Africa
# and another probability outside Africa.
# 
# There are two types of datapoints: 
#     - one testing individuals for phenotype (a-b-), meaning both chromosomes have the silencing
#       mutation.
#     - one testing individuals for expression of Fya on either chromosome.
# It's easy to make a likelihood model for either of these, just a bit complicated.
# The maps we'll eventually want to make will be of (a-b-) frequency, meaining the postprocessing
# function will need to close on model variables. The generic package doesn't currently support this.


import numpy as np
import pymc as pm
import gc
from map_utils import *
from generic_mbg import *
from st_cov_fun import *
import generic_mbg
import warnings
# from agecorr import age_corr_likelihoods
from dufvax import P_trace, S_trace, F_trace, a_pred
from scipy import interpolate as interp
import scipy
from pylab import csv2rec
from theano import tensor as T
import theano

__all__ = ['make_model']

class strip_time(object):
    def __init__(self, f):
        self.f = f
    def __call__(self, x, y, *args, **kwds):
        return self.f(x[:,:2],y[:,:2],*args,**kwds)
    def diag_call(self, x, *args, **kwds):
        return self.f.diag_call(x[:,:2],*args,**kwds) 

continent = 'Africa'

# Prior parameters specified by Simon, Pete and Andy
# Af_scale_params = {'mu': -2.54, 'tau': 1.42, 'alpha': -.015}
Af_scale_params = {'alpha': -0.015, 'mu': -2, 'tau': 5.21}
Af_amp_params = {'mu': .0535, 'tau': 1.79, 'alpha': 3.21}

Am_scale_params = {'mu': -2.58, 'tau': 1.27, 'alpha': .051}
Am_amp_params = {'mu': .607, 'tau': .809, 'alpha': -1.17}

As_scale_params = {'mu': -2.97, 'tau': 1.75, 'alpha': -.143}
As_amp_params = {'mu': .0535, 'tau': 1.79, 'alpha': 3.21}

# Poor man's sparsification
if continent == 'Americas':
    scale_params = Am_scale_params
    amp_params = Am_amp_params
    disttol = 0/6378.
    ttol = 0
elif continent == 'Asia':
    scale_params = As_scale_params
    amp_params = As_amp_params    
    disttol = 5./6378.
    ttol = 1./12
elif continent == 'Africa':
    scale_params = Af_scale_params
    amp_params = Af_amp_params    
    disttol = 0./6378.
    ttol = 0./12
else:
    scale_params = Af_scale_params
    amp_params = Af_amp_params
    disttol = 0./6378.
    ttol = 0.

# TODO:
# Get approximate evidence by using the INLA trick.
# Grid the priors, Gibbs sample.

def approximate_gaussian_full_conditional(M_blocks,Q_blocks,p,x_blocks):
    like_vals = None
    like_covs = None
    Mc_blocks = None
    Qc_blocks = None
    return like_vals, like_covs, Mc_blocks, Qc_blocks

def gaussian_evidence(like_vals, like_covs, M_blocks, Q_blocks, Mc_blocks, Qc_blocks):
    return 0
    
def covariance_submodel(suffix, ra, mesh, covariate_keys, ui, fname, temporal=False):
    """
    A small function that creates the mean and covariance object
    of the random field.
    """
    
    if not temporal:
        mesh = mesh[:,:2]
    
    # Subjective skew-normal prior on amp (the partial sill, tau) in log-space.
    # Parameters are passed in in manual_MCMC_supervisor.
    log_amp = pm.SkewNormal('log_amp_%s'%suffix,value=amp_params['mu'],observed=True,**amp_params)
    amp = pm.Lambda('amp_%s'%suffix, lambda log_amp = log_amp: np.exp(log_amp))

    # Subjective skew-normal prior on scale (the range, phi_x) in log-space.
    log_scale = pm.SkewNormal('log_scale_%s'%suffix,value=-1,observed=True,**scale_params)
    scale = pm.Lambda('scale_%s'%suffix, lambda log_scale = log_scale: np.exp(log_scale))
    
    # scale_shift = pm.Exponential('scale_shift_%s'%suffix, .1, value=.08)
    # scale = pm.Lambda('scale_%s'%suffix,lambda s=scale_shift: s+.01)
    scale_in_km = scale*6378.1
    
    # This parameter controls the degree of differentiability of the field.
    diff_degree = pm.Uniform('diff_degree_%s'%suffix, .1, 3, value=.5, observed=True)
    
    # The nugget variance.
    V = pm.Gamma('V_%s'%suffix, 4, 40, value=.1,observed=True)
    
    if temporal:
        inc = 0
        ecc = 0
        # Exponential prior on the temporal scale/range, phi_t. Standard one-over-x
        # doesn't work bc data aren't strong enough to prevent collapse to zero.
        scale_t = pm.Exponential('scale_t_%s'%suffix, 5,value=1,observed=True)

        # Uniform prior on limiting correlation far in the future or past.
        t_lim_corr = pm.Uniform('t_lim_corr_%s'%suffix,0,1,value=.8,observed=True)

        # # Uniform prior on sinusoidal fraction in temporal variogram
        sin_frac = pm.Uniform('sin_frac_%s'%suffix,0,1,value=.1,observed=True)
        
        @pm.potential(name='st_constraint_%s'%suffix)
        def st_constraint(sd=.5, sf=sin_frac, tlc=t_lim_corr):    
            if -sd >= 1./(-sf*(1-tlc)+tlc):
                return -np.Inf
            else:
                return 0.
        
        # covfac_pow = pm.Exponential('covfac_pow_%s'%suffix, .1, value=.5)
        covfac_pow = 0
        
        covariate_names = covariate_keys
        @pm.observed
        @pm.stochastic(name='log_covfacs_%s'%suffix)
        def log_covfacs(value=-np.ones(len(covariate_names))*.01, k=covfac_pow):
            """Induced prior on covfacs is p(x)=(1+k)(1-x)^k, x\in [0,1]"""
            if np.all(value<0):
                return np.sum(value+np.log(1+k)+k*np.log(1-np.exp(value)))
            else:
                return -np.inf

        # covfacs are uniformly distributed on [0,1]        
        covfacs = pm.Lambda('covfacs_%s'%suffix, lambda x=log_covfacs: np.exp(x))

        @pm.deterministic(trace=False,name='C_%s'%suffix)
        def C(amp=amp,scale=scale,inc=inc,ecc=ecc,scale_t=scale_t, t_lim_corr=t_lim_corr, sin_frac=sin_frac, diff_degree=diff_degree, covfacs=covfacs, covariate_keys=covariate_keys, ra=ra, mesh=mesh, ui=ui):
            facdict = dict([(k,1.e2*covfacs[i]) for i,k in enumerate(covariate_keys)])
            facdict['m'] = 1.e6
            eval_fun = CovarianceWithCovariates(my_st, fname, covariate_keys, ui, fac=facdict, ra=ra)
            return pm.gp.FullRankCovariance(eval_fun, amp=amp, scale=scale, inc=inc, ecc=ecc,st=scale_t, sd=diff_degree, tlc=t_lim_corr, sf = sin_frac)
               
    else:
        # Create the covariance & its evaluation at the data locations.
        @pm.deterministic(trace=False,name='C_%s'%suffix)
        def C(amp=amp, scale=scale, diff_degree=diff_degree, covariate_keys=covariate_keys, ra=ra, mesh=mesh, ui=ui):
            eval_fun = CovarianceWithCovariates(strip_time(pm.gp.matern.geo_rad), fname, covariate_keys, ui, fac=1.e4, ra=ra)
            return pm.gp.FullRankCovariance(eval_fun, amp=amp, scale=scale, diff_degree=diff_degree)
    
    # Create the mean function    
    @pm.deterministic(trace=False, name='M_%s'%suffix)
    def M():
        return pm.gp.Mean(pm.gp.zero_fn)
    
    @pm.deterministic(trace=False,name='C_eval_plus_nugget_%s'%suffix)
    def C_eval_plus_nugget(C=C,V=V,mesh=mesh):
        out=C(mesh,mesh)
        return out + np.eye(out.shape[0])*V
            
    @pm.deterministic(trace=False,name='Q_%s'%suffix)
    def Q(C_eval_plus_nugget=C_eval_plus_nugget):
        return C_eval_plus_nugget.I
    
    return locals()
        
# =========================
# = Haplotype frequencies =
# =========================
xb = T.dvector('xb')
x0 = T.dvector('x0')
xv = T.dvector('xv')
p1 = T.dscalar('p1')

def theano_invlogit(x):
    return T.exp(x)/(T.exp(x)+1)

pb = theano_invlogit(xb)
p0 = theano_invlogit(x0)
# FIXME: This should be the age-correction business.
pv = theano_invlogit(xv)

h_freqs = {'a': (1-pb)*(1-p1),
            'b': pb*(1-p0),
            '0': pb*p0,
            '1': (1-pb)*p1}
hfk = ['a','b','0','1']
hfv = [h_freqs[key] for key in hfk]

# ========================
# = Genotype frequencies =
# ========================
g_freqs = {}
for i in xrange(4):
    for j in xrange(i,4):
        if i != j:
            g_freqs[hfk[i]+hfk[j]] = 2 * hfv[i] * hfv[j]
        else:
            g_freqs[hfk[i]*2] = hfv[i]**2
        
g_freqs_fns = dict([(k,theano.function([pb,p0,p1], v)) for k,v in g_freqs.items()])
for i in xrange(1000):
    pb_,p0_,p1_ = np.random.random(size=3)
    np.testing.assert_almost_equal(np.sum([gfi([pb_],[p0_],p1_) for gfi in g_freqs_fns.values()]),1.)

p_prom = (pb*p0+(1-pb)*p1)**2
p_aphe = 1-(1-(1-pb)*(1-p1))**2
p_bphe = 1-(1-pb*(1-p0))**2
p_phe = [\
    g_freqs['ab'],
    g_freqs['a0']+g_freqs['a1']+g_freqs['aa'],
    g_freqs['b0']+g_freqs['b1']+g_freqs['bb'],
    g_freqs['00']+g_freqs['01']+g_freqs['11']]    
p_gen = [g_freqs[key] for key in ['aa','ab','a0','a1','bb','b0','b1','00','01','11']]
pphe0 = g_freqs['00']+g_freqs['01']+g_freqs['11']
p_vivax = pv*(1-pphe0)

def theano_binomial(k, n, p):
    return T.sum(k*T.log(p) + (n-k)*T.log(1-p))
    
def theano_multinomial(x,p):
    N = len(x)
    return T.sum(T.dot(T.log(p),np.asarray(x)))
    
def likelihood_expression_to_potential(name, expr, x_theano, x_pymc):
    expr_fn = theano.function(expr, x_theano)
    @pm.Potential(name=name)
    def pot(x=x_pymc, expr_fn=expr_fn):
        return expr_fn(*x)
    return pot

def zipmap(f, keys):
    return dict(zip(keys, map(f, keys)))

x_test = np.array([[10,12,2,4],[30,2,14,6],[3,45,26,3]])

#TODO: Cut both Duffy and Vivax    
def make_model(lon,lat,t,input_data,covariate_keys,n,datatype,
                genaa,genab,genbb,gen00,gena0,genb0,gena1,genb1,gen01,gen11,
                pheab,phea,pheb,
                phe0,prom0,promab,
                aphea,aphe0,
                bpheb,bphe0,
                vivax_pos,vivax_neg,
                lo_age, up_age,
                cpus=1):
    """
    This function is required by the generic MBG code.
    """
    ra = csv2rec(input_data)
    
    # Step method granularity    
    grainsize = 20
    
    where_vivax = np.where(datatype=='vivax')
    from dufvax import disttol, ttol
    
    # Duffy needs to be modelled everywhere Duffy or Vivax is observed.
    # Vivax only needs to be modelled where Vivax is observed.
    # Complication: Vivax can have multiple co-located observations at different times,
    # all corresponding to the same Duffy observation.
    data_mesh = generic_mbg.combine_st_inputs(lon,lat,t)
    
    data_mesh_dict = {'b': data_mesh, '0': data_mesh, 'v': data_mesh[where_vivax]}
        
    # Create the mean & its evaluation at the data locations.
    init_OK = False
        
    covariate_key_dict = {'v': set(covariate_keys), 'b': ['africa'], '0':[]}        
    temporal_dict = {'b': False, '0': False, 'v': True}
    
    p1_pymc = pm.Uniform('p1_pymc',0,.4,value=.1,observed=True)
    
    init_OK = False
    while not init_OK:
        try:
            spatial_vars = zipmap(lambda k: covariance_submodel(k, ra, data_mesh_dict[k], covariate_key_dict[k], np.arange(len(data_mesh)), input_data, temporal_dict[k]), ['b','0','v'])
            tau = zipmap(lambda k: 1./spatial_vars[k]['V'], ['b','0','v'])
        
            # Loop over data clusters, adding nugget and applying link function.
            init_OK = True
        except pm.ZeroProbability, msg:
            print 'Trying again: %s'%msg
            init_OK = False
            gc.collect()        
    
    V_b, V_0, V_v = [spatial_vars[k]['V'] for k in ['b','0','v']]    
        
    warnings.warn('Not using age correction')
    # junk, splreps = age_corr_likelihoods(lo_age[where_vivax], up_age[where_vivax], vivax_pos[where_vivax], vivax_neg[where_vivax], 10000, np.arange(.01,1.,.01), a_pred, P_trace, S_trace, F_trace)
    # for i in xrange(len(splreps)):
    #     splreps[i] = list(splreps[i])
    splreps = [None]*len(where_vivax[0])
    
    where_prom = np.where(datatype=='prom')
    cur_obs = np.array([prom0[where_prom], promab[where_prom]]).T
    cur_n = n[where_prom]
    np.testing.assert_equal(cur_n, np.sum(cur_obs,axis=1))
    # Need to have either b and 0 or a and 1 on both chromosomes    
    theano_likelihood_prom = theano_binomial(prom0[where_prom], cur_n, p_prom)
        
    where_aphe = np.where(datatype=='aphe')
    cur_obs = np.array([aphea[where_aphe], aphe0[where_aphe]]).T
    cur_n = n[where_aphe]
    np.testing.assert_equal(cur_n, np.sum(cur_obs, axis=1))
    # Need to have (a and not 1) on either chromosome, or not (not (a and not 1) on both chromosomes.
    theano_likelihood_aphe = theano_binomial(aphea[where_aphe], cur_n, p_aphe)
        
    where_bphe = np.where(datatype=='bphe')
    cur_n = n[where_bphe]
    cur_obs = np.array([bpheb[where_bphe], bphe0[where_bphe]]).T
    np.testing.assert_equal(cur_n, np.sum(cur_obs, axis=1))
    # Need to have (b and not 0) on either chromosome
    theano_likelihood_bphe = theano_binomial(bpheb[where_bphe], cur_n, p_bphe)
        
    where_phe = np.where(datatype=='phe')
    cur_obs = np.array([pheab[where_phe],
                        phea[where_phe],
                        pheb[where_phe],
                        phe0[where_phe]])
    cur_n = n[where_phe]
    np.testing.assert_equal(cur_n, np.sum(cur_obs, axis=0))
    theano_likelihood_phe = theano_multinomial(cur_obs, p_phe)
    
    where_gen = np.where(datatype=='gen')
    cur_n = n[where_gen]
    cur_obs = np.array([genaa[where_gen],
                        genab[where_gen],
                        gena0[where_gen],
                        gena1[where_gen],
                        genbb[where_gen],
                        genb0[where_gen],
                        genb1[where_gen],
                        gen00[where_gen],
                        gen01[where_gen],
                        gen11[where_gen]])
    np.testing.assert_equal(cur_n, np.sum(cur_obs,axis=0))
    theano_likelihood_gen = theano_multinomial(cur_obs, p_gen)
    
    # Now vivax.
    cur_obs = np.array([vivax_pos[where_vivax], vivax_neg[where_vivax]]).T
    cur_n = np.sum(cur_obs,axis=1)
    np.testing.assert_equal(cur_n, np.sum(cur_obs,axis=1))
    theano_likelihood_vivax = theano_binomial(vivax_pos[where_vivax], cur_n, p_vivax)
    
    theano_likelihood = theano_likelihood_prom + theano_likelihood_aphe + theano_likelihood_bphe + theano_likelihood_phe + theano_likelihood_gen + theano_likelihood_vivax
    
    @pm.deterministic(trace=False)
    def full_conditional_params(M_blocks=[spatial_vars[k]['M'] for k in ['b','0','v']],
                                Q_blocks=[spatial_vars[k]['Q'] for k in ['b','0','v']],
                                p = theano_likelihood,
                                x_blocks = [xb,x0,xv]):
        return approximate_gaussian_full_conditional(M_blocks,Q_blocks,p,x_blocks)
    
    @pm.potential
    def evidence(full_conditional_params=full_conditional_params,
                    M_blocks=[spatial_vars[k]['M'] for k in ['b','0','v']],
                    Q_blocks=[spatial_vars[k]['Q'] for k in ['b','0','v']]):
        like_vals, like_covs, Mc_blocks, Qc_blocks = full_conditional_params
        return gaussian_evidence(like_vals, like_covs, M_blocks, Q_blocks, Mc_blocks, Qc_blocks)
    
    if np.any(np.isnan(cur_obs)):
        raise ValueError
            
    
    return locals()
