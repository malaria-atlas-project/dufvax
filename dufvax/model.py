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
from pylab import csv2rec

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
    disttol = 5./6378.
    ttol = 1./12
else:
    scale_params = Af_scale_params
    amp_params = Af_amp_params
    disttol = 0./6378.
    ttol = 0.

def covariance_submodel(suffix, ra, mesh, covariate_keys, ui, fname, temporal=False):
    """
    A small function that creates the mean and covariance object
    of the random field.
    """
    
    # Subjective skew-normal prior on amp (the partial sill, tau) in log-space.
    # Parameters are passed in in manual_MCMC_supervisor.
    log_amp = pm.SkewNormal('log_amp_%s'%suffix,value=amp_params['mu'],**amp_params)
    amp = pm.Lambda('amp_%s'%suffix, lambda log_amp = log_amp: np.exp(log_amp))

    # Subjective skew-normal prior on scale (the range, phi_x) in log-space.
    log_scale = pm.SkewNormal('log_scale_%s'%suffix,value=-1,**scale_params)
    scale = pm.Lambda('scale_%s'%suffix, lambda log_scale = log_scale: np.exp(log_scale))
    
    # scale_shift = pm.Exponential('scale_shift_%s'%suffix, .1, value=.08)
    # scale = pm.Lambda('scale_%s'%suffix,lambda s=scale_shift: s+.01)
    scale_in_km = scale*6378.1
    
    # This parameter controls the degree of differentiability of the field.
    diff_degree = pm.Uniform('diff_degree_%s'%suffix, .1, 3, value=.5, observed=True)
    
    # The nugget variance.
    V = pm.Gamma('V_%s'%suffix, 4, 40, value=.1)
    
    if temporal:
        inc = 0
        ecc = 0
        # Exponential prior on the temporal scale/range, phi_t. Standard one-over-x
        # doesn't work bc data aren't strong enough to prevent collapse to zero.
        scale_t = pm.Exponential('scale_t_%s'%suffix, 5,value=1)

        # Uniform prior on limiting correlation far in the future or past.
        t_lim_corr = pm.Uniform('t_lim_corr_%s'%suffix,0,1,value=.8)

        # # Uniform prior on sinusoidal fraction in temporal variogram
        sin_frac = pm.Uniform('sin_frac_%s'%suffix,0,1,value=.1)
        
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
    
    # Create the GP submodel    
    sp_sub = pm.gp.GPSubmodel('sp_sub_%s'%suffix,M,C,mesh,tally_f=False)
    sp_sub.f.trace=False
    sp_sub.f_eval.value = sp_sub.f_eval.value - sp_sub.f_eval.value.mean()    
    
    return locals()
        
# =========================
# = Haplotype frequencies =
# =========================
h_freqs = {'a': lambda pb, p0, p1: (1-pb)*(1-p1),
            'b': lambda pb, p0, p1: pb*(1-p0),
            '0': lambda pb, p0, p1: pb*p0,
            '1': lambda pb, p0, p1: (1-pb)*p1}
hfk = ['a','b','0','1']
hfv = [h_freqs[key] for key in hfk]

# ========================
# = Genotype frequencies =
# ========================
g_freqs = {}
for i in xrange(4):
    for j in xrange(i,4):
        if i != j:
            g_freqs[hfk[i]+hfk[j]] = lambda pb, p0, p1, i=i, j=j: 2 * hfv[i](pb,p0,p1) * hfv[j](pb,p0,p1)
        else:
            g_freqs[hfk[i]*2] = lambda pb, p0, p1, i=i: hfv[i](pb,p0,p1)**2
            
for i in xrange(1000):
    pb,p0,p1 = np.random.random(size=3)
    np.testing.assert_almost_equal(np.sum([gfi(pb,p0,p1) for gfi in g_freqs.values()]),1.)

def zipmap(f, keys):
    return dict(zip(keys, map(f, keys)))

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
    print 'Uniquifying.'
    duffy_data_mesh, duffy_logp_mesh, duffy_fi, duffy_ui, duffy_ti = uniquify_tol(disttol,ttol,lon,lat)
    duffy_data_mesh = np.hstack((duffy_data_mesh, np.atleast_2d(t).T))
    duffy_logp_mesh = np.hstack((duffy_logp_mesh, np.atleast_2d(t[duffy_ui]).T))
    vivax_data_mesh, vivax_logp_mesh, vivax_fi, vivax_ui, vivax_ti = uniquify_tol(disttol,ttol,lon[where_vivax],lat[where_vivax],t[where_vivax])
    
    print 'Done uniquifying.'
    
    duffy_data_locs = map(tuple,duffy_data_mesh[:,:2])
    vivax_data_locs = map(tuple,vivax_data_mesh[:,:2])
    
    full_vivax_ui = np.arange(len(lon))[where_vivax][vivax_ui]

    # Create the mean & its evaluation at the data locations.
    init_OK = False
    
    # Probability of mutation in the promoter region, given that the other thing is a.
    p1 = pm.Uniform('p1', 0, .04, value=.01)
    
    covariate_key_dict = {'v': set(covariate_keys), 'b': ['africa'], '0':[]}
    ui_dict = {'v': full_vivax_ui, 'b': duffy_ui, '0': duffy_ui}
        
    logp_mesh_dict = {'b': duffy_logp_mesh, '0': duffy_logp_mesh, 'v': vivax_logp_mesh}
    temporal_dict = {'b': False, '0': False, 'v': True}
    
    init_OK = False
    while not init_OK:
        try:
            spatial_vars = zipmap(lambda k: covariance_submodel(k, ra, logp_mesh_dict[k], covariate_key_dict[k], ui_dict[k], input_data, temporal_dict[k]), ['b','0','v'])
            tau = zipmap(lambda k: 1./spatial_vars[k]['V'], ['b','0','v'])
        
            # Loop over data clusters, adding nugget and applying link function.
            init_OK = True
        except pm.ZeroProbability, msg:
            print 'Trying again: %s'%msg
            init_OK = False
            gc.collect()        

    sp_sub_b, sp_sub_0, sp_sub_v = [spatial_vars[k]['sp_sub'] for k in ['b','0','v']]
    V_b, V_0, V_v = [spatial_vars[k]['V'] for k in ['b','0','v']]

    eps_p_f = {}
    p = {}
    
    for k in ['b','0','v']:
        if k=='v':
            fi = vivax_fi
        else:
            fi = duffy_fi
        eps_p_f[k] = pm.Normal('eps_p_f_%s'%k, spatial_vars[k]['sp_sub'].f_eval[fi], tau[k], value=np.random.normal(size=len(fi)))
        p[k] = pm.InvLogit('p_%s'%k, eps_p_f[k], trace=False)
    
    pb, p0, pv = p['b'], p['0'], p['v']
    
    warnings.warn('Not using age correction')
    # junk, splreps = age_corr_likelihoods(lo_age[where_vivax], up_age[where_vivax], vivax_pos[where_vivax], vivax_neg[where_vivax], 10000, np.arange(.01,1.,.01), a_pred, P_trace, S_trace, F_trace)
    # for i in xrange(len(splreps)):
    #     splreps[i] = list(splreps[i])
    splreps = [None]*len(where_vivax[0])
    
    where_prom = np.where(datatype=='prom')
    cur_obs = np.array([prom0[where_prom], promab[where_prom]]).T
    # Need to have either b and 0 or a and 1 on both chromosomes
    p_prom = pm.Lambda('p_prom', lambda pb=pb[where_prom], p0=p0[where_prom], p1=p1: (pb*p0+(1-pb)*p1)**2, trace=False)
    n = np.sum(cur_obs,axis=1)
    data_prom = pm.Binomial('data_prom', p=p_prom, n=n, value=prom0[where_prom], observed=True)
        
    where_aphe = np.where(datatype=='aphe')
    cur_obs = np.array([aphea[where_aphe], aphe0[where_aphe]]).T
    n = np.sum(cur_obs, axis=1)
    # Need to have (a and not 1) on either chromosome, or not (not (a and not 1) on both chromosomes)
    p_aphe = pm.Lambda('p_aphe', lambda pb=pb[where_aphe], p0=p0[where_aphe], p1=p1: 1-(1-(1-pb)*(1-p1))**2, trace=False)
    data_aphe = pm.Binomial('data_aphe', p=p_aphe, n=n, value=aphea[where_aphe], observed=True)
        
    where_bphe = np.where(datatype=='bphe')
    cur_obs = np.array([bpheb[where_bphe], bphe0[where_bphe]]).T
    n = np.sum(cur_obs, axis=1)
    # Need to have (b and not 0) on either chromosome
    p_bphe = pm.Lambda('p_bphe', lambda pb=pb[where_bphe], p0=p0[where_bphe], p1=p1: 1-(1-pb*(1-p0))**2, trace=False)
    data_bphe = pm.Binomial('data_bphe', p=p_bphe, n=n, value=bpheb[where_bphe], observed=True)
        
    where_phe = np.where(datatype=='phe')
    cur_obs = np.array([pheab[where_phe],phea[where_phe],pheb[where_phe],phe0[where_phe]]).T
    n = np.sum(cur_obs, axis=1)
    p_phe = pm.Lambda('p_%i'%i, lambda pb=pb[where_phe], p0=p0[where_phe], p1=p1: np.array([\
        g_freqs['ab'](pb,p0,p1),
        g_freqs['a0'](pb,p0,p1)+g_freqs['a1'](pb,p0,p1)+g_freqs['aa'](pb,p0,p1),
        g_freqs['b0'](pb,p0,p1)+g_freqs['b1'](pb,p0,p1)+g_freqs['bb'](pb,p0,p1),
        g_freqs['00'](pb,p0,p1)+g_freqs['01'](pb,p0,p1)+g_freqs['11'](pb,p0,p1)]).T, trace=False)
    np.testing.assert_almost_equal(p_phe.value.sum(axis=1), 1)
    data_phe = pm.Multinomial('data_phe', p=p_phe, n=n, value=cur_obs, observed=True)    
    
    where_gen = np.where(datatype=='gen')
    cur_obs = np.array([genaa[where_gen],genab[where_gen],gena0[where_gen],gena1[where_gen],genbb[where_gen],genb0[where_gen],genb1[where_gen],gen00[where_gen],gen01[where_gen],gen11[where_gen]]).T
    n = np.sum(cur_obs,axis=1)
    p_gen = pm.Lambda('p_gen', lambda pb=pb[where_gen], p0=p0[where_gen], p1=p1, g_freqs=g_freqs: \
        np.array([g_freqs[key](pb,p0,p1) for key in ['aa','ab','a0','a1','bb','b0','b1','00','01','11']]).T, trace=False)
    np.testing.assert_almost_equal(p_gen.value.sum(axis=1), 1)
    data_gen = pm.Multinomial('data_gen', p=p_gen, n=n, value=cur_obs, observed=True)
    
    # Now vivax.
    cur_obs = np.array([vivax_pos[where_vivax], vivax_neg[where_vivax]]).T
    pphe0 = pm.Lambda('pphe0_%i'%i, lambda pb=pb[where_vivax], p0=p0[where_vivax], p1=p1: (g_freqs['00'](pb,p0,p1)+g_freqs['01'](pb,p0,p1)+g_freqs['11'](pb,p0,p1)), trace=False)
    p_vivax = pm.Lambda('p_vivax', lambda pphe0=pphe0, pv=pv: pv*(1-pphe0), trace=False)
    try:
        warnings.warn('Not using age correction')
        @pm.observed
        @pm.stochastic(dtype=np.int)
        def data_vivax(value = vivax_pos[where_vivax], splrep = None, p = p_vivax, n = np.sum(cur_obs,axis=1)):
            return pm.binomial_like(x=value, n=n, p=p)
    except ValueError:
        raise ValueError, 'Log-likelihood is nan for vivax.'
        
    if np.any(np.isnan(cur_obs)):
        raise ValueError
            
    
    return locals()
