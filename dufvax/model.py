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

def covariance_submodel(suffix, ra, mesh, covariate_keys, ui, fname, temporal=False):
    """
    A small function that creates the mean and covariance object
    of the random field.
    """
    
    # The partial sill.
    amp = pm.Exponential('amp_%s'%suffix, .1, value=9.)
    
    # The range parameter. Units are RADIANS. 
    # 1 radian = the radius of the earth, about 6378.1 km
    # scale = pm.Exponential('scale', 1./.08, value=.08)
    
    scale = pm.Exponential('scale_%s'%suffix, 5, value=.5)
    # scale_shift = pm.Exponential('scale_shift_%s'%suffix, .1, value=.08)
    # scale = pm.Lambda('scale_%s'%suffix,lambda s=scale_shift: s+.01)
    scale_in_km = scale*6378.1
    
    # This parameter controls the degree of differentiability of the field.
    diff_degree = pm.Uniform('diff_degree_%s'%suffix, .5, 3, value=1, observed=True)
    
    # The nugget variance.
    V = pm.Exponential('V_%s'%suffix, 1, value=1.)
    
    @pm.potential
    def V_bound(V=V):
        if V<.1:
            return -np.inf
        else:
            return 0
    
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

        @pm.deterministic(trace=True,name='C_%s'%suffix)
        def C(amp=amp,scale=scale,inc=inc,ecc=ecc,scale_t=scale_t, t_lim_corr=t_lim_corr, sin_frac=sin_frac, diff_degree=diff_degree, covfacs=covfacs, covariate_keys=covariate_keys, ra=ra, mesh=mesh, ui=ui):
            facdict = dict([(k,1.e2*covfacs[i]) for i,k in enumerate(covariate_keys)])
            facdict['m'] = 1.e6
            eval_fun = CovarianceWithCovariates(my_st, fname, covariate_keys, ui, fac=facdict, ra=ra)
            return pm.gp.FullRankCovariance(eval_fun, amp=amp, scale=scale, inc=inc, ecc=ecc,st=scale_t, sd=diff_degree, tlc=t_lim_corr, sf = sin_frac)
                                            
    else:
        # Create the covariance & its evaluation at the data locations.
        @pm.deterministic(trace=True,name='C_%s'%suffix)
        def C(amp=amp, scale=scale, diff_degree=diff_degree, covariate_keys=covariate_keys, ra=ra, mesh=mesh, ui=ui):
            eval_fun = CovarianceWithCovariates(strip_time(pm.gp.matern.geo_rad), fname, covariate_keys, ui, fac=1.e4, ra=ra)
            return pm.gp.FullRankCovariance(eval_fun, amp=amp, scale=scale, diff_degree=diff_degree)
    
    # Create the mean function    
    @pm.deterministic(trace=True, name='M_%s'%suffix)
    def M():
        return pm.gp.Mean(pm.gp.zero_fn)
    
    # Create the GP submodel    
    sp_sub = pm.gp.GPSubmodel('sp_sub_%s'%suffix,M,C,mesh)
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
            g_freqs[hfk[i]+hfk[j]] = lambda pb, p0, p1, i=i, j=j: 2 * np.asscalar(hfv[i](pb,p0,p1) * hfv[j](pb,p0,p1))
        else:
            g_freqs[hfk[i]*2] = lambda pb, p0, p1, i=i: np.asscalar(hfv[i](pb,p0,p1))**2
            
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
    duffy_data_mesh, duffy_logp_mesh, duffy_fi, duffy_ui, duffy_ti = uniquify_tol(disttol,ttol,lon,lat)
    duffy_data_mesh = np.hstack((duffy_data_mesh, np.atleast_2d(t).T))
    duffy_logp_mesh = np.hstack((duffy_logp_mesh, np.atleast_2d(t[duffy_ui]).T))
    vivax_data_mesh, vivax_logp_mesh, vivax_fi, vivax_ui, vivax_ti = uniquify_tol(disttol,ttol,lon[where_vivax],lat[where_vivax],t[where_vivax])
    
    duffy_data_locs = duffy_data_mesh[:,:2]
    vivax_data_locs = vivax_data_mesh[:,:2]
    
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

    eps_p_f_d = {'b':[], '0':[], 'v':[]}
    p_d = {'b':[], '0': [], 'v': []}
    eps_p_f = {}
    eps_p_f_groups = []
    p_groups = []
    
    cur_group = {'b0': [],'v':[]}
    groupmap = []
    
    def make_data_group(cur_group):
        sl = {'b': duffy_fi[cur_group['b0']], '0': duffy_fi[cur_group['b0']], 'v': vivax_fi[cur_group['v']]}
        ge = {}
        gp= {}
        fs = {}

        for k in ['b','0','v']:
            if len(sl[k])==0:
                ge[k] = None
                gp[k] = None
                continue
            fs[k] = spatial_vars[k]['sp_sub'].f_eval[sl[k]]
            
            eps_p_f_d[k].append(pm.Normal('eps_p_f%s_%i'%(k,len(eps_p_f_d[k])), fs[k], tau[k], value=np.random.normal(size=len(sl[k])), trace=False))
            p_d[k].append(pm.Lambda('p%s_%i'%(k,len(eps_p_f_d[k])),lambda lt=eps_p_f_d[k][-1]: invlogit(np.atleast_1d(lt)),trace=False))
            ge[k] = eps_p_f_d[k][-1] 
            gp[k] = p_d[k][-1]

        if len(sl['v'])>0:
            duffy_locs_here = set(map(tuple, duffy_logp_mesh[fs['b'].parents['index']][:,:2]))
            vivax_locs_here = set(map(tuple, vivax_logp_mesh[fs['v'].parents['index']][:,:2]))
            if not duffy_locs_here.issuperset(vivax_locs_here):
                raise RuntimeError
            
        eps_p_f_groups.append(ge)
        p_groups.append(gp)
            
    for i_b0,loc_ in enumerate(duffy_data_locs):

        cur_group['b0'].append(i_b0)
        groupmap.append({'groupnum': len(eps_p_f_groups), 'b0_index': len(cur_group['b0'])-1, 'v_index': None})

        where_eq = np.where((vivax_data_locs==loc_).prod(axis=1))

        if len(where_eq[0])>0:
            cur_group['v'].extend(list(where_eq[0]))
            # vivax_data_locs.remove(cur_group['v'][-1])
            groupmap[-1]['v_index']=len(cur_group['v'])-1

        if len(cur_group['v'])+2*len(cur_group['b0'])>=grainsize:
            make_data_group(cur_group)
            cur_group = {'b0': [],'v': []}
    
    make_data_group(cur_group)
    
    for k in ['b','0','v']:
        # The fields plus the nugget
        eps_p_f[k] = pm.Lambda('eps_p_f%s'%k, lambda eps_p_f_d=eps_p_f_d[k]: np.hstack(eps_p_f_d))

    # The likelihoods.
    data_d = []    
    
    warnings.warn('Not using age correction')
    # junk, splreps = age_corr_likelihoods(lo_age[where_vivax], up_age[where_vivax], vivax_pos[where_vivax], vivax_neg[where_vivax], 10000, np.arange(.01,1.,.01), a_pred, P_trace, S_trace, F_trace)
    # for i in xrange(len(splreps)):
    #     splreps[i] = list(splreps[i])
    splreps = [None]*len(where_vivax[0])
    
    for i in xrange(len(n)):

        groupnum, b0_index, v_index = groupmap[i]['groupnum'], groupmap[i]['b0_index'], groupmap[i]['v_index']
        
        # See duffy/doc/model.tex for explanations of the likelihoods.
        pb,p0 = p_groups[groupnum]['b'][b0_index], p_groups[groupnum]['0'][b0_index]
        pv = None
        
        if datatype[i]=='prom':
            cur_obs = [prom0[i], promab[i]]
            # Need to have either b and 0 or a and 1 on both chromosomes
            p = pm.Lambda('p_%i'%i, lambda pb=pb, p0=p0, p1=p1: (pb*p0+(1-pb)*p1)**2, trace=False)
            n = np.sum(cur_obs)
            data_d.append(pm.Binomial('data_%i'%i, p=p, n=n, value=prom0[i], observed=True))
            
        elif datatype[i]=='aphe':
            cur_obs = [aphea[i], aphe0[i]]
            n = np.sum(cur_obs)
            # Need to have (a and not 1) on either chromosome, or not (not (a and not 1) on both chromosomes)
            p = pm.Lambda('p_%i'%i, lambda pb=pb, p0=p0, p1=p1: 1-(1-(1-pb)*(1-p1))**2, trace=False)
            data_d.append(pm.Binomial('data_%i'%i, p=p, n=n, value=aphea[i], observed=True))
            
        elif datatype[i]=='bphe':
            cur_obs = [bpheb[i], bphe0[i]]
            n = np.sum(cur_obs)
            # Need to have (b and not 0) on either chromosome
            p = pm.Lambda('p_%i'%i, lambda pb=pb, p0=p0, p1=p1: 1-(1-pb*(1-p0))**2, trace=False)
            data_d.append(pm.Binomial('data_%i'%i, p=p, n=n, value=aphea[i], observed=True))            
            
        elif datatype[i]=='phe':
            cur_obs = np.array([pheab[i],phea[i],pheb[i],phe0[i]])
            n = np.sum(cur_obs)
            p = pm.Lambda('p_%i'%i, lambda pb=pb, p0=p0, p1=p1: np.array([\
                g_freqs['ab'](pb,p0,p1),
                g_freqs['a0'](pb,p0,p1)+g_freqs['a1'](pb,p0,p1)+g_freqs['aa'](pb,p0,p1),
                g_freqs['b0'](pb,p0,p1)+g_freqs['b1'](pb,p0,p1)+g_freqs['bb'](pb,p0,p1),
                g_freqs['00'](pb,p0,p1)+g_freqs['01'](pb,p0,p1)+g_freqs['11'](pb,p0,p1)]), trace=False)
            np.testing.assert_almost_equal(p.value.sum(), 1)
            data_d.append(pm.Multinomial('data_%i'%i, p=p, n=n, value=cur_obs, observed=True))
            
        elif datatype[i]=='gen':
            cur_obs = np.array([genaa[i],genab[i],gena0[i],gena1[i],genbb[i],genb0[i],genb1[i],gen00[i],gen01[i],gen11[i]])
            n = np.sum(cur_obs)
            p = pm.Lambda('p_%i'%i, lambda pb=pb, p0=p0, p1=p1, g_freqs=g_freqs: \
                np.array([g_freqs[key](pb,p0,p1) for key in ['aa','ab','a0','a1','bb','b0','b1','00','01','11']]), trace=False)
            np.testing.assert_almost_equal(p.value.sum(), 1)
            data_d.append(pm.Multinomial('data_%i'%i, p=p, n=n, value=cur_obs, observed=True))
        
        elif datatype[i]=='vivax':
            # Since the vivax 'p' uses a different indexing system,
            # figure out which element of vivax 'p' to grab to correspond
            # to the i'th row of the datafile.
            pv = p_groups[groupnum]['v'][v_index]
            i_vivax = 0
            cur_obs = np.array([vivax_pos[i], vivax_neg[i]])
            
            pphe0 = pm.Lambda('pphe0_%i'%i, lambda pb=pb, p0=p0, p1=p1: (g_freqs['00'](pb,p0,p1)+g_freqs['01'](pb,p0,p1)+g_freqs['11'](pb,p0,p1)), trace=False)
            p = pm.Lambda('p_%i'%i, lambda pphe0=pphe0, pv=pv: pv*(1-pphe0), trace=False)
            try:
                warnings.warn('Not using age correction')
                @pm.observed
                @pm.stochastic(name='data_%i'%i,dtype=np.int)
                def d_now(value = vivax_pos[i], splrep = splreps[i_vivax], p = p, n = np.sum(cur_obs)):
                    return pm.binomial_like(x=value, n=n, p=p)
                    # return interp.splev(p, splrep)
            except ValueError:
                raise ValueError, 'Log-likelihood is nan at chunk %i'%i
            data_d.append(d_now)
            
        if np.any(np.isnan(cur_obs)):
            raise ValueError
    
    return locals()
