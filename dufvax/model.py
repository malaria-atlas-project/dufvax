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
from agecorr import age_corr_likelihoods
from dufvax import P_trace, S_trace, F_trace, a_pred
from scipy import interpolate as interp

__all__ = ['make_model']

class strip_time(object):
    def __init__(self, f):
        self.f = f
    def __call__(self, x, y, *args, **kwds):
        return self.f(x[:,:2],y[:,:2],*args,**kwds)
    def diag_call(self, x, *args, **kwds):
        return self.f.diag_call(x[:,:2],*args,**kwds)
    

def covariance_submodel(suffix, mesh, covariate_values, temporal=False):
    """
    A small function that creates the mean and covariance object
    of the random field.
    """
    
    # The partial sill.
    amp = pm.Exponential('amp_%s'%suffix, .1, value=1.)
    
    # The range parameter. Units are RADIANS. 
    # 1 radian = the radius of the earth, about 6378.1 km
    # scale = pm.Exponential('scale', 1./.08, value=.08)
    
    scale = pm.Uniform('scale_%s'%suffix,.01,2,value=.08)
    # scale_shift = pm.Exponential('scale_shift_%s'%suffix, .1, value=.08)
    # scale = pm.Lambda('scale_%s'%suffix,lambda s=scale_shift: s+.01)
    scale_in_km = scale*6378.1
    
    # This parameter controls the degree of differentiability of the field.
    diff_degree = pm.Uniform('diff_degree_%s'%suffix, .5, 3, value=.5)
    
    # The nugget variance. Lower-bounded to preserve mixing.
    V = pm.Exponential('V_%s'%suffix, .1, value=1.)
    
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
        scale_t = pm.Exponential('scale_t_%s'%suffix, .01,value=.1)

        # Uniform prior on limiting correlation far in the future or past.
        t_lim_corr = pm.Uniform('t_lim_corr_%s'%suffix,0,1,value=.01)

        # # Uniform prior on sinusoidal fraction in temporal variogram
        sin_frac = pm.Uniform('sin_frac_%s'%suffix,0,1,value=.01)
        
        @pm.potential(name='st_constraint_%s'%suffix)
        def st_constraint(sd=.5, sf=sin_frac, tlc=t_lim_corr):    
            if -sd >= 1./(-sf*(1-tlc)+tlc):
                return -np.Inf
            else:
                return 0.
                
        @pm.deterministic(trace=True,name='C_%s'%suffix)
        def C(amp=amp,scale=scale,inc=inc,ecc=ecc,scale_t=scale_t, t_lim_corr=t_lim_corr, sin_frac=sin_frac, diff_degree=diff_degree):
            eval_fun = CovarianceWithCovariates(my_st, mesh, covariate_values)
            return pm.gp.FullRankCovariance(eval_fun, amp=amp, scale=scale, inc=inc, ecc=ecc,st=scale_t, sd=diff_degree,
                                            tlc=t_lim_corr, sf = sin_frac)
                                            
    else:
        # Create the covariance & its evaluation at the data locations.
        @pm.deterministic(trace=True,name='C_%s'%suffix)
        def C(amp=amp, scale=scale, diff_degree=diff_degree):
            eval_fun = CovarianceWithCovariates(strip_time(pm.gp.matern.geo_rad), mesh, covariate_values, fac=1e4)
            return pm.gp.FullRankCovariance(eval_fun, amp=amp, scale=scale, diff_degree=diff_degree)
    
    # Create the mean function    
    @pm.deterministic(trace=True, name='M_%s'%suffix)
    def M():
        return pm.gp.Mean(pm.gp.zero_fn)
    
    # Create the GP submodel    
    sp_sub = pm.gp.GPSubmodel('sp_sub_%s'%suffix,M,C,mesh)

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

def uniquify_tol(disttol, ttol, *cols):
    locs = [tuple([col[0] for col in cols])]
    fi = [0]
    ui = [0]
    dx = np.empty(1)
    for i in xrange(1,len(cols[0])):

        # If repeat location, add observation
        loc = np.array([col[i] for col in cols])
        for j in xrange(len(locs)):
            pm.gp.geo_rad(dx, np.atleast_2d(loc[:2]*np.pi/180.), np.atleast_2d(locs[j][:2]))
            if len(cols)>2:
                dt = np.abs(loc[2]-locs[j][2])
            else:
                dt = 0
            if dx[0]<=disttol and dt<=ttol:
                fi.append(j)
                break

        # Otherwise, new obs
        else:
            locs.append(loc)
            fi.append(max(fi)+1)
            ui.append(i)
    fi = np.array(fi)
    ti = [np.where(fi == i)[0] for i in xrange(max(fi)+1)]
    ui = np.asarray(ui)

    locs = np.array(locs)
    if len(cols)==3:
        data_mesh = combine_st_inputs(*cols)
        logp_mesh = combine_st_inputs(locs[:,0], locs[:,1], locs[:,2])
    else:
        data_mesh = combine_spatial_inputs(*cols)
        logp_mesh = combine_spatial_inputs(locs[:,0], locs[:,1])

    return data_mesh, logp_mesh, fi, ui, ti


def uniquify(*cols):

    locs = [tuple([col[0] for col in cols])]
    fi = [0]
    ui = [0]
    for i in xrange(1,len(cols[0])):
        
        # If repeat location, add observation
        loc = tuple([col[i] for col in cols])
        if loc in locs:
            fi.append(locs.index(loc))
            
        # Otherwise, new obs
        else:
            locs.append(loc)
            fi.append(max(fi)+1)
            ui.append(i)
    fi = np.array(fi)
    ti = [np.where(fi == i)[0] for i in xrange(max(fi)+1)]
    ui = np.asarray(ui)
    
    locs = np.array(locs)
    if len(cols)==3:
        data_mesh = combine_st_inputs(*cols)
        logp_mesh = combine_st_inputs(locs[:,0], locs[:,1], locs[:,2])
    else:
        data_mesh = combine_spatial_inputs(*cols)
        logp_mesh = combine_spatial_inputs(locs[:,0], locs[:,1])
        
    return data_mesh, logp_mesh, fi, ui, ti
    

#TODO: Cut both Duffy and Vivax    
def make_model(lon,lat,t,covariate_values,n,datatype,
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
    # Step method granularity    
    grainsize = 20
    
    where_vivax = np.where(datatype=='vivax')
    from dufvax import disttol, ttol
    
    
    # Rebind input variables for convenience
    # for dlab in ['genaa','genab','genbb','gen00','gena0','genb0','gena1','genb1','gen01','gen11','pheab','phea','pheb','phe0','prom0','promab','aphea','aphe0','bpheb','bphe0']:
    #     exec('%s=%s[where_duffy]'(%dlab,dlab))
    
    # Duffy needs to be modelled everywhere Duffy or Vivax is observed.
    # Vivax only needs to be modelled where Vivax is observed.
    # Complication: Vivax can have multiple co-located observations at different times,
    # all corresponding to the same Duffy observation.
    duffy_data_mesh, duffy_logp_mesh, duffy_fi, duffy_ui, duffy_ti = uniquify(lon,lat)
    duffy_data_mesh = np.hstack((duffy_data_mesh, np.atleast_2d(t).T))
    duffy_logp_mesh = np.hstack((duffy_logp_mesh, np.atleast_2d(t[duffy_ui]).T))
    vivax_data_mesh, vivax_logp_mesh, vivax_fi, vivax_ui, vivax_ti = uniquify(lon[where_vivax],lat[where_vivax],t[where_vivax])
    
    
    # Create the mean & its evaluation at the data locations.
    init_OK = False
    
    # Probability of mutation in the promoter region, given that the other thing is a.
    p1 = pm.Uniform('p1', 0, .04, value=.01)
    
    vivax_keys = set(covariate_values.keys())
    
    bigkeys = filter(lambda k: covariate_values[k].max()>10, covariate_values.keys())
    
    vivax_covariate_values = dict([(k,covariate_values[k][vivax_ui]) for k in vivax_keys])
    logp_mesh_dict = {'b': duffy_logp_mesh, '0': duffy_logp_mesh, 'v': vivax_logp_mesh}
    temporal_dict = {'b': False, '0': False, 'v': True}
    covariate_value_dict = {'b': {'globcover_channel_200': covariate_values['globcover_channel_200'][duffy_ui]},
                            '0': {},
                            'v': vivax_covariate_values}
    
    for k,v in covariate_value_dict.iteritems():
        if k.find('channel')>-1:
            print 'Hi!'
            values = set(v)
            print values
            nmin = np.inf
            for value in values:
                nmin = min(np.sum(v==value),nmin)
            if nmin < 100:
                warnings.warn('Not good representation for covariate %s'%key)
    
    while not init_OK:
        # try:
        spatial_vars = zipmap(lambda k: covariance_submodel(k, logp_mesh_dict[k], covariate_value_dict[k], temporal_dict[k]), ['b','0','v'])
        sp_sub = zipmap(lambda k: spatial_vars[k]['sp_sub'], ['b','0','v'])
        sp_sub_b, sp_sub_0, sp_sub_v = [sp_sub[k] for k in ['b','0','v']]
        V = zipmap(lambda k: spatial_vars[k]['V'], ['b','0','v'])
        V_b, V_0, V_v = [V[k] for k in ['b','0','v']]
        tau = zipmap(lambda k: 1./spatial_vars[k]['V'], ['b','0','v'])
        
        # Loop over data clusters, adding nugget and applying link function.
        f = zipmap(lambda k: spatial_vars[k]['sp_sub'].f_eval, ['b','0','v'])
        init_OK = True
    # except pm.ZeroProbability, msg:
    #     print 'Trying again: %s'%msg
    #     init_OK = False
    #     gc.collect()        

    eps_p_f_d = {'b':[], '0':[], 'v':[]}
    p_d = {'b':[], '0': [], 'v': []}
    eps_p_f = {}
        
    # Duffy eps_p_f's and p's, eval'ed everywhere.
    for k in ['b','0']:    
        for i in xrange(int(np.ceil(len(n)/float(grainsize)))):
            sl = slice(i*grainsize,(i+1)*grainsize,None)                
            if sl.stop>sl.start:
            
                this_f = f[k][duffy_fi[sl]]

                # Nuggeted field in this cluster
                eps_p_f_d[k].append(pm.Normal('eps_p_f%s_%i'%(k,i), this_f, tau[k], value=np.random.normal(size=np.shape(this_f.value)), trace=False))
        
                # The allele frequency
                p_d[k].append(pm.Lambda('p%s_%i'%(k,i),lambda lt=eps_p_f_d[k][-1]: invlogit(np.atleast_1d(lt)),trace=False))

        # The fields plus the nugget
        eps_p_f[k] = pm.Lambda('eps_p_f%s'%k, lambda eps_p_f_d=eps_p_f_d[k]: np.hstack(eps_p_f_d))
        
    # Vivax eps_p_f's and p's, only eval'ed on vivax points.
    for i in xrange(int(np.ceil(len(n[where_vivax])/float(grainsize)))):
        sl = slice(i*grainsize,(i+1)*grainsize,None)                
        if sl.stop>sl.start:
        
            this_f = f['v'][vivax_fi[sl]]

            # Nuggeted field in this cluster
            eps_p_f_d['v'].append(pm.Normal('eps_p_fv_%i'%i, this_f, tau['v'], value=np.random.normal(size=np.shape(this_f.value)), trace=False))
    
            # The allele frequency
            p_d['v'].append(pm.Lambda('p%s_%i'%(k,i),lambda lt=eps_p_f_d['v'][-1]: invlogit(np.atleast_1d(lt)),trace=False))

    # The fields plus the nugget
    eps_p_f['v'] = pm.Lambda('eps_p_fv', lambda eps_p_f_d=eps_p_f_d['v']: np.hstack(eps_p_f_d))    

    # The likelihoods.
    data_d = []    
    
    warnings.warn('Not using age correction')
    # junk, splreps = age_corr_likelihoods(lo_age[where_vivax], up_age[where_vivax], vivax_pos[where_vivax], vivax_neg[where_vivax], 10000, np.arange(.01,1.,.01), a_pred, P_trace, S_trace, F_trace)
    # for i in xrange(len(splreps)):
    #     splreps[i] = list(splreps[i])
    splreps = [None]*len(where_vivax[0])
    
    for i in xrange(len(n)):

        sl_ind = int(i/grainsize)
        sub_ind = i%grainsize
        
        if sl_ind == len(p_d['b']):
            break
        
        # See duffy/doc/model.tex for explanations of the likelihoods.
        pb,p0 = map(lambda k: p_d[k][sl_ind][sub_ind], ['b','0'])
        
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
            i_vivax = np.where(where_vivax[0]==i)[0][0]
            sl_ind_vivax = int(i_vivax/grainsize)
            sub_ind_vivax = i_vivax%grainsize
            pv = p_d['v'][sl_ind_vivax][sub_ind_vivax]
            
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
