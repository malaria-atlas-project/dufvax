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
try:
    # from agecorr import age_corr_likelihoods
    from dufvax import P_trace, S_trace, F_trace, a_pred
except:
    pass
from scipy import interpolate as interp
import scipy
from pylab import csv2rec
from theano import tensor as T
from theano import function as tfun
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
    
def covariance_submodel(suffix, ra, mesh, covariate_keys, ui, fname, temporal=False):
    """
    A small function that creates the mean and covariance object
    of the random field.
    """

    if not temporal:
        mesh = mesh[:,:2]

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
    sp_sub.temporal = temporal

    return locals()

def grads(expr, x, all_inputs):
    from copy import copy
    other_inputs = copy(all_inputs)
    other_inputs.remove(x)
    grad1 = T.grad(expr, x)
    grad2, _ = theano.scan(fn=lambda i, x, grad1, *other_inputs: T.grad(grad1[i], x)[i], sequences=T.arange(x.shape[0]), non_sequences=[x, grad1]+other_inputs)
    return [grad1, grad2]
    
class DufvaxStep(pm.AdaptiveMetropolis):
    def __init__(self, sp_sub, data_mesh, nugget, field_plus_nugget, theano_to_pymc_fpns, theano_likelihood, delay=1000, interval=200, scales=None):
        """
        field_plus_nugget should be a Theano variable.
        thean_to_pymc_fpns should be a map from Theano variables to PyMC variables.
        """
        from copy import copy

        self.sp_sub = sp_sub
        self.nugget = nugget
        self.theano_likelihood = theano_likelihood
        self.prior_params = sp_sub.f_eval.extended_parents
        self.theano_to_pymc_fpns = theano_to_pymc_fpns
        self.name_to_pymc_fpns = dict([(k.name, v) for k,v in theano_to_pymc_fpns.items()])
        self.data_mesh = data_mesh
        
        other_keys = copy(theano_to_pymc_fpns.keys())
        other_keys.remove(field_plus_nugget)
        
        self.likelihood_function = tfun([field_plus_nugget]+other_keys, theano_likelihood)
        grads_out = grads(theano_likelihood, field_plus_nugget, theano_to_pymc_fpns.keys())
        grad1 = T.grad(theano_likelihood, field_plus_nugget)
        self.grad1_func = tfun([field_plus_nugget]+other_keys, grad1)
        self.grad_function = tfun([field_plus_nugget]+other_keys, grads_out)        
        self.field_plus_nugget = theano_to_pymc_fpns[field_plus_nugget]
        
        @pm.deterministic(trace=False)
        def C_eval_plus_nugget(C=sp_sub.C,V=nugget,mesh=data_mesh):
            out=C(mesh,mesh)
            return np.asmatrix(out + np.eye(out.shape[0])*V)

        @pm.deterministic(trace=False)
        def Q(C_eval_plus_nugget=C_eval_plus_nugget):
            return C_eval_plus_nugget.I

        other_x={}
        for k in other_keys:
            other_x[k.name] = theano_to_pymc_fpns[k]
        @pm.deterministic(trace=False)
        def approximate_gaussian_full_conditional(M=sp_sub.M(data_mesh), Q=Q, gf=self.grad_function, g1f=self.grad1_func, other_x=other_x, tol=1.e-8, fpn=self.field_plus_nugget,g1=grad1,lf=self.likelihood_function,fpn_key=field_plus_nugget.name):

            from scipy import linalg

            x=fpn
            delta = x*0+np.inf
            while np.abs(delta).max() > tol:
                d1, d2 = gf(x,**other_x)
                
                like_prec = -d2
                like_var = d2+np.inf
                like_vals = d2
                where_nonzero = np.where(d2!=0)
                like_vals[where_nonzero] = -d1[where_nonzero]/d2[where_nonzero]+x[where_nonzero]
                like_var[where_nonzero] = 1/like_prec[where_nonzero]

                grad1_full = np.ravel(M*Q)+d1
                if np.any(np.isnan(d1)) or np.any(np.isnan(d2)):
                    print 'Got some nans, skipping'
                    return None, None, None, None
                Qc = np.asmatrix(Q+np.diag(like_prec))
                
                # import pylab as pl
                # xplot = np.linspace(-10,10,201)
                # xi = x.copy()
                # pl.clf()
                # j = np.argmax(d2)
                # yplot = []
                # for i in xrange(201):
                #     xi[j] = xplot[i]
                #     yplot.append(lf(xi,**other_x))
                # pl.plot(xplot, yplot)
                # pl.title(d2[j])
                
                # import pdb
                # pdb.set_trace()

                x_ = linalg.solve(Qc,grad1_full)
                delta = x_-x
                x=x+delta

            # return like_vals, like_vars, Mc, Qc
            return like_vals,like_var,x,Qc            

        @pm.deterministic(trace=False)
        def S_cond(agfc=approximate_gaussian_full_conditional):
            like_vals, like_vars, Mc, Qc = agfc
            if Qc is None:
                return None
            else:
                try:
                    # Sometimes Qc.I is PD but Qc is not, for some reason.
                    return np.linalg.cholesky(Qc.I)
                except np.linalg.LinAlgError:
                    return None
        
        @pm.deterministic(trace=False)
        def evidence(agfc=approximate_gaussian_full_conditional, M=sp_sub.M(data_mesh), C=C_eval_plus_nugget, S_cond=S_cond):
            if S_cond is None:
                return -np.inf
            else:
                like_vals, like_vars, Mc, Qc = agfc
                where_finite = np.where(True-np.isinf(like_vars))
                return pm.mv_normal_cov_like(like_vals[where_finite], M[where_finite], np.asarray(C[where_finite[0]][:,where_finite[0]])+np.diag(like_vars[where_finite]))

        self.Q = Q
        self.approximate_gaussian_full_conditional = approximate_gaussian_full_conditional
        self.evidence = evidence
        self.C_eval_plus_nugget = C_eval_plus_nugget

        pm.AdaptiveMetropolis.__init__(self, stochastic=list(self.prior_params))

    def _get_logp_plus_loglike(self):
        sum = pm.logp_of_set(self.prior_params) + self.evidence.value
        if self.verbose>2:
            print '\t' + self._id + ' Current log-likelihood plus current log-probability', sum
        return sum

    # Make get property for retrieving log-probability
    logp_plus_loglike = property(fget = _get_logp_plus_loglike, doc="The summed log-probability of all stochastic variables that depend on \n self.stochastics, and self.stochastics, but not the field or the nuggeted field.")

    def step(self):
        from copy import copy
        pm.AdaptiveMetropolis.step(self)
        like_vals, like_vars, Mc, Qc = self.approximate_gaussian_full_conditional.value
        self.field_plus_nugget.value = pm.rmv_normal_cov(Mc,Qc.I)
        M, C = copy(self.sp_sub.M.value), copy(self.sp_sub.C.value)
        if self.sp_sub.temporal:
            obs_mesh = self.data_mesh
        else:
            obs_mesh = self.data_mesh[:,:2]
        pm.gp.observe(M,C,obs_mesh,self.field_plus_nugget.value,obs_V=pm.utils.value(self.nugget))
    
        self.sp_sub.f_eval.value = pm.rmv_normal_cov(M(self.sp_sub.mesh), C(self.sp_sub.mesh, self.sp_sub.mesh))

def theano_invlogit(x):
    return T.exp(x)/(T.exp(x)+1)

def theano_binomial(k, n, p):
    return T.sum(k*T.log(p) + (n-k)*T.log(1-p))

def theano_multinomial(x,p):
    return T.sum([T.dot(T.log(p[i]),x[i]) for i in xrange(x.shape[0])])

# # Test multinomial.
# x1 = np.random.normal(size=(10,51))
# offsets = np.random.normal(size=10)**2
# p1 = T.dvector()
# p2 = T.dvector()
# ps = [p1+p2*o for o in offsets]
# mf = tfun([p1,p2], theano_multinomial(x1,ps))
# 
# p1_test = np.random.normal(size=51)**2
# p2_test = np.random.normal(size=51)**2
# ps_test = np.array([p1_test+p2_test*o for o in offsets])
# 
# m2 = np.sum(np.log(ps_test)*x1)
# np.testing.assert_almost_equal(np.asscalar(mf(p1_test,p2_test)),m2)

def likelihood_expression_to_potential(name, expr, x_theano, x_pymc):
    expr_fn = tfun(expr, x_theano)
    @pm.Potential(name=name)
    def pot(x=x_pymc, expr_fn=expr_fn):
        return expr_fn(*x)
    return pot

def zipmap(f, keys):
    return dict(zip(keys, map(f, keys)))

def incorporate_zeros(obs, n, predicate):
    "Where data is missing, set observation = n = 0."
    n_new = n.copy()
    obs_new = obs.copy()
    n_new[np.where(True-predicate)]=0
    obs_new[np.where(True-predicate)]=0
    return obs_new, n_new
    

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

    
    # =========================
    # = Haplotype frequencies =
    # =========================
    xb = T.dvector('xb')
    x0 = T.dvector('x0')
    xv = T.dvector('xv')
    x_dict = {'b': xb, '0': x0, 'v': xv}
    p1 = .01

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

    p_prom = (h_freqs['0']+h_freqs['1'])**2
    p_aphe = 1-(1-h_freqs['a'])**2
    p_bphe = 1-(1-h_freqs['b'])**2
    p_phe = [\
        g_freqs['ab'],
        g_freqs['a0']+g_freqs['a1']+g_freqs['aa'],
        g_freqs['b0']+g_freqs['b1']+g_freqs['bb'],
        g_freqs['00']+g_freqs['01']+g_freqs['11']]    
    gfreq_keys = ['aa','ab','a0','a1','bb','b0','b1','00','01','11']
    p_gen = [g_freqs[key] for key in gfreq_keys]
    p_vivax = pv*(1-p_prom)
        
    # Create the mean & its evaluation at the data locations.
    init_OK = False
        
    covariate_key_dict = {'v': set(covariate_keys), 'b': ['africa'], '0':[]}        
    ui_dict = {'v': full_vivax_ui, 'b': duffy_ui, '0': duffy_ui}
    logp_mesh_dict = {'b': duffy_logp_mesh, '0': duffy_logp_mesh, 'v': vivax_logp_mesh}
    data_mesh_dict = {'b': duffy_data_mesh, '0': duffy_data_mesh, 'v': vivax_data_mesh}
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
    
    V_b, V_0, V_v = [spatial_vars[k]['V'] for k in ['b','0','v']]
    eps_p_f = {}
        
    # Duffy eps_p_f's and p's, eval'ed everywhere.
    for k in ['b','0','v']:
        if k in ['b','0']:
            fi = duffy_fi
            data_mesh = duffy_data_mesh
        else:
            fi = vivax_fi
            data_mesh = vivax_data_mesh
        
        # Nuggeted field
        eps_p_f[k] = pm.Normal('eps_p_f_%s'%k, spatial_vars[k]['sp_sub'].f_eval[fi], tau[k], value=np.random.normal(size=len(data_mesh)), trace=False)

    warnings.warn('Not using age correction')
            
    prom_obs, prom_n = incorporate_zeros(prom0, n, datatype=='prom')
    theano_likelihood_prom = theano_binomial(prom_obs, prom_n, p_prom)
        
    aphe_obs, aphe_n = incorporate_zeros(aphea, n, datatype=='aphe')
    theano_likelihood_aphe = theano_binomial(aphe_obs, aphe_n, p_aphe)
        
    bphe_obs, bphe_n = incorporate_zeros(bpheb, n, datatype=='bphe')
    theano_likelihood_bphe = theano_binomial(bphe_obs, bphe_n, p_bphe)
        
    phe_obs, phe_n = incorporate_zeros(np.array([pheab,
                        phea,
                        pheb,
                        phe0]).T, n, datatype=='phe')
    theano_likelihood_phe = theano_multinomial(phe_obs.T, p_phe)
        
    gen_obs, gen_n = incorporate_zeros(np.array([genaa,
                        genab,
                        gena0,
                        gena1,
                        genbb,
                        genb0,
                        genb1,
                        gen00,
                        gen01,
                        gen11]).T, n, datatype=='gen')
    theano_likelihood_gen = theano_multinomial(gen_obs.T, p_gen)
    
    # Now vivax.
    vivax_obs, vivax_n = incorporate_zeros(vivax_pos, n, datatype=='vivax')
    theano_vivax_likelihood_for_duffy = theano_binomial(vivax_obs, vivax_n, p_vivax)
    theano_vivax_likelihood_for_vivax = theano_binomial(vivax_obs[where_vivax], vivax_n[where_vivax], p_vivax)
    
    theano_duffy_likelihood = theano_likelihood_prom + theano_likelihood_aphe + theano_likelihood_bphe + theano_likelihood_phe + theano_likelihood_gen
    
    return locals()