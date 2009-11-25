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
import generic_mbg

__all__ = ['make_model']

def ibd_covariance_submodel():
    """
    A small function that creates the mean and covariance object
    of the random field.
    """
    
    # from duffy import cut_matern
    
    # The partial sill.
    amp = pm.Exponential('amp', .1, value=1.)
    
    # The range parameter. Units are RADIANS. 
    # 1 radian = the radius of the earth, about 6378.1 km
    # scale = pm.Exponential('scale', 1./.08, value=.08)
    
    scale_shift = pm.Exponential('scale_shift', .1, value=.08)
    scale = pm.Lambda('scale',lambda s=scale_shift: s+.01)
    scale_in_km = scale*6378.1
    
    # This parameter controls the degree of differentiability of the field.
    diff_degree = pm.Uniform('diff_degree', .01, 1.5)
    
    # The nugget variance.
    V = pm.Exponential('V', .1, value=1.)
    
    # Create the covariance & its evaluation at the data locations.
    @pm.deterministic(trace=True)
    def C(amp=amp, scale=scale, diff_degree=diff_degree):
        """A covariance function created from the current parameter values."""
        return pm.gp.FullRankCovariance(pm.gp.matern.geo_rad, amp=amp, scale=scale, diff_degree=diff_degree)
    
    return locals()
    
def make_model(lon,lat,covariate_values,n,africa,datatype,genaa,genab,genbb,gen00,gena0,genb0,gfga,gfgb,gfg0,pheab,phea,pheb,phe0,pos0,negab,aphea,aphe0,gfpa,gfpb,gfp0,gfpb0,cpus=1):
    """
    This function is required by the generic MBG code.
    """
        
    # Non-unique data locations
    data_mesh = combine_spatial_inputs(lon, lat)
    
    # Uniquify the data locations.
    locs = [(lon[0], lat[0])]
    fi = [0]
    ui = [0]
    for i in xrange(1,len(lon)):

        # If repeat location, add observation
        loc = (lon[i], lat[i])
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

    lon = np.array(locs)[:,0]
    lat = np.array(locs)[:,1]

    # Unique data locations
    logp_mesh = combine_spatial_inputs(lon,lat)
    
    # Create the mean & its evaluation at the data locations.
    M, M_eval = trivial_means(logp_mesh)
    print data_mesh.shape
    init_OK = False
    
    while not init_OK:
        try:        
            # Space-time component
            sp_sub = ibd_covariance_submodel()    
            covariate_dict, C_eval = cd_and_C_eval(covariate_values, sp_sub['C'], data_mesh, ui)
        
            # The field evaluated at the uniquified data locations            
            f = pm.MvNormalCov('f', M_eval, C_eval)
            # Make f start somewhere a bit sane
            f.value = f.value - np.mean(f.value)
            
            # Loop over data clusters
            eps_p_f_d = []
            p0_d = []
        
            for i in xrange(len(n)):
            
                this_f = pm.Lambda('f_%i'%i, lambda f=f, i=i, fi=fi: f[fi[i]], trace=False)
            
                # Nuggeted field in this cluster
                eps_p_f_d.append(pm.Normal('eps_p_f_%i'%i, this_f, 1./sp_sub['V'], value=0.,trace=False))
        
                # The allele frequency
                p0_d.append(pm.Lambda('s_%i'%i,lambda lt=eps_p_f_d[-1]: np.asscalar(invlogit(lt)),trace=False))
        
            # The field plus the nugget
            @pm.deterministic
            def eps_p_f(eps_p_fd = eps_p_f_d):
                """Concatenated version of eps_p_f, for postprocessing & Gibbs sampling purposes"""
                return np.hstack(eps_p_fd)
        
            init_OK = True
        except pm.ZeroProbability, msg:
            print 'Trying again: %s'%msg
            init_OK = False
            gc.collect()

    # The observed allele frequencies
    data_d = []    
    for i in xrange(len(n)):
        p0=p0_d[i]
        pb = .5
        if datatype[i]=='gf':
            pass
        elif datatype[i]=='pos':
            cur_obs = [pos0[i], negab[i]]
            p = pm.Lambda('p_%i'%i, lambda pb=pb, p0=p0: (pb*p0)**2, trace=False)
            n = np.sum(cur_obs)
            data_d.append(pm.Binomial('data_%i'%i, p=p, n=n, value=pos0[i], observed=True))
        elif datatype[i]=='aphe':
            cur_obs = [aphea[i], aphe0[i]]
            n = np.sum(cur_obs)
            p = pm.Lambda('p_%i'%i, lambda pb=pb, p0=p0: (1-pb)**2+2*(1-pb)*pb, trace=False)
            data_d.append(pm.Binomial('data_%i'%i, p=p, n=n, value=aphea[i], observed=True))
        elif datatype[i]=='phe':
            cur_obs = np.array([pheab[i],phea[i],pheb[i],phe0[i]])
            n = np.sum(cur_obs)
            p = pm.Lambda('p_%i'%i, lambda pb=pb, p0=p0: np.array([\
                2*(1-pb)*pb*(1-p0),
                2*(1-pb)*pb*p0+(1-pb)**2,
                2*pb**2*(1-p0)*p0+(pb*(1-p0))**2,
                (pb*p0)**2]), trace=False)
            data_d.append(pm.Multinomial('data_%i'%i, p=p, n=n, value=cur_obs, observed=True))
        elif datatype[i]=='gen':
            cur_obs = np.array([genaa[i],genab[i],gen00[i],gena0[i],genb0[i],genbb[i]])
            n = np.sum(cur_obs)
            p = pm.Lambda('p_%i'%i, lambda pb=pb, p0=p0: np.array([\
                (1-pb)**2,
                2*(1-pb)*pb*(1-p0),
                (pb*p0)**2,
                2*(1-pb)*pb*p0,
                2*pb**2*(1-p0)*p0,
                (pb*(1-p0))**2]), trace=False)
            data_d.append(pm.Multinomial('data_%i'%i, p=p, n=n, value=cur_obs, observed=True))

        if np.any(np.isnan(cur_obs)):
            raise ValueError
    
    out = locals()
    out.pop('sp_sub')
    out.update(sp_sub)

    return out