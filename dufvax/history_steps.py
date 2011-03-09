import pymc as pm
import numpy as np

class HistoryCovarianceStepper(pm.StepMethod):
    
    _state = ['n_points','history','tally','verbose']
    
    def __init__(self, stochastic, n_points=None, init_history=None, verbose=None, tally=False):
        
        if not np.iterable(stochastic) or isinstance(stochastic, pm.Variable):
            stochastic = [stochastic]

        pm.StepMethod.__init__(self, stochastic, verbose, tally)

        # Initialization methods
        self.check_type()
        self.dimension()
        self.n_points = n_points or max(self.dim, 20)
        self.index = 0
        
        if init_history is None:
            self.history = np.empty((self.n_points, self.dim))
            for i in xrange(self.n_points):
                for s in self.stochastics:
                    self.history[i,self._slices[s]] = s._random(**s.parents.value)
        else:
            self.history = init_history
            
        self.history_mean = np.mean(self.history, axis=0)
        
        self._state = HistoryCovarianceStepper._state
        
    def unstore(self):
        index = (self.index-1)%self.n_points
        self.history_mean += (self.last_value - self.history[index])/self.n_points
        self.history[index] = self.last_value

    def get_current_value(self):
        current_value = np.empty(self.dim)
        for s in self.stochastics:
            current_value[self._slices[s]] = s.value
        return current_value

    def set_current_value(self, v):
        for s in self.stochastics:
            s.value = v[self._slices[s]]
    
    def store(self, v):
        self.last_value = self.history[self.index].copy()        
        self.history[self.index] = v
        self.index = (self.index + 1) % self.n_points        
        self.history_mean += (v-self.last_value)/self.n_points

    def check_type(self):
        """Make sure each stochastic has a correct type, and identify discrete stochastics."""
        self.isdiscrete = {}
        for stochastic in self.stochastics:
            if stochastic.dtype in pm.StepMethods.integer_dtypes:
                self.isdiscrete[stochastic] = True
            elif stochastic.dtype in pm.StepMethods.bool_dtypes:
                raise 'Binary stochastics not supported by AdaptativeMetropolis.'
            else:
                self.isdiscrete[stochastic] = False

    def choose_direction(self, norm=True):
        direction = np.dot(np.random.normal(size=self.n_points), self.history) - self.history_mean
        if norm:
            direction /= np.dot(direction, direction)
        return direction

    def dimension(self):
        """Compute the dimension of the sampling space and identify the slices
        belonging to each stochastic.
        """
        self.dim = 0
        self._slices = {}
        for stochastic in self.stochastics:
            if isinstance(stochastic.value, np.matrix):
                p_len = len(stochastic.value.A.ravel())
            elif isinstance(stochastic.value, np.ndarray):
                p_len = len(stochastic.value.ravel())
            else:
                p_len = 1
            self._slices[stochastic] = slice(self.dim, self.dim + p_len)
            self.dim += p_len
            
class HistoryAM(HistoryCovarianceStepper, pm.Metropolis):

    _state = HistoryCovarianceStepper._state + ['accepted','rejected','adaptive_scale_factor','proposal_distribution','proposal_sd']
    
    def __init__(self, stochastic, n_points=None, init_history=None, verbose=None, tally=False, proposal_sd=1):
        HistoryCovarianceStepper.__init__(self, stochastic, n_points, init_history, verbose, tally)
        self.proposal_distribution = "None"
        self.adaptive_scale_factor = 1
        self.accepted = 0
        self.rejected = 0        
        self.proposal_sd=proposal_sd
        self._state = HistoryAM._state

    def reject(self):
        for stochastic in self.stochastics:
            stochastic.revert()
        self.unstore()
    
    def propose(self):
        direction = self.choose_direction()
        proposed_value = self.get_current_value() + direction * np.random.normal()*self.adaptive_scale_factor*self.proposal_sd
        self.set_current_value(proposed_value)
        self.store(proposed_value)
    
class HRAM(HistoryCovarianceStepper):
    
    _state = HistoryCovarianceStepper._state + ['xprime_n', 'xprime_sds']
    
    def __init__(self, stochastic, n_points=None, init_history=None, verbose=None, tally=False, xprime_sds=5, xprime_n=101):
        HistoryCovarianceStepper.__init__(self, stochastic, n_points, init_history, verbose, tally)
        self.xprime_sds = xprime_sds
        self.xprime_n = xprime_n
        self._state = HRAM._state
        
    def step(self):
        direction = self.choose_direction(norm=False)
        current_value = self.get_current_value()
        x_prime = np.vstack((current_value, np.outer(np.linspace(-self.xprime_sds,self.xprime_sds,self.xprime_n),direction) + current_value))
        lps = np.empty(self.xprime_n+1)
        lps[0] = self.logp_plus_loglike
        for i in xrange(self.xprime_n):
            self.set_current_value(x_prime[i+1])
            lps[i+1] = self.logp_plus_loglike
        next_value = x_prime[pm.rcategorical(np.exp(lps-pm.flib.logsum(lps)))]
        self.set_current_value(next_value)
        self.store(next_value)
        
    
if __name__ == '__main__':
    import time
    for kls in [HistoryAM, HRAM, pm.AdaptiveMetropolis]:
        x = pm.Normal('x',0,1,size=1000)
        y = pm.Normal('y',x,100,size=1000)
    
        M = pm.MCMC([x,y])
        # M.use_step_method(HistoryAM, [x,y])
        if kls is pm.AdaptiveMetropolis:
            M.use_step_method(kls, [x,y], delay=10, interval=10)
        else:
            M.use_step_method(kls, [x,y])
        sm = M.step_method_dict[x][0]
        if kls is HRAM:
            sm.accepted = 0
            sm.rejected = 0

        t1 = time.time()
        M.sample(100)
        t2 = time.time()
        print 'Class %s: %s seconds, accepted %i, rejected %i'%(kls.__name__, t2-t1, sm.accepted, sm.rejected)
        import pylab as pl
        pl.clf()
        pl.plot(M.trace('x')[:,0], M.trace('y')[:,0],linestyle='none',marker='.',markersize=8,color=(.1,0,.8,),alpha=10./len(M.trace('x')[:]),markeredgewidth=0)
        pl.title(kls.__name__)
        
        from IPython.Debugger import Pdb
        Pdb(color_scheme='Linux').set_trace()   
    
    # a = np.empty((1000,2))
    # for i in xrange(1000):
    #     M.step_method_dict[x][0].step()
    #     a[i,0] = x.value
    #     a[i,1] = y.value
    #     
    # import pylab as pl
    # pl.plot(a[:,0], a[:,1], 'b.')