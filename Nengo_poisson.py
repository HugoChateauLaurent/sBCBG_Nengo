#-------------------------------------------------------------------------------
# Adapted from github.com/nengo/nengo/issues/579#issuecomment-260552809
# maybe more complete implementation here: github.com/nengo/nengo/issues/1487
#-------------------------------------------------------------------------------

class PoissonSpikingExact(object):
    def __init__(self, size, dt=0.001):
        self.rng = np.random.RandomState(seed=seed)
        self.dt = dt
        self.value = 1.0 / dt
        self.size = size
        self.output = np.zeros(size)
    def next_spike_times(self, rate):        
        return -np.log(1.0-self.rng.rand(len(rate))) / rate
    def __call__(self, t, x):
        self.output[:] = 0

        next_spikes = self.next_spike_times(x)
        s = np.where(next_spikes<self.dt)[0]
        count = len(s)
        self.output[s] += self.value
        while count > 0:
            next_spikes[s] += self.next_spike_times(x[s])
            s2 = np.where(next_spikes[s]<self.dt)[0]
            count = len(s2)
            s = s[s2]
            self.output[s] += self.value

        return self.output