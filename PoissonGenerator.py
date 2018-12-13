# Taken from Terrence Stewart's Github
# https://github.com/tcstewar/testing_notebooks/blob/master/spike%20trains%20-%20poisson%20and%20regular.ipynb

import numpy as np
import nengo
from nengo.params import Parameter, NumberParam, FrozenObject


class PoissonGenerator(object):
    def __init__(self, rate, size, seed, dt=0.001, parrot=True):
        parrot = True
        self.rng = np.random.RandomState(seed=seed)
        self.dt = dt
        self.value = 1.0 / dt
        self.size = size
        self.parrot = parrot
        self.rate = float(rate)
        self.output = np.zeros((size if parrot else 1))
        
    def next_spike_times(self, size):        
        return -np.log(1.0-self.rng.rand(size)) / self.rate
    
    def __call__(self, t, x):
        self.output[:] = 0
                
        next_spikes = self.next_spike_times(self.size if self.parrot else 1)
        s = np.where(next_spikes<self.dt)[0]
        count = len(s)
        self.output[s] += self.value
        while count > 0:
            next_spikes[s] += self.next_spike_times(count)
            s2 = np.where(next_spikes[s]<self.dt)[0]
            count = len(s2)
            s = s[s2]
            self.output[s] += self.value
            
        return self.output if self.parrot else np.array([self.output[0]]*self.size)

class Parrot(nengo.neurons.NeuronType):
    """Fake neuron that repeats the spikes he gets in input
    """

    probeable = ('spikes', 'voltage')
    def step_math(self, dt, J, output):
        output[:] = J