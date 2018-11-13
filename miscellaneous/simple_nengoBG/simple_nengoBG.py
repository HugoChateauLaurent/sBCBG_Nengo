import nengo
import numpy as np


class ActionIterator(object):
    def __init__(self, dimensions):
        self.actions = np.ones(dimensions) * 0.1
        self.dimensions = dimensions

    def step(self, x):
        # one action at time dominates
        
        dominate = int(x % self.dimensions)
        self.actions[:] = 0.1
        self.actions[dominate] = 0.8
        return self.actions
        
dim = 20
action_iterator = ActionIterator(dimensions=dim)

model = nengo.Network(label='Basal Ganglia')
with model:
    basal_ganglia = nengo.networks.BasalGanglia(dimensions=dim)
    actions = nengo.Node(action_iterator.step, label="actions")
    nengo.Connection(actions, basal_ganglia.input, synapse=None)
    
    
