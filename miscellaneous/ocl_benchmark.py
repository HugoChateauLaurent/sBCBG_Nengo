import nengo
import nengo_ocl
import numpy as np

model = nengo.Network()

with model:
    size = 1300
    dims = 500

    Ie = nengo.Node([50]*dims)
    
    ens1 = nengo.Ensemble(size, dims)
    
    ens2 = nengo.Ensemble(size, dims)

    nengo.Connection(Ie, ens1)
    nengo.Connection(ens1, ens2)

ocl = False
if ocl:
	sim = nengo_ocl.Simulator(model)
else:
	sim = nengo.Simulator(model)
sim.run(10.)


