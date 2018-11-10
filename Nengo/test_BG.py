# Tutorial 5: Computing a function

# Whenever we make a Connection between groups of neurons, we don't have to
# just pass the information from one group of neurons to the next.  Instead,
# we can also modify that information.  We do this by specifying a function,
# and Nengo will connect the individual neurons to best approximate that
# function.

# In the example here, we are computing the square of the value.  So for an
# input of -1 it should output 1, for 0 it should output 0, and for 1 it should
# output 1.

# You can change the function by adjusting the computations done in the
# part of the code labelled "compute_this".  This can be any arbitrary Python
# function.  For example, try computing the negative of x ("return -x").  Try
# the absolute value ("return abs(x)").  You can also try more complex
# functions like "return 1 if x > 0 else 0".

import nengo
import nengo.spa as spa

D = 32   # the dimensionality of the vectors

model = spa.SPA()
with model:
    model.  stim = nengo.Node(0)
    
    model.vision = spa.State(dimensions = D, neurons_per_dimension = 1)
    model.speech = spa.State(dimensions = D, neurons_per_dimension = 1)
    
    """ actions = spa.Actions(
        'dot(vision, DOG) --> speech=BARK',
        'dot(vision, CAT) --> speech=MEOW',
        'dot(vision, RAT) --> speech=SQUEAK',
        'dot(vision, COW) --> speech=MOO',
        '0.5 --> speech=0'
        )""" 
        
    actions = spa.Actions(
        'dot(vision, DOG) --> speech=BARK',
        'dot(vision, CAT) --> speech=MEOW'
        )
        
    model.bg = spa.BasalGanglia(actions)
    model.thalamus = spa.Thalamus(model.bg)
