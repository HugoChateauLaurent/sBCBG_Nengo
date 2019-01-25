#!/usr/bin/python
# -*- coding: utf-8 -*-


import os
import sys
import csv
import numpy as np
import numpy.random as rnd
from math import sqrt, cosh, exp, pi
import copy
from PoissonGenerator import PoissonGenerator, Parrot

dt = .001 # ms


from modelParams import *

import nengo
import nengo_ocl
from nengo.synapses import Alpha
net = nengo.Network(seed=params['nestSeed'])
net.pops = {}


#------------------------------------------
# Start LGneurons.py
interactive = False # avoid loading X dependent things
                   # set to False for simulations on Sango
storeGDF = True # unless overriden by run.py, keep spike rasters




AMPASynapseCounter = 0 # counter variable for the fast connect

#-------------------------------------------------------------------------------
# Loads a given LG14 model parameterization
# ID must be in [0,14]
#-------------------------------------------------------------------------------
def loadLG14params(ID):
  # Load the file with the Lienard solutions:
  LG14SolutionsReader = csv.DictReader(open("solutions_simple_unique.csv"),delimiter=';')
  LG14Solutions = []
  for row in LG14SolutionsReader:
    LG14Solutions.append(row)

  print('### Parameterization #'+str(ID)+' from (Lienard & Girard, 2014) is used. ###')

  for k,v in alpha.items():
    try:
      if k == 'Arky->MSN':
        alpha[k] = round(float(LG14Solutions[ID]['ALPHA_GPe_MSN']),0)
      elif k == 'Arky->FSI':
        alpha[k] = round(float(LG14Solutions[ID]['ALPHA_GPe_FSI']),0)
      elif k == 'CMPf->Arky':
        alpha[k] = round(float(LG14Solutions[ID]['ALPHA_CMPf_GPe']),0)
      elif k == 'CMPf->Prot':
        alpha[k] = round(float(LG14Solutions[ID]['ALPHA_CMPf_GPe']),0)
      elif k == 'MSN->Arky':
        alpha[k] = round(float(LG14Solutions[ID]['ALPHA_MSN_GPe']),0)
      elif k == 'MSN->Prot':
        alpha[k] = round(float(LG14Solutions[ID]['ALPHA_MSN_GPe']),0)
      elif k == 'Prot->STN':
        alpha[k] = round(float(LG14Solutions[ID]['ALPHA_GPe_STN']),0)
      elif k == 'STN->Arky':
        alpha[k] = round(float(LG14Solutions[ID]['ALPHA_STN_GPe']),0)
      elif k == 'STN->Prot':
        alpha[k] = round(float(LG14Solutions[ID]['ALPHA_STN_GPe']),0)
      elif k == 'Arky->Arky':
        alpha[k] = round(float(LG14Solutions[ID]['ALPHA_GPe_GPe']),0)
      elif k == 'Arky->Prot':
        alpha[k] = round(float(LG14Solutions[ID]['ALPHA_GPe_GPe']),0)
      elif k == 'Prot->Arky':
        alpha[k] = round(float(LG14Solutions[ID]['ALPHA_GPe_GPe']),0)
      elif k == 'Prot->Prot':
        alpha[k] = round(float(LG14Solutions[ID]['ALPHA_GPe_GPe']),0)
      elif k == 'Prot->GPi':
        alpha[k] = round(float(LG14Solutions[ID]['ALPHA_GPe_GPi']),0)
      else:
        alpha[k] = round(float(LG14Solutions[ID]['ALPHA_'+k.replace('->','_')]),0) 
    except:
      print(('Could not find LG14 parameters for connection `'+k+'`, trying to run anyway.'))

  for k,v in p.items():
      try:
        if k == 'Arky->MSN':
          p[k] = round(float(LG14Solutions[ID]['DIST_GPe_MSN']),2)
        elif k == 'Arky->FSI':
          p[k] = round(float(LG14Solutions[ID]['DIST_GPe_FSI']),2)
        elif k == 'CMPf->Arky':
          p[k] = round(float(LG14Solutions[ID]['DIST_CMPf_GPe']),2)
        elif k == 'CMPf->Prot':
          p[k] = round(float(LG14Solutions[ID]['DIST_CMPf_GPe']),2)
        elif k == 'MSN->Arky':
          p[k] = round(float(LG14Solutions[ID]['DIST_MSN_GPe']),2)
        elif k == 'MSN->Prot':
          p[k] = round(float(LG14Solutions[ID]['DIST_MSN_GPe']),2)
        elif k == 'Prot->STN':
          p[k] = round(float(LG14Solutions[ID]['DIST_GPe_STN']),2)
        elif k == 'STN->Arky':
          p[k] = round(float(LG14Solutions[ID]['DIST_STN_GPe']),2)
        elif k == 'STN->Prot':
          p[k] = round(float(LG14Solutions[ID]['DIST_STN_GPe']),2)
        elif k == 'Arky->Arky':
          p[k] = round(float(LG14Solutions[ID]['DIST_GPe_GPe']),2)
        elif k == 'Arky->Prot':
          p[k] = round(float(LG14Solutions[ID]['DIST_GPe_GPe']),2)
        elif k == 'Prot->Arky':
          p[k] = round(float(LG14Solutions[ID]['DIST_GPe_GPe']),2)
        elif k == 'Prot->Prot':
          p[k] = round(float(LG14Solutions[ID]['DIST_GPe_GPe']),2)
        elif k == 'Prot->GPi':
          p[k] = round(float(LG14Solutions[ID]['DIST_GPe_GPi']),2)
        else:
          p[k] = round(float(LG14Solutions[ID]['DIST_'+k.replace('->','_')]),2) 
      except:
        print(('Could not find LG14 parameters for connection `'+k+'`, trying to run anyway.'))

  for k,v in BGparams.items():
    try:
      if k == 'Arky':
          BGparams[k]['V_th'] = round(float(LG14Solutions[ID]['THETA_GPe']),1)
      elif k == 'Prot':
          BGparams[k]['V_th'] = round(float(LG14Solutions[ID]['THETA_GPe']),1)
      else:
          BGparams[k]['V_th'] = round(float(LG14Solutions[ID]['THETA_'+k]),1)
    except:
      print(('Could not find LG14 parameters for connection `'+k+'`, trying to run anyway.'))


def loadThetaFromCustomparams(params):
  for k,v in BGparams.items():
    try:
      newval = round(float(params['THETA_'+k]), 2)
      print(("WARNING: overwriting LG14 value for theta in "+k+" from original value of "+str(BGparams[k]['V_th'])+" to new value: "+str(newval)))
      BGparams[k]['V_th'] = newval # firing threshold
    except:
      print(("INFO: keeping LG14 value for theta in "+k+" to its original value of "+str(BGparams[k]['V_th'])))
      pass

#-------------------------------------------------------------------------------
# Changes the default of the iaf_psc_alpha_multisynapse neurons
# Very important because it defines the 3 types of receptors (AMPA, NMDA, GABA) that will be needed
# Has to be called after any KernelReset
#-------------------------------------------------------------------------------
def initNeurons():

  commonNeuronType = nengo.neurons.LIF(tau_rc=0.01, tau_ref=CommonParams['t_ref']/1000.)#, min_voltage=CommonParams['V_min'])
  commonNeuronType.params = {}
  commonNeuronType.params['tau_syn'] = [tau/1000. for tau in CommonParams['tau_syn']]
  commonNeuronType.params['V_th'] = CommonParams['V_th']
  
  return commonNeuronType


#-------------------------------------------------------------------------------
# Creates a population of neurons
# name: string naming the population, as defined in NUCLEI list
# fake: if fake is True, the neurons will be replaced by Poisson generators, firing
#       at the rate indicated in the "rate" dictionary
# parrot: do we use parrot neurons or not? If not, there will be no correlations in the inputs, and a waste of computation power...
#-------------------------------------------------------------------------------
def create(name,fake=False,parrot=True):
  if nbSim[name] == 0:
    print('ERROR: create(): nbSim['+name+'] = 0')
    exit()

  if fake:
    if rate[name] == 0:
      print('ERROR: create(): rate['+name+'] = 0 Hz')
    print('* '+name+'(fake):',nbSim[name],'Poisson generators with avg rate:',rate[name])

    if not parrot:
      print("/!\ /!\ /!\ /!\ \nWARNING: parrot neurons not used, no correlations in inputs\n")
    
  
    with net:
      poisson = False
      if not poisson:
      	
        poisson_ens = nengo.Ensemble(int(nbSim[name]), 1, encoders=np.ones((int(nbSim[name]),1)), 
                              gain=np.ones((int(nbSim[name]))), 
                              bias=np.zeros((int(nbSim[name]))),
                              neuron_type=nengo.neurons.SpikingRectifiedLinear(),
                              label=name)

        # Add constant input
        # nengo.Connection(nengo.Node([rate[name]], label='Ie node '+name), poisson_ens, 
        #                 synapse=None, label='Ie connection '+name)

      else:
        poisson_node = nengo.Node(PoissonGenerator(rate[name], int(nbSim[name]), 
                                                    seed=params['nestSeed'], dt=dt, parrot=parrot), 
                                    size_in=int(nbSim[name]))
        poisson_ens = nengo.Ensemble(int(nbSim[name]), 1, 
                                      encoders=np.ones((int(nbSim[name]),1)), 
                                      gain=np.ones((int(nbSim[name]))), 
                                      bias=np.zeros((int(nbSim[name]))),
                                      neuron_type=Parrot(),
                                      label=name)

        nengo.Connection(poisson_node, poisson_ens.neurons, synapse=None)
        

      poisson_ens.efferents = []
      Pop[name] = poisson_ens
      net.pops[name] = poisson_ens
      print('TODONengo: seed and dt as parameters for Poisson generators')


  else:
    print('* '+name+':',nbSim[name],'neurons with parameters:',BGparams[name])
  
    with net:
      Pop[name] = nengo.Ensemble(int(nbSim[name]), 1, encoders=np.ones((int(nbSim[name]),1)), 
                            gain=np.ones((int(nbSim[name]))), 
                            bias=np.zeros((int(nbSim[name]))), 
                            neuron_type=neuronTypes[name], 
                            label=name)
      net.pops[name] = Pop[name]

      Pop[name].efferents = []

      Pop[name].Ie = nengo.Connection(nengo.Node([0], label='Ie node '+name), Pop[name], 
                        transform=1./Pop[name].neuron_type.params['V_th'], 
                        synapse=None, label='Ie connection '+name)
#-------------------------------------------------------------------------------
# Creates a popolation of neurons subdivided in Multiple Channels
#
# name: string naming the population, as defined in NUCLEI list
# nbCh: integer stating the number of channels to be created
# fake: if fake is True, the neurons will be replaced by Poisson generators, firing
#       at the rate indicated in the "rate" dictionary
# parrot: do we use parrot neurons or not? If not, there will be no correlations in the inputs, and a waste of computation power...
#-------------------------------------------------------------------------------
def createMC(name,nbCh,fake=False,parrot=True):
  print(nbCh)
  if nbSim[name] == 0:
    print('ERROR: create(): nbSim['+name+'] = 0')
    exit()

  Pop[name]=[]
  with net:
    net.pops[name] = []

  if fake:
    if rate[name] == 0:
      print('ERROR: create(): rate['+name+'] = 0 Hz')
    print('* '+name+'(fake):',nbSim[name],'Poisson generators with avg rate:',rate[name])

    if not parrot:
      print("/!\ /!\ /!\ /!\ \nWARNING: parrot neurons not used, no correlations in inputs\n")
    
  
    with net:
      poisson_node = nengo.Node(PoissonGenerator(rate[name], int(nbSim[name]), 
                                                  seed=params['nestSeed'], dt=dt, parrot=parrot), 
                                  size_in=int(nbSim[name]))
      for i in range(nbCh):  

        poisson = False
        if not poisson:
          
          poisson_ens = nengo.Ensemble(int(nbSim[name]), 1, encoders=np.ones((int(nbSim[name]),1)), 
                                gain=np.ones((int(nbSim[name]))), 
                                bias=np.zeros((int(nbSim[name]))),
                                neuron_type=nengo.neurons.SpikingRectifiedLinear(),
                                label=name+' ch'+str(i))

          # Add constant input to CMPf and PTN
          if name != "CSN":
            nengo.Connection(nengo.Node([rate[name]], label='Ie node '+name), poisson_ens, 
                            synapse=None, label='Ie connection '+name)

        else:          
          poisson_ens = nengo.Ensemble(int(nbSim[name]), 1, 
                                        encoders=np.ones((int(nbSim[name]),1)), 
                                        gain=np.ones((int(nbSim[name]))), 
                                        bias=np.zeros((int(nbSim[name]))),
                                        neuron_type=Parrot(),
                                        label=name+' ch'+str(i))

          nengo.Connection(poisson_node, poisson_ens.neurons, synapse=None)
        
        poisson_ens.efferents = []
        Pop[name].append(poisson_ens)
        net.pops[name].append(poisson_ens)
        print('TODONengo: seed and dt as parameters for Poisson generators')
      

  else:
    print('* '+name+':',nbSim[name]*nbCh,'neurons (divided in',nbCh,'channels) with parameters:',BGparams[name])
  
    with net:
      for i in range(nbCh):
        Pop[name].append(nengo.Ensemble(int(nbSim[name]), 1, encoders=np.ones((int(nbSim[name]),1)), 
                                  gain=np.ones((int(nbSim[name]))), 
                                  bias=np.zeros((int(nbSim[name]))), 
                                  neuron_type=neuronTypes[name], 
                                  label=name+' ch'+str(i)))
        net.pops[name].append(Pop[name][-1])
        Pop[name][-1].efferents = []
        Pop[name][-1].Ie = nengo.Connection(nengo.Node([0], label='Ie node '+name+' ch'+str(i)), Pop[name][-1],
                                                                    transform=1./Pop[name][-1].neuron_type.params['V_th'], 
                                                                    synapse=None, 
                                                                    label='Ie connection '+name+' ch'+str(i))

#------------------------------------------------------------------------------
# Nengo only:
# TODO: rewrite
# Computes the weight matrix of size m*n for a connection between pre -> post
# with n_pre and n_post number of neurons and a given integer_inDegree.
# Assumption: a cell from pre cannot be connected twice to a cell from post -> TODO rewrite: Not in Nest
#------------------------------------------------------------------------------
def connectivity_matrix(rule, parameter, n_pre, n_post):
  connectivity = np.zeros((n_post, n_pre))

  if rule=='fixed_indegree':
    for post_neuron in range(n_post):
      for in_i in range(parameter):
        pre_neuron = rnd.randint(0, n_pre)
        connectivity[post_neuron, pre_neuron] += 1
  
  elif rule=='fixed_total_number':
    for connection in range(parameter):
      pre_neuron = rnd.randint(0, n_pre)
      post_neuron = rnd.randint(0, n_post)
      connectivity[post_neuron, pre_neuron] += 1

  else:
    raise ValueError('Nengo error: unknown rule in connectivity_matrix')

  return connectivity

#------------------------------------------------------------------------------
# Nengo only:
#------------------------------------------------------------------------------
class Delay(object):
  def __init__(self, dimensions, timesteps):
    self.history = np.zeros((timesteps, dimensions))
  def step(self, t, x):
    self.history = np.roll(self.history, -1, 0)
    self.history[-1] = x
    return self.history[0]

def delayed_connection(pre, post, delay, transform, synapse):
  with net:
    delayNode = nengo.Node(Delay(pre.n_neurons, int((delay/1000.) / dt)).step, 
                            size_in=pre.n_neurons, 
                            size_out=pre.n_neurons)

    pre_to_delay = nengo.Connection(pre.neurons, delayNode, 
                                    transform=np.ones((pre.n_neurons)),
                                    synapse=None)

    delay_to_post = nengo.Connection(delayNode, post.neurons, 
                                      transform=transform, 
                                      synapse=synapse)

    return {'pre_to_delay':pre_to_delay, 'delay_to_post':delay_to_post}




#------------------------------------------------------------------------------
# Routine to perform the fast connection using nest built-in `connect` function
# - `source` & `dest` are lists defining Nest IDs of source & target population
# - `synapse_label` is used to tag connections and be able to find them quickly
#   with function `mass_mirror`, that adds NMDA on top of AMPA connections
# - `inDegree`, `receptor_type`, `weight`, `delay` are Nest connection params
#------------------------------------------------------------------------------
def mass_connect(source, dest, synapse_label, inDegree, receptor_type, weight, delay, stochastic_delays=None, verbose=False):
  def printv(text):
    if verbose:
      print(text)
  
  sigmaDependentInterval = True # Hugo's method

  n_source = source.n_neurons
  n_dest = dest.n_neurons

  # potential initialization of stochastic delays
  if stochastic_delays != None and delay > 0 and stochastic_delays > 0.:
    printv('Using stochastic delays in mass-connect')
    sigma = delay * stochastic_delays
    if sigmaDependentInterval:
      n = 2 # number of standard deviation to include in the distribution
      if stochastic_delays >= 1./n:
        print('Error : stochastic_delays >= 1/n and the distribution of delays therefore includes 0 which is not possible -> Jean\'s method is used')
        sigmaDependentInterval = False
      else:
        low = delay - n*sigma
        high = delay + n*sigma

      
    if not sigmaDependentInterval:
      low = .5*delay
      high = 1.5*delay

    delay =  {'distribution': 'normal_clipped', 'low': low, 'high': high, 'mu': delay, 'sigma': sigma}

  # The first `fixed_indegree` connection ensures that all neurons in `dest`
  # are targeted by the same number of axons (an integer number)
  integer_inDegree = np.floor(inDegree)
  if integer_inDegree>0:      
    printv('Adding '+str(int(integer_inDegree*n_dest))+' connections with rule `fixed_indegree`\n')
  
    if stochastic_delays:
      raise NotImplementedError("TODONengo: delay distribution")
    connectivity = connectivity_matrix('fixed_indegree', int(integer_inDegree), source.n_neurons, dest.n_neurons)
    nengo_weight = weight*exp(1)*dest.neuron_type.params['tau_syn'][receptor_type-1]/dest.neuron_type.params['V_th']
    synapse = Alpha(dest.neuron_type.params['tau_syn'][receptor_type-1])
    synapse.label = synapse_label
    connection = delayed_connection(source, dest, delay, connectivity*nengo_weight, synapse)

    #store weight to retrieve connectivity matrix in mass_mirror:
    connection['delay_to_post'].weight = nengo_weight

    source.efferents.append(connection)

  # The second `fixed_total_number` connection distributes remaining axonal
  # contacts at random (i.e. the remaining fractional part after the first step)
  float_inDegree = inDegree - integer_inDegree
  remaining_connections = np.round(float_inDegree * n_dest)
  if remaining_connections > 0:
  
    
    if stochastic_delays:
      raise NotImplementedError("TODONengo: delay distribution")
    
    connectivity = connectivity_matrix('fixed_total_number', int(remaining_connections), source.n_neurons, dest.n_neurons)
    nengo_weight = weight*exp(1)*dest.neuron_type.params['tau_syn'][receptor_type-1]/dest.neuron_type.params['V_th']
    synapse = Alpha(dest.neuron_type.params['tau_syn'][receptor_type-1])
    synapse.label = synapse_label
    connection = delayed_connection(source, dest, delay, connectivity*nengo_weight, synapse)
    
    #store weight to retrieve connectivity matrix in mass_mirror:
    connection['delay_to_post'].weight = nengo_weight

    source.efferents.append(connection)

#------------------------------------------------------------------------------
# Routine to duplicate a connection made with a specific receptor, with another
# receptor (typically to add NMDA connections to existing AMPA connections)
# - `source` & `synapse_label` should uniquely define the connections of
#   interest - typically, they are the same as in the call to `mass_connect`
# - `receptor_type`, `weight`, `delay` are Nest connection params
#------------------------------------------------------------------------------
def mass_mirror(source, synapse_label, receptor_type, weight, delay, stochastic_delays, verbose=False):

  def printv(text):
    if verbose:
      print(text)

  # find all AMPA connections for the given projection type
  printv('looking for AMPA connections to mirror with NMDA...\n')
  

  ampa_conns = [conn['delay_to_post'] for conn in source.efferents if conn['delay_to_post'].synapse.label==synapse_label]
  for conn in ampa_conns:
    if not (np.round(np.array(conn.transform)/conn.weight)!=0).any():
      print("Nengo implementation warning: empty connection in mass mirror. See proposed solution below.")
      # proposed solution: [conn for conn in source.efferents if conn.synapse.label==synapse_label and (np.round(np.array(conn.transform)/conn.weight)!=0).any()]
  
  # in rare cases, there may be no connections, guard against that
  if ampa_conns:
    # extract just source and target GID lists, all other information is irrelevant here
    printv('found '+str(len(ampa_conns))+' AMPA connections\n')
    
    if stochastic_delays != None and delay > 0:
      raise NotImplementedError("TODONengo: delay distribution")
      printv('Using stochastic delays in mass-mirror')
    
  
    for conn in ampa_conns:        
      connectivity = np.array(conn.transform)/conn.weight
      nengo_weight = weight*exp(1)*conn.post.ensemble.neuron_type.params['tau_syn'][receptor_type-1]/conn.post.ensemble.neuron_type.params['V_th']        
      synapse = Alpha(conn.post.ensemble.neuron_type.params['tau_syn'][receptor_type-1])
      synapse.label = synapse_label
      connection = delayed_connection(source, conn.post.ensemble, delay, connectivity*nengo_weight, synapse)
      
      #store weight to retrieve connectivity matrix in mass_mirror:
      connection['delay_to_post'].weight = nengo_weight

      source.efferents.append(connection)

#-------------------------------------------------------------------------------
# Establishes a connexion between two populations, following the results of LG14
# type : a string 'ex' or 'in', defining whether it is excitatory or inhibitory
# nameTgt, nameSrc : strings naming the populations, as defined in NUCLEI list
# redundancy : value that characterizes the number of repeated axonal contacts from one neuron of Src to one neuron of Tgt (see RedundancyType for interpretation of this value)
# RedundancyType : string
#   if 'inDegreeAbs': `redundancy` is the number of neurons from Src that project to a single Tgt neuron
#   if 'outDegreeAbs': `redundancy` is number of axonal contacts between each neuron from Src onto a single Tgt neuron
#   if 'outDegreeCons': `redundancy` is a scaled proportion of axonal contacts between each neuron from Src onto a single Tgt neuron given arithmetical constraints, ranging from 0 (minimal number of contacts to achieve required axonal bouton counts) to 1 (maximal number of contacts with respect to population numbers)
# LCGDelays: shall we use the delays obtained by (Liénard, Cos, Girard, in prep) or not (default = True)
# gain : allows to amplify the weight normally deduced from LG14
#-------------------------------------------------------------------------------
def connect(type, nameSrc, nameTgt, redundancy, RedundancyType, LCGDelays=True, gain=1., stochastic_delays=None, verbose=False, projType=''):

  def printv(text):
    if verbose:
      print(text)

  printv("* connecting "+nameSrc+" -> "+nameTgt+" with "+type+" connection")

  if RedundancyType == 'inDegreeAbs':
    # inDegree is already provided in the right form
    inDegree = float(redundancy)
  elif RedundancyType == 'outDegreeAbs':
    #### fractional outDegree is expressed as a fraction of max axo-dendritic contacts
    inDegree = get_frac(1./redundancy, nameSrc, nameTgt, neuronCounts[nameSrc], neuronCounts[nameTgt], verbose=verbose)
  elif RedundancyType == 'outDegreeCons':
    #### fractional outDegree is expressed as a ratio of min/max axo-dendritic contacts
    inDegree = get_frac(redundancy, nameSrc, nameTgt, neuronCounts[nameSrc], neuronCounts[nameTgt], useMin=True, verbose=verbose)
  else:
    raise KeyError('`RedundancyType` should be one of `inDegreeAbs`, `outDegreeAbs`, or `outDegreeCons`.')

  # check if in degree acceptable (not larger than number of neurons in the source nucleus)
  if inDegree  > nbSim[nameSrc]:
    printv("/!\ WARNING: required 'in degree' ("+str(inDegree)+") larger than number of neurons in the source population ("+str(nbSim[nameSrc])+"), thus reduced to the latter value")
    inDegree = nbSim[nameSrc]

  if inDegree == 0.:
    printv("/!\ WARNING: non-existent connection strength, will skip")
    return

  global AMPASynapseCounter

  # process receptor types
  if type == 'ex':
    lRecType = ['AMPA','NMDA']
    AMPASynapseCounter = AMPASynapseCounter + 1
    lbl = AMPASynapseCounter # needs to add NMDA later
  elif type == 'AMPA':
    lRecType = ['AMPA']
    lbl = 0
  elif type == 'NMDA':
    lRecType = ['NMDA']
    lbl = 0
  elif type == 'in':
    lRecType = ['GABA']
    lbl = 0
  else:
    raise KeyError('Undefined connexion type: '+type)

  W = computeW(lRecType, nameSrc, nameTgt, inDegree, gain, verbose=False)

  printv("  W="+str(W)+" and inDegree="+str(inDegree))

  #if nameSrc+'->'+nameTgt in ConnectMap:
  #  loadConnectMap = True
  #else:
  #  loadConnectMap = False
  #  ConnectMap[nameSrc+'->'+nameTgt] = []

  # determine which transmission delay to use:
  if LCGDelays:
    delay= tau[nameSrc+'->'+nameTgt]
  else:
    delay= 1.

  mass_connect(Pop[nameSrc], Pop[nameTgt], lbl, inDegree, recType[lRecType[0]], W[lRecType[0]], delay, stochastic_delays = stochastic_delays)
  if type == 'ex':
    # mirror the AMPA connection with similarly connected NMDA connections
    mass_mirror(Pop[nameSrc], lbl, recType['NMDA'], W['NMDA'], delay, stochastic_delays = stochastic_delays)

  return W


#-------------------------------------------------------------------------------
# Establishes a connexion between two populations, following the results of LG14, in a MultiChannel context
# type : a string 'ex' or 'in', defining whether it is excitatory or inhibitory
# nameTgt, nameSrc : strings naming the populations, as defined in NUCLEI list
# projType : type of projections. For the moment: 'focused' (only channel-to-channel connection) and
#            'diffuse' (all-to-one with uniform distribution)
# redundancy, RedundancyType : contrains the inDegree - see function `connect` for details
# LCGDelays : shall we use the delays obtained by (Liénard, Cos, Girard, in prep) or not (default = True)
# gain : allows to amplify the weight normally deduced from LG14
# source_channels : By default with `source_channels=None`, the connection is implemented using all source channels
#                   Specify a custom list of channels to implement connections only from these channels
#                   For example, calling successively `connectMC(...,projType='focused',source_channels=[0])` and then `connectMC(...,projType='diffuse',source_channels=[1])` would implement first a focused projection using only source channel 0 and then a diffuse connection using only source channel 1:
#                   Src channels:   (0) (1)
#                                    | / |
#                   Tgt channels:   (0) (1)
#-------------------------------------------------------------------------------
def connectMC(type, nameSrc, nameTgt, projType, redundancy, RedundancyType, LCGDelays=True, gain=1., source_channels = None, stochastic_delays=None, verbose=False):

  def printv(text):
    if verbose:
      print(text)

  printv("* connecting "+nameSrc+" -> "+nameTgt+" with "+projType+" "+type+" connection")

  if source_channels == None:
    # if not specified, assume that the connection originates from all channels
    source_channels = list(range(len(Pop[nameSrc])))

  if RedundancyType == 'inDegreeAbs':
    # inDegree is already provided in the right form
    inDegree = float(redundancy)
  elif RedundancyType == 'outDegreeAbs':
    #### fractional outDegree is expressed as a fraction of max axo-dendritic contacts
    inDegree = get_frac(1./redundancy, nameSrc, nameTgt, neuronCounts[nameSrc], neuronCounts[nameTgt], verbose=verbose)
  elif RedundancyType == 'outDegreeCons':
    #### fractional outDegree is expressed as a ratio of min/max axo-dendritic contacts
    inDegree = get_frac(redundancy, nameSrc, nameTgt, neuronCounts[nameSrc], neuronCounts[nameTgt], useMin=True, verbose=verbose)
  else:
    raise KeyError('`RedundancyType` should be one of `inDegreeAbs`, `outDegreeAbs`, or `outDegreeCons`.')

  # check if in degree acceptable (not larger than number of neurons in the source nucleus)
  if projType == 'focused' and inDegree > nbSim[nameSrc]:
    printv("/!\ WARNING: required 'in degree' ("+str(inDegree)+") larger than number of neurons in individual source channels ("+str(nbSim[nameSrc])+"), thus reduced to the latter value")
    inDegree = nbSim[nameSrc]
  if projType == 'diffuse' and inDegree  > nbSim[nameSrc]*len(source_channels):
    printv("/!\ WARNING: required 'in degree' ("+str(inDegree)+") larger than number of neurons in the overall source population ("+str(nbSim[nameSrc]*len(source_channels))+"), thus reduced to the latter value")
    inDegree = nbSim[nameSrc]*len(source_channels)

  if inDegree == 0.:
    printv("/!\ WARNING: non-existent connection strength, will skip")
    return

  global AMPASynapseCounter

  inDegree = inDegree * (float(len(source_channels)) / float(len(Pop[nameSrc])))

  # prepare receptor type lists:
  if type == 'ex':
    lRecType = ['AMPA','NMDA']
    AMPASynapseCounter = AMPASynapseCounter + 1
    lbl = AMPASynapseCounter # needs to add NMDA later
  elif type == 'AMPA':
    lRecType = ['AMPA']
    lbl = 0
  elif type == 'NMDA':
    lRecType = ['NMDA']
    lbl = 0
  elif type == 'in':
    lRecType = ['GABA']
    lbl = 0
  else:
    raise KeyError('Undefined connexion type: '+type)

  # compute the global weight of the connection, for each receptor type:
  W = computeW(lRecType, nameSrc, nameTgt, inDegree, gain, verbose=False)

  printv("  W="+str(W)+" and inDegree="+str(inDegree))

  ## check whether a connection map has already been drawn or not:
  #if nameSrc+'->'+nameTgt in ConnectMap:
  #  #print "Using existing connection map"
  #  loadConnectMap = True
  #else:
  #  #print "Will create a connection map"
  #  loadConnectMap = False
  #  ConnectMap[nameSrc+'->'+nameTgt] = [[] for i in range(len(Pop[nameTgt]))]

  # determine which transmission delay to use:
  if LCGDelays:
    delay = tau[nameSrc+'->'+nameTgt]
  else:
    delay = 1.

  if projType == 'focused': # if projections focused, input come only from the same channel as tgtChannel
     for src_channel in source_channels: # for each relevant channel of the Source nucleus
       mass_connect(Pop[nameSrc][src_channel], Pop[nameTgt][src_channel-source_channels[0]], lbl, inDegree, recType[lRecType[0]], W[lRecType[0]], delay, stochastic_delays = stochastic_delays)
  elif projType == 'diffuse': # if projections diffused, input connections are shared among each possible input channel equally
    for src_channel in source_channels: # for each relevant channel of the Source nucleus
      for tgt_channel in range(len(Pop[nameTgt])): # for each channel of the Target nucleus
        mass_connect(Pop[nameSrc][src_channel], Pop[nameTgt][tgt_channel], lbl, inDegree/len(Pop[nameTgt]), recType[lRecType[0]], W[lRecType[0]], delay, stochastic_delays = stochastic_delays)

  if type == 'ex':
    # mirror the AMPA connection with similarly connected NMDA connections
    for src_channel in source_channels: # for each relevant channel of the Source nucleus
      mass_mirror(Pop[nameSrc][src_channel], lbl, recType['NMDA'], W['NMDA'], delay, stochastic_delays = stochastic_delays)

  return W

#-------------------------------------------------------------------------------
# returns the minimal & maximal numbers of distinct input neurons for one connection
#-------------------------------------------------------------------------------
def get_input_range(nameSrc, nameTgt, cntSrc, cntTgt, verbose=False):
  if nameSrc=='CSN' or nameSrc=='PTN':
    nu = alpha[nameSrc+'->'+nameTgt]
    nu0 = 0
    if verbose:
      print(('\tMaximal number of distinct input neurons (nu): '+str(nu)))
      print('\tMinimal number of distinct input neurons     : unknown (set to 0)')
  else:
    nu = cntSrc / float(cntTgt) * P[nameSrc+'->'+nameTgt] * alpha[nameSrc+'->'+nameTgt]
    nu0 = cntSrc / float(cntTgt) * P[nameSrc+'->'+nameTgt]
    if verbose:
      print(('\tMaximal number of distinct input neurons (nu): '+str(nu)))
      print(('\tMinimal number of distinct input neurons     : '+str(nu0)))
  return [nu0, nu]

#-------------------------------------------------------------------------------
# computes the inDegree as a fraction of maximal possible inDegree
# FractionalOutDegree: outDegree, expressed as a fraction
#-------------------------------------------------------------------------------
def get_frac(FractionalOutDegree, nameSrc, nameTgt, cntSrc, cntTgt, useMin=False, verbose=False):
  if useMin == False:
    # 'FractionalOutDegree' is taken to be relative to the maximal number of axo-dendritic contacts
    inDegree = get_input_range(nameSrc, nameTgt, cntSrc, cntTgt, verbose=verbose)[1] * FractionalOutDegree
  else:
    # 'FractionalOutDegree' is taken to be relative to the maximal number of axo-dendritic contacts and their minimal number
    r = get_input_range(nameSrc, nameTgt, cntSrc, cntTgt, verbose=verbose)
    inDegree = (r[1] - r[0]) * FractionalOutDegree + r[0]
  if verbose:
    print(('\tConverting the fractional outDegree of '+nameSrc+' -> '+nameTgt+' from '+str(FractionalOutDegree)+' to inDegree neuron count: '+str(round(inDegree, 2))+' (relative to minimal value possible? '+str(useMin)+')'))
  return inDegree

#-------------------------------------------------------------------------------
# computes the weight of a connection, based on LG14 parameters
#-------------------------------------------------------------------------------
def computeW(listRecType, nameSrc, nameTgt, inDegree, gain=1.,verbose=False):
  nu = get_input_range(nameSrc, nameTgt, neuronCounts[nameSrc], neuronCounts[nameTgt], verbose=verbose)[1]
  if verbose:
    print('\tCompare with the effective chosen inDegree   :',str(inDegree))

  # attenuation due to the distance from the receptors to the soma of tgt:
  attenuation = cosh(LX[nameTgt]*(1-p[nameSrc+'->'+nameTgt])) / cosh(LX[nameTgt])

  w={}
  for r in listRecType:
    w[r] = nu / float(inDegree) * attenuation * wPSP[recType[r]-1] * gain

  return w

#-------------------------------------------------------------------------------

# Acceptable firing rate ranges (FRR) in normal and deactivation experiments
# extracted from LG14 Table 5

FRRNormal = {'MSN': [0,1],
             'FSI': [7.8,14.0], # the refined constraint of 10.9 +/- 3.1 Hz was extracted from the following papers: Adler et al., 2016; Yamada et al., 2016 (summarizing date from three different experiments); and Marche and Apicella, 2017
             'STN': [15.2,22.8],
             'GPe': [55.7,74.5],
             'Arky': [55.7,74.5],
             'Prot': [55.7,74.5],
             'GPi': [59.1,79.5],
             }

FRRGPi = {'AMPA+NMDA+GABAA':[53.4,96.8],
          'NMDA':[27.2451,78.6255],
          'NMDA+AMPA':[6.811275,52.364583],
          'AMPA':[5.7327,66.0645],
          'GABAA':[44.1477,245.8935],
          }

FRRGPe = {'AMPA':[4.2889,58.7805],
          'AMPA+GABAA':[10.0017148,137.076126],
          'NMDA':[29.5767,61.1645],
          'GABAA':[74.8051,221.4885],
          }

FRRAnt = {'Arky':FRRGPe,'Prot':FRRGPe,'GPe':FRRGPe,'GPi':FRRGPi}

# imported from Chadoeuf "connexweights"
# All the parameters needed to replicate Lienard model
#
#-------------------------


# fixed parameters
A_GABA=-0.25 # mV
A_AMPA= 1.
A_NMDA= 0.025
D_GABA=5./exp(1)   # ms ; /e because Dn is peak half-time in LG14, while it is supposed to be tau_peak in NEST
D_AMPA=5./exp(1)
D_NMDA=100./exp(1)
Ri=200.E-2   # Ohms.m
Rm=20000.E-4 # Ohms.m^2

if params['splitGPe']:
  NUCLEI=['MSN','FSI','STN','Arky','Prot','GPi']
else:
  NUCLEI=['MSN','FSI','STN','GPe','GPi']

# Number of neurons in the real macaque brain
# one hemisphere only, based on Hardman et al. 2002 paper, except for striatum & CM/Pf
neuronCounts={'MSN': 26448.0E3,
              'FSI':   532.0E3,
              'STN':    77.0E3,
              'GPe':   251.0E3,
              'Arky':  251.0E3,
              'Prot':  251.0E3,
              'GPi':   143.0E3,
              'CMPf':   86.0E3,
              'CSN': None, 'PTN': None # prevents key error
             }

# Number of neurons that will be simulated

nbSim = {'MSN': 0.,
         'FSI': 0.,
         'STN': 0.,
         'GPe': 0.,
         'Arky': 0.,
         'Prot': 0.,
         'GPi': 0.,
         'CMPf':0.,
         'CSN': 0.,
         'PTN': 0.,}

# P(X->Y): probability that a given neuron from X projects to at least neuron of Y
P = {'MSN->GPe': 1.,
     'MSN->Arky': 1.,
     'MSN->Prot': 1.,
     'MSN->GPi': 0.82,
     'MSN->MSN': 1.,
     
     'FSI->MSN': 1.,
     'FSI->FSI': 1.,
     
     'STN->GPe':  0.83,
     'STN->Arky': 0.83,
     'STN->Prot': 0.83,
     'STN->GPi':  0.72,
     'STN->MSN':  0.17,
     'STN->FSI':  0.17,
     
     'GPe->STN': 1.,
     'GPe->GPe': 0.84,
     'GPe->GPi': 0.84,
     'GPe->MSN': 0.16,
     'GPe->FSI': 0.16,

     'Arky->Arky': 0.84,
     'Arky->Prot': 0.84,
     'Arky->MSN': 0.16,
     'Arky->FSI': 0.16,
     
     'Prot->STN': 1.,
     'Prot->Arky': 0.84,
     'Prot->Prot': 0.84,
     'Prot->GPi': 0.84,
     
     'CSN->MSN': 1.,
     'CSN->FSI': 1.,
     
     'PTN->MSN': 1.,
     'PTN->FSI': 1.,
     'PTN->STN': 1.,
     
     'CMPf->STN': 1.,
     'CMPf->MSN': 1.,
     'CMPf->FSI': 1.,
     'CMPf->GPe': 1.,
     'CMPf->Arky': 1.,
     'CMPf->Prot': 1.,
     'CMPf->GPi': 1.,}

# alpha X->Y: average number of synaptic contacts made by one neuron of X to one neuron of Y, when there is a connexion
# for the moment set from one specific parameterization, should be read from Jean's solution file
alpha = {'MSN->GPe':   171,
         'MSN->Arky':   171,
         'MSN->Prot':   171,
         'MSN->GPi':   210,
         'MSN->MSN':   210,
         
         'FSI->MSN':  4362,
         'FSI->FSI':   116,
         
         'STN->GPe':   428,
         'STN->Arky':   428,
         'STN->Prot':   428,
         'STN->GPi':   233,
         'STN->MSN':     0,
         'STN->FSI':    91,
         
         'GPe->STN':    19,
         'GPe->GPe':    38,
         'GPe->GPi':    16,
         'GPe->MSN':     0,
         'GPe->FSI':   353,

         'Arky->Arky':    38,
         'Arky->Prot':    38,
         'Arky->MSN':     0,
         'Arky->FSI':   353,
         
         'Prot->STN':    19,
         'Prot->Arky':    38,
         'Prot->Prot':    38,
         'Prot->GPi':    16,
         
         'CSN->MSN':   342, # here, represents directly \nu
         'CSN->FSI':   250, # here, represents directly \nu
         
         'PTN->MSN':     5, # here, represents directly \nu
         'PTN->FSI':     5, # here, represents directly \nu
         'PTN->STN':   259, # here, represents directly \nu
         
         'CMPf->MSN': 4965,
         'CMPf->FSI': 1053,
         'CMPf->STN':   76,
         'CMPf->GPe':   79,
         'CMPf->Arky':   79,
         'CMPf->Prot':   79,
         'CMPf->GPi':  131,}

# p(X->Y): relative distance on the dendrite from the soma, where neurons from X projects to neurons of Y
# Warning: p is not P!
p = {'MSN->GPe':  0.48,
     'MSN->Arky':  0.48,
     'MSN->Prot':  0.48,
     'MSN->GPi':  0.59,
     'MSN->MSN':  0.77,
     
     'FSI->MSN':  0.19,
     'FSI->FSI':  0.16,
     
     'STN->GPe':  0.30,
     'STN->Prot':  0.30,
     'STN->Arky':  0.30,
     'STN->GPi':  0.59,
     'STN->MSN':  0.16,
     'STN->FSI':  0.41,
     
     'GPe->STN':  0.58,
     'GPe->GPe':  0.01,
     'GPe->GPi':  0.13,
     'GPe->MSN':  0.06,
     'GPe->FSI':  0.58,

     'Arky->Arky':  0.01,
     'Arky->Prot':  0.01,
     'Arky->MSN':  0.06,
     'Arky->FSI':  0.58,
     
     'Prot->STN':  0.58,
     'Prot->Arky':  0.01,
     'Prot->Prot':  0.01,
     'Prot->GPi':  0.13,
     
     'CSN->MSN':  0.95,
     'CSN->FSI':  0.82,
     
     'PTN->MSN':  0.98,
     'PTN->FSI':  0.70,
     'PTN->STN':  0.97,
     
     'CMPf->STN': 0.46,
     'CMPf->MSN': 0.27,
     'CMPf->FSI': 0.06,
     'CMPf->GPe': 0.00,
     'CMPf->Arky': 0.00,
     'CMPf->Prot': 0.00,
     'CMPf->GPi': 0.48,}

# electrotonic constant L computation:
dx={'MSN':1.E-6,'FSI':1.5E-6,'STN':1.5E-6,'GPe':1.7E-6,'Arky':1.7E-6,'Prot':1.7E-6,'GPi':1.2E-6}
lx={'MSN':619E-6,'FSI':961E-6,'STN':750E-6,'GPe':865E-6,'Arky':865E-6,'Prot':865E-6,'GPi':1132E-6}
LX={}
for n in NUCLEI:
    LX[n]=lx[n]*sqrt((4*Ri)/(dx[n]*Rm))

# tau: communication delays
tau = {'MSN->GPe':    7.,
       'MSN->Arky':    7.,
       'MSN->Prot':    7.,
       'MSN->GPi':   11.,
       'MSN->MSN':    1.,
       
       'FSI->MSN':    1.,
       'FSI->FSI':    1.,
       
       'STN->GPe':    3.,
       'STN->Arky':    3.,
       'STN->Prot':    3.,
       'STN->GPi':    3.,
       'STN->MSN':    3.,
       'STN->FSI':    3.,
       
       'GPe->STN':   10.,
       'GPe->GPe':    1.,
       'GPe->GPi':    3.,
       'GPe->MSN':    3.,
       'GPe->FSI':    3.,
       
       'Arky->Arky':    1.,
       'Arky->Prot':    1.,
       'Arky->MSN':    3.,
       'Arky->FSI':    3.,
       
       'Prot->STN':   10.,
       'Prot->Arky':    1.,
       'Prot->Prot':    1.,
       'Prot->GPi':    3.,
       
       'CSN->MSN':    7.,
       'CSN->FSI':    7.,
       
       'PTN->MSN':    3.,
       'PTN->FSI':    3.,
       'PTN->STN':    3.,
       
       'CMPf->MSN':   7.,
       'CMPf->FSI':   7.,
       'CMPf->STN':   7.,#4
       'CMPf->GPe':   7.,#5
       'CMPf->Arky':   7.,#5
       'CMPf->Prot':   7.,#5
       'CMPf->GPi':   7.,#6
       }


# setting the 3 input ports for AMPA, NMDA and GABA receptor types
#-------------------------

nbPorts = 3
recType = {'AMPA':1,'NMDA':2,'GABA':3}
tau_syn = [D_AMPA, D_NMDA, D_GABA]
wPSP = [A_AMPA, A_NMDA, A_GABA]  # PSP amplitude (mV) ; A in LG14 notation

# parameterization of each neuronal type
#-------------------------

CommonParams = {'t_ref':         2.0,
                'V_m':           0.0,
                'V_th':         10.0, # dummy value to avoid NEST complaining about identical V_th and V_reset values
                'E_L':           0.0, # resting potential
                'V_reset':       0.0,
                'I_e':           0.0,
                'V_min':       -20.0, # as in HSG06
                'tau_syn':   tau_syn,}

commonNeuronType = initNeurons() # sets the default params of iaf_psc_alpha_mutisynapse neurons to CommonParams

MSNparams = {'tau_m':        13.0, # according to SBE12
             'V_th':         30.0, # value of the LG14 example model, table 9
             'C_m':          13.0  # so that R_m=1, C_m=tau_m
            }

FSIparams = {'tau_m':         3.1, # from http://www.neuroelectro.org/article/75165/
             'V_th':         16.0, # value of the LG14 example model, table 9
             'C_m':           3.1  # so that R_m=1, C_m=tau_m
            }

STNparams = {'tau_m':         6.0, # as in HSG06 (but they model rats...)
             'V_th':         26.0, # value of the LG14 example model, table 9
             'C_m':           6.0  # so that R_m=1, C_m=tau_m
            }

GPeparams = {'tau_m':        14.0, # 20 -> 14 based on Johnson & McIntyre 2008, JNphy)
             'V_th':         11.0, # value of the LG14 example model, table 9
             'C_m':          14.0  # so that R_m=1, C_m=tau_m
            }

Arkyparams = {'tau_m':        14.0, # 20 -> 14 based on Johnson & McIntyre 2008, JNphy)
             'V_th':         11.0, # value of the LG14 example model, table 9
             'C_m':          14.0  # so that R_m=1, C_m=tau_m
            }

Protparams = {'tau_m':        14.0, # 20 -> 14 based on Johnson & McIntyre 2008, JNphy)
             'V_th':         11.0, # value of the LG14 example model, table 9
             'C_m':          14.0  # so that R_m=1, C_m=tau_m
            }
GPiparams = {'tau_m':        14.0, # 20 -> 14 based on Johnson & McIntyre 2008, JNphy)
             'V_th':          6.0, # value of the LG14 example model, table 9
             'C_m':          14.0  # so that R_m=1, C_m=tau_m
            }


# dictionary of the parameterizations of each neuronal type
#-------------------------

BGparams = {'MSN':MSNparams,
            'FSI':FSIparams,
            'STN':STNparams,
            'GPe':GPeparams,
            'Arky':Arkyparams,
            'Prot':Protparams,
            'GPi':GPiparams}

  


def initNeuronTypes(params, commonNeuronType):
  neuronType = {}
  for pop in params:
    type_pop = nengo.neurons.LIF(tau_rc=params[pop]['tau_m']/1000., tau_ref=commonNeuronType.tau_ref, min_voltage=commonNeuronType.min_voltage)
    type_pop.label = 'neuron type ' + pop
    type_pop.params = copy.deepcopy(commonNeuronType.params)
    type_pop.params['V_th'] = params[pop]['V_th']

    neuronType[pop] = type_pop
  return neuronType

neuronTypes = initNeuronTypes(BGparams, commonNeuronType)


Pop = {}
Fake= {} # Fake contains the Poisson Generators, that will feed the parrot_neurons, stored in Pop
ConnectMap = {} # when connections are drawn, in "create()", they are stored here so as to be re-usable

# the dictionary used to store the desired discharge rates of the various Poisson generators that will be used as external inputs
rate = {'CSN':   2.  ,
        'PTN':  15.  ,
        'CMPf':  4.  ,
        'MSN':   0.25, # MSN and the following will be used when the corresponding nucleus is not explicitely simulated
        'FSI':  16.6 ,
        'STN':  14.3 ,
        'GPe':  62.6 ,
        'Arky':  62.6 ,
        'Prot':  62.6 ,
        'GPi':  64.2 ,
        }

# End LGneurons.py
#------------------------------------------



#------------------------------------------
# Start iniBG.py


import sys

import csv


#------------------------------------------
# Creates the populations of neurons necessary to simulate a BG circuit
#------------------------------------------
def createBG():
  #==========================
  # Creation of neurons
  #-------------------------
  print('\nCreating neurons\n================')

  # single or multi-channel?
  if params['nbCh'] == 1:
    def create_pop(*args, **kwargs):
      if 'nbCh' in list(kwargs.keys()):
        # remove the extra arg
        kwargs.pop("nbCh", None)
      create(*args, **kwargs)
    
  
    def update_Ie(p):
      Ie = Pop[p].Ie.pre
      if Ie.label[:2] != 'Ie':
        print(Ie)
        raise LookupError(p+'.Ie is not Ie node')
      Ie.output = params['Ie'+p]
  else:
    def create_pop(*args, **kwargs):
      if 'nbCh' not in list(kwargs.keys()):
        # enforce the default
        kwargs['nbCh'] = params['nbCh']
      createMC(*args, **kwargs)

    def update_Ie(p):
      for i in range(len(Pop[p])):
        Ie = Pop[p][i].Ie.pre
        if Ie.label[:2] != 'Ie':
          print(Ie)
          raise LookupError(p+str(i)+'.Ie is not Ie node')
        Ie.output = params['Ie'+p]
    

  nbSim['MSN'] = params['nbMSN']
  create_pop('MSN')
  update_Ie('MSN')

  nbSim['FSI'] = params['nbFSI']
  create_pop('FSI')
  update_Ie('FSI')

  nbSim['STN'] = params['nbSTN']
  create_pop('STN')
  update_Ie('STN')

  if params['splitGPe']:
    nbSim['Arky'] = params['nbArky']
    create_pop('Arky')
    update_Ie('Arky')
  
    nbSim['Prot'] = params['nbProt']
    create_pop('Prot')
    update_Ie('Prot')
  else:
    nbSim['GPe'] = params['nbGPe']
    create_pop('GPe')
    update_Ie('GPe')
      
      
  nbSim['GPi'] = params['nbGPi']
  create_pop('GPi')
  update_Ie('GPi')

  parrot = True # switch to False at your risks & perils...
  nbSim['CSN'] = params['nbCSN']
  if 'nbCues' in list(params.keys()):
    # cue channels are present
    CSNchannels = params['nbCh']+params['nbCues']
  else:
    CSNchannels = params['nbCh']
  create_pop('CSN', nbCh=CSNchannels, fake=True, parrot=parrot)

  nbSim['PTN'] = params['nbPTN']
  create_pop('PTN', fake=True, parrot=parrot)

  nbSim['CMPf'] = params['nbCMPf']
  create_pop('CMPf', fake=True, parrot=params['parrotCMPf']) # was: False

  print("Number of simulated neurons:", nbSim)

#------------------------------------------
# Connects the populations of a previously created multi-channel BG circuit
#------------------------------------------
def connectBG(antagInjectionSite,antag):

  # single or multi-channel?
  if params['nbCh'] == 1:
    connect_pop = lambda *args, **kwargs: connect(*args, RedundancyType=params['RedundancyType'], stochastic_delays=params['stochastic_delays'], **kwargs)
  else:
    def connect_pop(*args, **kwargs):
      if 'source_channels' not in list(kwargs.keys()):
        # enforce the default
        kwargs['source_channels'] = list(range(params['nbCh']))
      return connectMC(*args, RedundancyType=params['RedundancyType'], stochastic_delays=params['stochastic_delays'], **kwargs)

  #-------------------------
  # connection of populations
  #-------------------------
  print('\nConnecting neurons\n================')
  print("**",antag,"antagonist injection in",antagInjectionSite,"**")
  print('* MSN Inputs')
  if 'nbCues' not in list(params.keys()):
    # usual case: CSN have as the same number of channels than the BG nuclei
    CSN_MSN = connect_pop('ex','CSN','MSN', projType=params['cTypeCSNMSN'], redundancy=params['redundancyCSNMSN'], gain=params['GCSNMSN'])
  else:
    # special case: extra 'cue' channels that target MSN
    CSN_MSN = connect_pop('ex','CSN','MSN', projType=params['cTypeCSNMSN'], redundancy=params['redundancyCSNMSN'], gain=Params['GCSNMSN']/2., source_channels=list(range(params['nbCh'])))
    connect_pop('ex','CSN','MSN', projType='diffuse', redundancy=params['redundancyCSNMSN'], gain=params['GCSNMSN']/2., source_channels=list(range(params['nbCh'], params['nbCh']+params['nbCues'])))
  PTN_MSN = connect_pop('ex','PTN','MSN', projType=params['cTypePTNMSN'], redundancy= params['redundancyPTNMSN'], gain=params['GPTNMSN'])
  CMPf_MSN = connect_pop('ex','CMPf','MSN',projType=params['cTypeCMPfMSN'],redundancy= params['redundancyCMPfMSN'],gain=params['GCMPfMSN'])
  connect_pop('in','MSN','MSN', projType=params['cTypeMSNMSN'], redundancy= params['redundancyMSNMSN'], gain=params['GMSNMSN'])
  connect_pop('in','FSI','MSN', projType=params['cTypeFSIMSN'], redundancy= params['redundancyFSIMSN'], gain=params['GFSIMSN'])
  # some parameterizations from LG14 have no STN->MSN or GPe->MSN synaptic contacts
  if alpha['STN->MSN'] != 0:
    print("alpha['STN->MSN']",alpha['STN->MSN'])
    connect_pop('ex','STN','MSN', projType=params['cTypeSTNMSN'], redundancy= params['redundancySTNMSN'], gain=params['GSTNMSN'])
  if alpha['GPe->MSN'] != 0:
    if params['splitGPe']:
      print("alpha['Arky->MSN']",alpha['Arky->MSN'])
      connect_pop('in','Arky','MSN', projType=params['cTypeArkyMSN'], redundancy= params['redundancyArkyMSN'], gain=params['GArkyMSN'])
    else:
      print("alpha['GPe->MSN']",alpha['GPe->MSN'])
      connect_pop('in','GPe','MSN', projType=params['cTypeGPeMSN'], redundancy= params['redundancyGPeMSN'], gain=params['GGPeMSN'])

  print('* FSI Inputs')
  connect_pop('ex','CSN','FSI', projType=params['cTypeCSNFSI'], redundancy= params['redundancyCSNFSI'], gain=params['GCSNFSI'])
  connect_pop('ex','PTN','FSI', projType=params['cTypePTNFSI'], redundancy= params['redundancyPTNFSI'], gain=params['GPTNFSI'])
  if alpha['STN->FSI'] != 0:
    connect_pop('ex','STN','FSI', projType=params['cTypeSTNFSI'],redundancy= params['redundancySTNFSI'],gain=params['GSTNFSI'])
  if params['splitGPe']:
    connect_pop('in','Arky','FSI', projType=params['cTypeArkyFSI'], redundancy= params['redundancyArkyFSI'], gain=params['GArkyFSI'])
  else:
    connect_pop('in','GPe','FSI', projType=params['cTypeGPeFSI'], redundancy= params['redundancyGPeFSI'], gain=params['GGPeFSI'])
    
  connect_pop('ex','CMPf','FSI',projType=params['cTypeCMPfFSI'],redundancy= params['redundancyCMPfFSI'],gain=params['GCMPfFSI'])
  connect_pop('in','FSI','FSI', projType=params['cTypeFSIFSI'], redundancy= params['redundancyFSIFSI'], gain=params['GFSIFSI'])

  print('* STN Inputs')
  connect_pop('ex','PTN','STN', projType=params['cTypePTNSTN'], redundancy= params['redundancyPTNSTN'],  gain=params['GPTNSTN'])
  connect_pop('ex','CMPf','STN',projType=params['cTypeCMPfSTN'],redundancy= params['redundancyCMPfSTN'], gain=params['GCMPfSTN'])
  if params['splitGPe']:
    connect_pop('in','Prot','STN', projType=params['cTypeProtSTN'], redundancy= params['redundancyProtSTN'],  gain=params['GProtSTN'])
  else:
    connect_pop('in','GPe','STN', projType=params['cTypeGPeSTN'], redundancy= params['redundancyGPeSTN'],  gain=params['GGPeSTN'])
  
  if params['splitGPe']:
      print('* Arky Inputs')
      if 'fakeArkyRecurrent' not in list(params.keys()):
        # usual case: Arky's recurrent collaterals are handled normally
        Arky_recurrent_source = 'Arky'
      else:
        # here collaterals are simulated with Poisson train spikes firing at the frequency given by params['fakeArkyRecurrent']
        rate['Fake_Arky'] = float(params['fakeArkyRecurrent'])
        for nucleus_dict in [nbSim, neuronCounts]:
          nucleus_dict['Fake_Arky'] = nucleus_dict['Arky']
        for connection_dict in [P, alpha, p, tau]:
          connection_dict['Fake_Arky->Arky'] = connection_dict['Arky->Arky']
        if params['nbCh'] == 1:
          create('Fake_Arky', fake=True, parrot=True)
        else:
          createMC('Fake_Arky', params['nbCh'], fake=True, parrot=True)
        Arky_recurrent_source = 'Fake_Arky'
      if antagInjectionSite == 'GPe':
        if   antag == 'AMPA':
          connect_pop('NMDA','CMPf','Arky',projType=params['cTypeCMPfArky'],redundancy= params['redundancyCMPfArky'],gain=params['GCMPfArky'])
          connect_pop('NMDA','STN','Arky', projType=params['cTypeSTNArky'], redundancy= params['redundancySTNArky'], gain=params['GSTNArky'])
          connect_pop('in','MSN','Arky',   projType=params['cTypeMSNArky'], redundancy= params['redundancyMSNArky'], gain=params['GMSNArky'])
          connect_pop('in','Prot','Arky', projType=params['cTypeProtArky'], redundancy= params['redundancyProtArky'], gain=params['GProtArky'])
          connect_pop('in', Arky_recurrent_source, 'Arky', projType=params['cTypeArkyArky'], redundancy= params['redundancyArkyArky'], gain=params['GArkyArky'])
        elif antag == 'NMDA':
          connect_pop('AMPA','CMPf','Arky',projType=params['cTypeCMPfArky'],redundancy= params['redundancyCMPfArky'],gain=params['GCMPfArky'])
          connect_pop('AMPA','STN','Arky', projType=params['cTypeSTNArky'], redundancy= params['redundancySTNArky'], gain=params['GSTNArky'])
          connect_pop('in','MSN','Arky',   projType=params['cTypeMSNArky'], redundancy= params['redundancyMSNArky'], gain=params['GMSNArky'])
          connect_pop('in','Prot','Arky', projType=params['cTypeProtArky'], redundancy= params['redundancyProtArky'], gain=params['GProtArky'])
          connect_pop('in', Arky_recurrent_source, 'Arky', projType=params['cTypeArkyArky'], redundancy= params['redundancyArkyArky'], gain=params['GArkyArky'])
        elif antag == 'AMPA+GABAA':
          connect_pop('NMDA','CMPf','Arky',projType=params['cTypeCMPfArky'],redundancy= params['redundancyCMPfArky'],gain=params['GCMPfArky'])
          connect_pop('NMDA','STN','Arky', projType=params['cTypeSTNArky'], redundancy= params['redundancySTNArky'], gain=params['GSTNArky'])
        elif antag == 'GABAA':
          connect_pop('ex','CMPf','Arky',projType=params['cTypeCMPfArky'],redundancy= params['redundancyCMPfArky'],gain=params['GCMPfArky'])
          connect_pop('ex','STN','Arky', projType=params['cTypeSTNArky'], redundancy= params['redundancySTNArky'], gain=params['GSTNArky'])
        else:
          print(antagInjectionSite,": unknown antagonist experiment:",antag)
      else:
        connect_pop('ex','CMPf','Arky',projType=params['cTypeCMPfArky'],redundancy= params['redundancyCMPfArky'],gain=params['GCMPfArky'])
        connect_pop('ex','STN','Arky', projType=params['cTypeSTNArky'], redundancy= params['redundancySTNArky'], gain=params['GSTNArky'])
        connect_pop('in','MSN','Arky', projType=params['cTypeMSNArky'], redundancy= params['redundancyMSNArky'], gain=params['GMSNArky'])
        connect_pop('in','Prot','Arky', projType=params['cTypeProtArky'], redundancy= params['redundancyProtArky'], gain=params['GProtArky'])
        connect_pop('in', Arky_recurrent_source, 'Arky', projType=params['cTypeArkyArky'], redundancy= params['redundancyArkyArky'], gain=params['GArkyArky'])
        
      print('* Prot Inputs')
      if 'fakeProtRecurrent' not in list(params.keys()):
        # usual case: Prot's recurrent collaterals are handled normally
        Prot_recurrent_source = 'Prot'
      else:
        # here collaterals are simulated with Poisson train spikes firing at the frequency given by params['fakeProtRecurrent']
        rate['Fake_Prot'] = float(params['fakeProtRecurrent'])
        for nucleus_dict in [nbSim, neuronCounts]:
          nucleus_dict['Fake_Prot'] = nucleus_dict['Prot']
        for connection_dict in [P, alpha, p, tau]:
          connection_dict['Fake_Prot->Prot'] = connection_dict['Prot->Prot']
        if params['nbCh'] == 1:
          create('Fake_Prot', fake=True, parrot=True)
        else:
          createMC('Fake_Prot', params['nbCh'], fake=True, parrot=True)
        Prot_recurrent_source = 'Fake_Prot'
      if antagInjectionSite == 'GPe':
        if   antag == 'AMPA':
          connect_pop('NMDA','CMPf','Prot',projType=params['cTypeCMPfProt'],redundancy= params['redundancyCMPfProt'],gain=params['GCMPfProt'])
          connect_pop('NMDA','STN','Prot', projType=params['cTypeSTNProt'], redundancy= params['redundancySTNProt'], gain=params['GSTNProt'])
          connect_pop('in','MSN','Prot',   projType=params['cTypeMSNProt'], redundancy= params['redundancyMSNProt'], gain=params['GMSNProt'])
          connect_pop('in','Arky','Prot',   projType=params['cTypeArkyProt'], redundancy= params['redundancyArkyProt'], gain=params['GArkyProt'])
          connect_pop('in', Prot_recurrent_source, 'Prot', projType=params['cTypeProtProt'], redundancy= params['redundancyProtProt'], gain=params['GProtProt'])
        elif antag == 'NMDA':
          connect_pop('AMPA','CMPf','Prot',projType=params['cTypeCMPfProt'],redundancy= params['redundancyCMPfProt'],gain=params['GCMPfProt'])
          connect_pop('AMPA','STN','Prot', projType=params['cTypeSTNProt'], redundancy= params['redundancySTNProt'], gain=params['GSTNProt'])
          connect_pop('in','MSN','Prot',   projType=params['cTypeMSNProt'], redundancy= params['redundancyMSNProt'], gain=params['GMSNProt'])
          connect_pop('in','Arky','Prot',   projType=params['cTypeArkyProt'], redundancy= params['redundancyArkyProt'], gain=params['GArkyProt'])
          connect_pop('in', Prot_recurrent_source, 'Prot', projType=params['cTypeProtProt'], redundancy= params['redundancyProtProt'], gain=params['GProtProt'])
        elif antag == 'AMPA+GABAA':
          connect_pop('NMDA','CMPf','Prot',projType=params['cTypeCMPfProt'],redundancy= params['redundancyCMPfProt'],gain=params['GCMPfProt'])
          connect_pop('NMDA','STN','Prot', projType=params['cTypeSTNProt'], redundancy= params['redundancySTNProt'], gain=params['GSTNProt'])
        elif antag == 'GABAA':
          connect_pop('ex','CMPf','Prot',projType=params['cTypeCMPfProt'],redundancy= params['redundancyCMPfProt'],gain=params['GCMPfProt'])
          connect_pop('ex','STN','Prot', projType=params['cTypeSTNProt'], redundancy= params['redundancySTNProt'], gain=params['GSTNProt'])
        else:
          print(antagInjectionSite,": unknown antagonist experiment:",antag)
      else:
        connect_pop('ex','CMPf','Prot',projType=params['cTypeCMPfProt'],redundancy= params['redundancyCMPfProt'],gain=params['GCMPfProt'])
        connect_pop('ex','STN','Prot', projType=params['cTypeSTNProt'], redundancy= params['redundancySTNProt'], gain=params['GSTNProt'])
        connect_pop('in','MSN','Prot', projType=params['cTypeMSNProt'], redundancy= params['redundancyMSNProt'], gain=params['GMSNProt'])
        connect_pop('in','Arky','Prot',   projType=params['cTypeArkyProt'], redundancy= params['redundancyArkyProt'], gain=params['GArkyProt'])
        connect_pop('in', Prot_recurrent_source, 'Prot', projType=params['cTypeProtProt'], redundancy= params['redundancyProtProt'], gain=params['GProtProt'])
        
  else:
      print('* GPe Inputs')
      if 'fakeGPeRecurrent' not in list(params.keys()):
        # usual case: GPe's recurrent collaterals are handled normally
        GPe_recurrent_source = 'GPe'
      else:
        # here collaterals are simulated with Poisson train spikes firing at the frequency given by params['fakeGPeRecurrent']
        rate['Fake_GPe'] = float(params['fakeGPeRecurrent'])
        for nucleus_dict in [nbSim, neuronCounts]:
          nucleus_dict['Fake_GPe'] = nucleus_dict['GPe']
        for connection_dict in [P, alpha, p, tau]:
          connection_dict['Fake_GPe->GPe'] = connection_dict['GPe->GPe']
        if params['nbCh'] == 1:
          create('Fake_GPe', fake=True, parrot=True)
        else:
          createMC('Fake_GPe', params['nbCh'], fake=True, parrot=True)
        GPe_recurrent_source = 'Fake_GPe'
      if antagInjectionSite == 'GPe':
        if   antag == 'AMPA':
          connect_pop('NMDA','CMPf','GPe',projType=params['cTypeCMPfGPe'],redundancy= params['redundancyCMPfGPe'],gain=params['GCMPfGPe'])
          connect_pop('NMDA','STN','GPe', projType=params['cTypeSTNGPe'], redundancy= params['redundancySTNGPe'], gain=params['GSTNGPe'])
          connect_pop('in','MSN','GPe',   projType=params['cTypeMSNGPe'], redundancy= params['redundancyMSNGPe'], gain=params['GMSNGPe'])
          connect_pop('in', GPe_recurrent_source, 'GPe', projType=params['cTypeGPeGPe'], redundancy= params['redundancyGPeGPe'], gain=params['GGPeGPe'])
        elif antag == 'NMDA':
          connect_pop('AMPA','CMPf','GPe',projType=params['cTypeCMPfGPe'],redundancy= params['redundancyCMPfGPe'],gain=params['GCMPfGPe'])
          connect_pop('AMPA','STN','GPe', projType=params['cTypeSTNGPe'], redundancy= params['redundancySTNGPe'], gain=params['GSTNGPe'])
          connect_pop('in','MSN','GPe',   projType=params['cTypeMSNGPe'], redundancy= params['redundancyMSNGPe'], gain=params['GMSNGPe'])
          connect_pop('in', GPe_recurrent_source, 'GPe', projType=params['cTypeGPeGPe'], redundancy= params['redundancyGPeGPe'], gain=params['GGPeGPe'])
        elif antag == 'AMPA+GABAA':
          connect_pop('NMDA','CMPf','GPe',projType=params['cTypeCMPfGPe'],redundancy= params['redundancyCMPfGPe'],gain=params['GCMPfGPe'])
          connect_pop('NMDA','STN','GPe', projType=params['cTypeSTNGPe'], redundancy= params['redundancySTNGPe'], gain=params['GSTNGPe'])
        elif antag == 'GABAA':
          connect_pop('ex','CMPf','GPe',projType=params['cTypeCMPfGPe'],redundancy= params['redundancyCMPfGPe'],gain=params['GCMPfGPe'])
          connect_pop('ex','STN','GPe', projType=params['cTypeSTNGPe'], redundancy= params['redundancySTNGPe'], gain=params['GSTNGPe'])
        else:
          print(antagInjectionSite,": unknown antagonist experiment:",antag)
      else:
        connect_pop('ex','CMPf','GPe',projType=params['cTypeCMPfGPe'],redundancy= params['redundancyCMPfGPe'],gain=params['GCMPfGPe'])
        connect_pop('ex','STN','GPe', projType=params['cTypeSTNGPe'], redundancy= params['redundancySTNGPe'], gain=params['GSTNGPe'])
        connect_pop('in','MSN','GPe', projType=params['cTypeMSNGPe'], redundancy= params['redundancyMSNGPe'], gain=params['GMSNGPe'])
        connect_pop('in', GPe_recurrent_source, 'GPe', projType=params['cTypeGPeGPe'], redundancy= params['redundancyGPeGPe'], gain=params['GGPeGPe'])

  print('* GPi Inputs')
  if antagInjectionSite =='GPi':
    if   antag == 'AMPA+NMDA+GABAA':
      pass
    elif antag == 'NMDA':
      connect_pop('in','MSN','GPi',   projType=params['cTypeMSNGPi'], redundancy= params['redundancyMSNGPi'], gain=params['GMSNGPi'])
      connect_pop('AMPA','STN','GPi', projType=params['cTypeSTNGPi'], redundancy= params['redundancySTNGPi'], gain=params['GSTNGPi'])
      if params['splitGPe']:
        connect_pop('in','Prot','GPi',   projType=params['cTypeProtGPi'], redundancy= params['redundancyProtGPi'], gain=params['GProtGPi'])
      else:
        connect_pop('in','GPe','GPi',   projType=params['cTypeGPeGPi'], redundancy= params['redundancyGPeGPi'], gain=params['GGPeGPi'])
      connect_pop('AMPA','CMPf','GPi',projType=params['cTypeCMPfGPi'],redundancy= params['redundancyCMPfGPi'],gain=params['GCMPfGPi'])
    elif antag == 'NMDA+AMPA':
      connect_pop('in','MSN','GPi', projType=params['cTypeMSNGPi'],redundancy= params['redundancyMSNGPi'], gain=params['GMSNGPi'])
      if params['splitGPe']:
        connect_pop('in','Prot','GPi',   projType=params['cTypeProtGPi'], redundancy= params['redundancyProtGPi'], gain=params['GProtGPi'])
      else:
        connect_pop('in','GPe','GPi',   projType=params['cTypeGPeGPi'], redundancy= params['redundancyGPeGPi'], gain=params['GGPeGPi'])
    elif antag == 'AMPA':
      connect_pop('in','MSN','GPi',   projType=params['cTypeMSNGPi'], redundancy= params['redundancyMSNGPi'], gain=params['GMSNGPi'])
      connect_pop('NMDA','STN','GPi', projType=params['cTypeSTNGPi'], redundancy= params['redundancySTNGPi'], gain=params['GSTNGPi'])
      if params['splitGPe']:
        connect_pop('in','Prot','GPi',   projType=params['cTypeProtGPi'], redundancy= params['redundancyProtGPi'], gain=params['GProtGPi'])
      else:
        connect_pop('in','GPe','GPi',   projType=params['cTypeGPeGPi'], redundancy= params['redundancyGPeGPi'], gain=params['GGPeGPi'])
      connect_pop('NMDA','CMPf','GPi',projType=params['cTypeCMPfGPi'],redundancy= params['redundancyCMPfGPi'],gain=params['GCMPfGPi'])
    elif antag == 'GABAA':
      connect_pop('ex','STN','GPi', projType=params['cTypeSTNGPi'], redundancy= params['redundancySTNGPi'], gain=params['GSTNGPi'])
      connect_pop('ex','CMPf','GPi',projType=params['cTypeCMPfGPi'],redundancy= params['redundancyCMPfGPi'],gain=params['GCMPfGPi'])
    else:
      print(antagInjectionSite,": unknown antagonist experiment:",antag)
  else:
    connect_pop('in','MSN','GPi', projType=params['cTypeMSNGPi'], redundancy= params['redundancyMSNGPi'], gain=params['GMSNGPi'])
    connect_pop('ex','STN','GPi', projType=params['cTypeSTNGPi'], redundancy= params['redundancySTNGPi'], gain=params['GSTNGPi'])
    if params['splitGPe']:
      connect_pop('in','Prot','GPi',   projType=params['cTypeProtGPi'], redundancy= params['redundancyProtGPi'], gain=params['GProtGPi'])
    else:
      connect_pop('in','GPe','GPi',   projType=params['cTypeGPeGPi'], redundancy= params['redundancyGPeGPi'], gain=params['GGPeGPi'])
    connect_pop('ex','CMPf','GPi',projType=params['cTypeCMPfGPi'],redundancy= params['redundancyCMPfGPi'],gain=params['GCMPfGPi'])

  base_weights = {'CSN_MSN': CSN_MSN, 'PTN_MSN': PTN_MSN, 'CMPf_MSN': CMPf_MSN}

  return base_weights

#------------------------------------------
# gets the nuclei involved in deactivation experiments in GPe/GPi
#------------------------------------------
def get_afferents(a):
  if params['splitGPe']:
    GABA_afferents = ['MSN', 'Arky', 'Prot'] # afferents with gabaergic connections
  else:
    GABA_afferents = ['MSN', 'GPe'] # afferents with gabaergic connections
    
  GLUT_afferents = ['STN', 'CMPf'] # afferents with glutamatergic connections
  if a == 'GABAA':
    afferents = GABA_afferents
  elif a == 'AMPA+GABAA':
    afferents = GABA_afferents + GLUT_afferents
  elif a == 'AMPA+NMDA+GABAA':
    afferents = GABA_afferents + GLUT_afferents
  else:
    afferents = GLUT_afferents
  return afferents

#------------------------------------------
# deactivate connections based on antagonist experiment
#------------------------------------------
def deactivate(site, a):
  ww = {}
  for src in get_afferents(a):
    ww[src] = None
    for rec in a.split('+'):
      w = alter_connection(src, site, rec, 0)
      if ww[src] == None:
        ww[src] = w # keep the original weights only once
  return ww

#------------------------------------------
# reactivate connections based on antagonist experiment
#------------------------------------------
def reactivate(site, a, ww):
  for src in get_afferents(a):
    for rec in a.split('+'):
      alter_connection(src, site, rec, ww[src])

#------------------------------------------
# Instantiate the BG network according to the `params` dictionnary
# For now, this instantiation respects the hardcoded antagonist injection sites
# In the future, these will be handled by changing the network weights
#------------------------------------------
def instantiate_BG(params={}, antagInjectionSite='none', antag=''):

  net = nengo.Network(seed=params['nestSeed'])
  net.pops = {}

  dataPath='log/'
  initNeurons()

  print('/!\ Using the following LG14 parameterization',params['LG14modelID'])
  loadLG14params(params['LG14modelID'])
  loadThetaFromCustomparams(params)

  # We check that all the necessary parameters have been defined. They should be in the modelParams.py file.
  # If one of them misses, we exit the program.
  if params['splitGPe']:
    necessaryParams=['nbCh','nbMSN','nbFSI','nbSTN','nbGPe','nbArky','nbProt','nbGPi','nbCSN','nbPTN','nbCMPf',
                     'IeMSN','IeFSI','IeSTN','IeGPe','IeArky','IeProt','IeGPi',
                     'GCSNMSN','GPTNMSN','GCMPfMSN','GMSNMSN','GFSIMSN','GSTNMSN','GGPeMSN','GArkyMSN',
                     'GCSNFSI','GPTNFSI','GSTNFSI','GGPeFSI','GArkyFSI','GCMPfFSI','GFSIFSI',
                     'GPTNSTN','GCMPfSTN','GGPeSTN','GProtSTN',
                     'GCMPfGPe','GSTNGPe','GMSNGPe','GGPeGPe',
                     'GCMPfArky','GSTNArky','GMSNArky','GArkyArky','GProtArky',
                     'GCMPfProt','GSTNProt','GMSNProt','GProtProt','GArkyProt',
                     'GMSNGPi','GSTNGPi','GGPeGPi','GProtGPi','GCMPfGPi',
                     'redundancyCSNMSN','redundancyPTNMSN','redundancyCMPfMSN','redundancyMSNMSN','redundancyFSIMSN','redundancySTNMSN','redundancyGPeMSN','redundancyArkyMSN',
                     'redundancyCSNFSI','redundancyPTNFSI','redundancySTNFSI','redundancyGPeFSI','redundancyArkyFSI','redundancyCMPfFSI','redundancyFSIFSI',
                     'redundancyPTNSTN','redundancyCMPfSTN','redundancyGPeSTN','redundancyProtSTN',
                     'redundancyCMPfGPe','redundancySTNGPe','redundancyMSNGPe','redundancyGPeGPe',
                     'redundancyCMPfArky','redundancySTNArky','redundancyMSNArky','redundancyArkyArky','redundancyProtArky',
                     'redundancyCMPfProt','redundancySTNProt','redundancyMSNProt','redundancyProtProt','redundancyArkyProt',
                     'redundancyMSNGPi','redundancySTNGPi','redundancyGPeGPi','redundancyProtGPi','redundancyCMPfGPi',]
  else:
    necessaryParams=['nbCh','nbMSN','nbFSI','nbSTN','nbGPe','nbGPi','nbCSN','nbPTN','nbCMPf',
                     'IeMSN','IeFSI','IeSTN','IeGPe','IeGPi',
                     'GCSNMSN','GPTNMSN','GCMPfMSN','GMSNMSN','GFSIMSN','GSTNMSN','GGPeMSN',
                     'GCSNFSI','GPTNFSI','GSTNFSI','GGPeFSI','GCMPfFSI','GFSIFSI',
                     'GPTNSTN','GCMPfSTN','GGPeSTN',
                     'GCMPfGPe','GSTNGPe','GMSNGPe','GGPeGPe',
                     'GMSNGPi','GSTNGPi','GGPeGPi','GCMPfGPi',
                     'redundancyCSNMSN','redundancyPTNMSN','redundancyCMPfMSN','redundancyMSNMSN','redundancyFSIMSN','redundancySTNMSN','redundancyGPeMSN',
                     'redundancyCSNFSI','redundancyPTNFSI','redundancySTNFSI','redundancyGPeFSI','redundancyCMPfFSI','redundancyFSIFSI',
                     'redundancyPTNSTN','redundancyCMPfSTN','redundancyGPeSTN',
                     'redundancyCMPfGPe','redundancySTNGPe','redundancyMSNGPe','redundancyGPeGPe',
                     'redundancyMSNGPi','redundancySTNGPi','redundancyGPeGPi','redundancyCMPfGPi',]
  for np in necessaryParams:
    if np not in params:
      raise KeyError('Missing parameter: '+np)

  #------------------------
  # creation and connection of the neural populations
  #------------------------

  createBG()
  return connectBG(antagInjectionSite,antag)

# End iniBG.py
#------------------------------------------

#-----------------------------------------------------------------------
def create_sBCBG(dimensions): # used

  params['nbCh'] = dimensions

  deactivationTests = False

  if len(sys.argv) >= 2:
    print("Command Line Parameters")
    paramKeys = ['LG14modelID',
                 'nbMSN',
                 'nbFSI',
                 'nbSTN',
                 'nbGPe',
                 'nbGPi',
                 'nbCSN',
                 'nbPTN',
                 'nbCMPf',
                 'GMSN',
                 'GFSI',
                 'GSTN',
                 'GGPe',
                 'GGPi', 
                 'IeGPe',
                 'IeGPi',
                 'inDegCSNMSN',
                 'inDegPTNMSN',
                 'inDegCMPfMSN',
                 'inDegFSIMSN',
                 'inDegMSNMSN', 
                 'inDegCSNFSI',
                 'inDegPTNFSI',
                 'inDegSTNFSI',
                 'inDegGPeFSI',
                 'inDegCMPfFSI',
                 'inDegFSIFSI',
                 'inDegPTNSTN',
                 'inDegCMPfSTN',
                 'inDegGPeSTN',
                 'inDegCMPfGPe',
                 'inDegSTNGPe',
                 'inDegMSNGPe',
                 'inDegGPeGPe',
                 'inDegMSNGPi',
                 'inDegSTNGPi',
                 'inDegGPeGPi',
                 'inDegCMPfGPi',
                 ]
    print(sys.argv)
    if len(sys.argv) == len(paramKeys)+1:
      print("Using command line parameters")
      print(sys.argv)
      i = 0
      for k in paramKeys:
        i+=1
        params[k] = float(sys.argv[i])
    else :
      print("Incorrect number of parameters:",len(sys.argv),"-",len(paramKeys),"expected")

  instantiate_BG(params, antagInjectionSite='none', antag='')

  return net
  

#---------------------------
#main()
