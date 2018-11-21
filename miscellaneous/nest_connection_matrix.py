import nest
import numpy as np

m = 3
n = 5

a = nest.Create("iaf_psc_alpha",n,
            params={'V_th':          10., 
                    'I_e':           1.5, 
                    't_ref':         2., 
                    'V_min':         0., 
                    'tau_m':         20., 
                    'C_m':           20.,
                    'V_m':           0.0,
                    'E_L':           0.0,
                    'V_reset':       0.0})

b = nest.Create("iaf_psc_alpha",m,
            params={'V_th':          10., 
                    'I_e':           1.5, 
                    't_ref':         2., 
                    'V_min':         0., 
                    'tau_m':         20., 
                    'C_m':           20.,
                    'V_m':           0.0,
                    'E_L':           0.0,
                    'V_reset':       0.0})

deg = 5

#nest.Connect(a,b, {"rule": "fixed_indegree", "indegree": deg}, {'weight':1})
nest.Connect(a,b, {'rule': 'fixed_total_number', 'N': 15}, {'weight':1})

co = []
for elt in nest.GetConnections(a,b):
	co.append(elt[:2])
	print(elt)
co = np.array(co)
print(co)

for i in range(n+1,m+n+1):
	print(i)