import nest
import nengo
from math import sqrt, cosh, exp, pi

# fixed parameters
A_GABA=-0.25 # mV
A_AMPA= 1.
A_NMDA= 0.025
D_GABA=5./exp(1)   # ms ; /e because Dn is peak half-time in LG14, while it is supposed to be tau_peak in NEST
D_AMPA=5./exp(1)
D_NMDA=100./exp(1)
Ri=200.E-2   # Ohms.m
Rm=20000.E-4 # Ohms.m^2

tau_syn = [D_AMPA, D_NMDA, D_GABA]
print('tau_syn', tau_syn)

CommonParams = {'t_ref':         2.0,
                'V_m':           0.0,
                'V_th':         10.0, # dummy value to avoid NEST complaining about identical V_th and V_reset values
                'E_L':           0.0,
                'V_reset':       0.0,
                'I_e':           0.0,
                'V_min':       -20.0, # as in HSG06
                'tau_syn':   tau_syn,}

nest.SetDefaults("iaf_psc_alpha_multisynapse", CommonParams)
print(nest.GetDefaults("iaf_psc_alpha_multisynapse"))
