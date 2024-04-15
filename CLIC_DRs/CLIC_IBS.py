# Copyright 2014-2017 CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3), 
# copied verbatim in the file LICENCE.md.
# In applying this licence, CERN does not waive the privileges and immunities 
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

'''
Reference script for LHC at flat-bottom including 
a simple model for IBS.

Author: M. Zampetakis, 2023
'''

import os
import sys
import time
import yaml
import numpy as np
import pandas as pd
from pyprof import  timing
import matplotlib.pyplot as plt
from scipy.constants import c
from prettytable import PrettyTable

from lib.loadlattice import prepareTwiss
#from lib.IBSfunctions import NagaitsevIBS
from lib.sdelta_IBSfunctions import NagaitsevIBS
from lib.general_functions import EnergySpread, BunchLength

#from blond.plots.plot import Plot
from blond.beam.beam import Beam, Positron
from blond.input_parameters.ring import Ring
from blond.monitors.monitors import BunchMonitor
from blond.impedances.impedance_sources import InputTable
from blond.input_parameters.rf_parameters import RFStation
from blond.beam.profile import CutOptions, FitOptions, Profile
from blond.trackers.tracker import RingAndRFTracker, FullRingAndRF
from blond.toolbox.action import oscillation_amplitude_from_coordinates
from blond.impedances.impedance import InducedVoltageFreq, TotalInducedVoltage
from blond.beam.distributions import bigaussian, matched_from_distribution_function


# !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~! #
# !~~~~~~~~~~~~~~~~~~~~ CLIC Parameters ~~~~~~~~~~~~~~~~~~~! #
# !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~! #
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# !~~~~~~~~~~~~ Bunch parameters ~~~~~~~~~~~! #
N_b = config['bunch_intensity'] # Number of particles; [1]
N_p = int(config['n_mparts'])   # Number of macro-particles; [1]
tau_0 = config['tau']           # 4 sigma bunch length, 4 sigma [s]

# !~~~~~~~ Machine and RF parameters ~~~~~~~! #
C = 427.5                       # Machine circumference [m]
p = config['energy'] * 1e9      # Synchronous momentum [eV/c]
h = config['h']                 # Harmonic number
phi = 0.                        # RF synchronous phase
Vrf = config['V0max'] * 1e9    # RF voltage [V]

gamma_t = 87.70580              # Transition gamma
alpha   = 1 / gamma_t**2        # First order mom. comp. factor

# !~~~~~~~~~~~~ Tracking details ~~~~~~~~~~~! #
N_t    = config['N_turns']      # Number of turns to track
N_ibs  = config['IBS_stp']      # Number of turns to update IBS
dt_plt = 50

print(N_b, N_p, tau_0, C, p, h, Vrf, gamma_t, alpha)

# !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~! #
# !~~~~~~~~~~~~~~~~~~~ Simulation Setup ~~~~~~~~~~~~~~~~~~~! #
# !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~! #
print("Setting up the simulation...\n")

wrkDir = os.getcwd()
print(f"Working directory is {wrkDir}")
if not os.path.isdir(os.path.join(wrkDir,'output')):
    os.mkdir("output")
    print(f"Created output folder {os.path.join(wrkDir,'output')}")


# !~~~~~~~ Define General Parameters ~~~~~~~! #
ring = Ring(C, alpha, p, Positron(), N_t)
print("Ring initialized...")


# !~~~~~~ Define RF Station Parameters ~~~~~! #
rf = RFStation(ring, [h], [Vrf], [phi])
print("RF station initialized...")


# !~~~~~~ Define beam and distribution ~~~~~! #
beam = Beam(ring, N_p, N_b)

profile = Profile(beam, CutOptions(n_slices=100, cut_left=0, 
                    cut_right=rf.t_rf[0, 0]), 
                    FitOptions=FitOptions(fit_option='rms'))

# !~~~~~~~~~~~~~~~~ Trackers ~~~~~~~~~~~~~~~! #
rf_station_tracker = RingAndRFTracker(rf, beam, Profile=profile)
tracker = FullRingAndRF([rf_station_tracker])


# !~~~~~~~~~~~~~~~~ Matching ~~~~~~~~~~~~~~! #
matched_from_distribution_function(beam, tracker,
    distribution_type = 'gaussian', bunch_length = tau_0,
    distribution_variable = 'Action', bunch_length_fit = 'gaussian', 
    n_iterations= 10)

profile.track()
print(profile.bunchLength)
#sys.exit()

# !~~~~~ Define what to save in file ~~~~~~! #
bunchmonitor = BunchMonitor(ring, rf, beam, wrkDir + '/output/lhc_output_data', 
                            Profile=profile)

# !~~~~~~~~~~~~ Accelerator map ~~~~~~~~~~~~! #
map_ = [tracker] + [profile] + [bunchmonitor]
print("Map set")
print("")

# !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~! #
# !~~~~~~~~~~~~~~~~~~~~~~ IBS Setup ~~~~~~~~~~~~~~~~~~~~~~~! #
# !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~! #
twiss = prepareTwiss(config['twissfile'])
twiss['slip'] = rf.eta_0[0]

IBS = NagaitsevIBS()
IBS.set_beam_parameters(beam)
IBS.set_optic_functions(twiss)

analytic = True
if analytic:
    ttime = []
    emit_x_all   = []
    emit_y_all   = []
    tau_all  = []
    delta_E_all  = []
    emit_x = config['epsn_x'] / beam.gamma / beam.beta
    emit_y = config['epsn_y'] / beam.gamma / beam.beta
    bunch_length  = profile.bunchLength / 4. * c * IBS.betar
    #se = EnergySpread(IBS.Circu, 2852., IBS.EnTot, IBS.slip, bunch_length, IBS.betar, 4.5e-3, 0, 1)
    sigma_epsilon = np.std(beam.dE) / (IBS.EnTot * 1e9)
    sigma_delta   = sigma_epsilon / IBS.betar**2
    #print(se, sigma_epsilon, sigma_delta)
    #sys.exit()
    for turn in range(0, N_t):
        if turn % 1000 == 0: print(f'Turn = {turn}')

        if turn % N_ibs == 0: 
            #IBS.growth_rates(emit_x, emit_y, sigma_epsilon, bunch_length)
            IBS.growth_rates(emit_x, emit_y, sigma_delta, bunch_length)
            ttime.append(turn / IBS.frev)
            emit_x_all.append(emit_x)
            emit_y_all.append(emit_y)
            tau_all.append(bunch_length)
            delta_E_all.append(sigma_epsilon)

        #emit_x, emit_y, sigma_epsilon, bunch_length = IBS.emittance_evolution(emit_x, emit_y, sigma_epsilon,
        #                                                                      bunch_length, 1 / ring.f_rev[0])
        emit_x, emit_y, sigma_delta, bunch_length = IBS.emittance_evolution(emit_x, emit_y,
                                                                            sigma_delta,
                                                                            bunch_length,
                                                                            1 / ring.f_rev[0])
        
        sigma_epsilon = sigma_delta * IBS.betar**2
        #bunch_length = BunchLength(IBS.Circu, 2852., IBS.EnTot, IBS.slip, sigma_epsilon, IBS.betar, 4.5e-3, 0, 1)
        
    df = pd.DataFrame({'time': ttime, 
                       'epsn_x': np.array(emit_x_all) * IBS.betar * IBS.gammar, 
                       'epsn_y': np.array(emit_y_all) * IBS.betar * IBS.gammar, 
                       'tau_m': tau_all, 
                       'tau_ns': np.array(tau_all) * 4. / c / IBS.betar, 
                       'deltaE': delta_E_all})
    df.to_parquet("IBS_output_python.parquet")


emit_x = config['epsn_x'] / beam.gamma / beam.beta
emit_y = config['epsn_y'] / beam.gamma / beam.beta

evolution = {'time': np.zeros(int(N_t / N_ibs)), 
             'epsn_x': np.zeros(int(N_t / N_ibs)), 
             'epsn_y': np.zeros(int(N_t / N_ibs)), 
             'tau_ns': np.zeros(int(N_t / N_ibs)), 
             'deltaE': np.zeros(int(N_t / N_ibs)),
             'deltaE_test': np.zeros(int(N_t / N_ibs))}

indx = 0
for i in range(1, N_t+1):
    if (i-1) % 10 == 0: print(f'Turn = {i}')

    if ((i-1) % N_ibs == 0):
        IBS.calculate_longitudinal_kick(emit_x, emit_y, beam, profile)
        evolution['time'][indx]  = (i-1)/ IBS.frev
        evolution['epsn_x'][indx] = emit_x * IBS.betar * IBS.gammar
        evolution['epsn_y'][indx] = emit_y * IBS.betar * IBS.gammar
        evolution['tau_ns'][indx] = profile.bunchLength
        evolution['deltaE'][indx] = np.std(beam.dE[beam.id > 0])
        indx += 1

    IBS.track(profile, beam)
    emit_x, emit_y = IBS.emittance_evolution_2D(emit_x, emit_y, 1 / ring.f_rev[0])

    # Track
    tracker.track()
    profile.track()

df = pd.DataFrame(evolution)
df.to_parquet("IBS_output_BLonD.parquet")

sys.exit()



time = []
emit_x_all   = []
emit_y_all   = []
sigma_z_all  = []

emit_x = config['epsn_x'] / beam.gamma / beam.beta
emit_y = config['epsn_y'] / beam.gamma / beam.beta
# !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~! #
# !~~~~~~~~~~~~~~~~~~~~~~~ Tracking ~~~~~~~~~~~~~~~~~~~~~~~! #
# !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~! #
for i in range(1, N_t+1):

    # Plot has to be done before tracking (at least for cases with separatrix)
    if (i % dt_plt) == 0:
        print("Outputting at time step %d..." % i)
        print("   Beam momentum %.6e eV" % beam.momentum)
        print("   Beam gamma %3.3f" % beam.gamma)
        print("   Beam beta %3.3f" % beam.beta)
        print("   Beam energy %.6e eV" % beam.energy)
        print("   Four-times r.m.s. bunch length %.4e s" % (4.*beam.sigma_dt))
        print("   Gaussian bunch length %.4e s" % profile.bunchLength)
        print("")

    if ((i-1) % N_ibs == 0):
        #IBS.calculate_longitudinal_kick(emit_x, emit_y, beam)
        IBS.calculate_longitudinal_kick_V2(emit_x, emit_y, beam, profile)
        time.append((i-1)/ IBS.frev)
        emit_x_all.append(emit_x)
        emit_y_all.append(emit_y)
        sigma_z_all.append(profile.bunchLength)


    #IBS.track(profile, beam)
    IBS.track_V2(beam)
    emit_x, emit_y = IBS.emittance_evolution_2D(emit_x, emit_y, 1/ring.f_rev[0])

    # Track
    for m in map_:
        m.track()

    plt.plot(profile.bunchLength, 'o')
    # Define losses according to separatrix and/or longitudinal position
    beam.losses_separatrix(ring, rf)
    beam.losses_longitudinal_cut(0., rf.t_rf[0, 0])


df = pd.DataFrame({'tt': time, 
                   'epsn_x': np.array(emit_x_all) * IBS.betar * IBS.gammar, 
                   'epsn_y': np.array(emit_y_all) * IBS.betar * IBS.gammar, 
                   'bl_ns': np.array(sigma_z_all)})
df.to_parquet("IBS_output_BLonD.parquet")
print("Done!")

sys.exit()
