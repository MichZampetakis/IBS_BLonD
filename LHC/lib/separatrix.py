import numpy as np
from blond.utils import bmath as bm
from blond.trackers.utilities import hamiltonian

def def_separatrix(Ring, RFStation, Beam, total_voltage=None):
    r"""Function to define the separatrix.
    Uses the single-RF sinusoidal Hamiltonian.

    Parameters
    ---------- 
    Ring : class
        A Ring type class
    RFStation : class
        An RFStation type class
    Beam : class
        A Beam type class
    total_voltage : float array
        Total voltage to be used if not single-harmonic RF

    Returns
    -------
    bool array
        True/False array for the given coordinates

    """
    
    counter = RFStation.counter[0]
    dt_sep = (np.pi - RFStation.phi_s[counter]
              - RFStation.phi_rf_d[0, counter]) / \
        RFStation.omega_rf[0, counter]

    Hsep = hamiltonian(Ring, RFStation, Beam, dt_sep, 0,
                       total_voltage=None)
    
    return Hsep


def is_in_separtrx(Ring, RFStation, Beam, Hsep, total_voltage=None):
    
    isin = bm.fabs(hamiltonian(Ring, RFStation, Beam,
                               Beam.dt, Beam.dE, total_voltage=None)) < bm.fabs(Hsep)
    lost_index = (isin == False)

    return lost_index
