import warnings
import numpy as np
from scipy.integrate import quad
from scipy.special import elliprd
from scipy.constants import c, hbar
from scipy.interpolate import interp1d
# from scipy.constants import physical_constants

from blond.utils import bmath as bm


class NagaitsevIBS():

    def __init__(self, *args, **kwargs):
        pass

    def _Phi(self, beta , alpha , eta , eta_d):
        return eta_d + alpha * eta / beta

    def set_beam_parameters(self, beam):
        self.Npart  = beam.ratio * beam.n_macroparticles_alive
        self.Ncharg = beam.Particle.charge
        self.EnTot  = beam.energy * 1e-9
        self.E_rest = beam.Particle.mass * 1e-9
        self.gammar = beam.gamma
        self.betar  = beam.beta
        self.c_rad  = beam.Particle.radius_cl

    def set_optic_functions(self, twiss):
        self.posit  = twiss['position']
        self.Circu  = twiss['position'][-1]
        self.bet_x  = twiss['betx']
        self.bet_y  = twiss['bety']
        self.alf_x  = twiss['alfx']
        self.alf_y  = twiss['alfy']
        self.eta_x  = twiss['dx']
        self.eta_dx = twiss['dpx']
        self.eta_y  = twiss['dy']
        self.eta_dy = twiss['dpy']
        self.slip   = abs(twiss['slip'])
        self.phi_x  = self._Phi(twiss['betx'], twiss['alfx'], twiss['dx'], twiss['dpx'])
        self.frev   = self.betar * c / self.Circu
        # ------------------------------------------------------------------------------
        # Interpolating and integrating the optics to compute the average values
        _bx_b = interp1d(self.posit, self.bet_x)
        _by_b = interp1d(self.posit, self.bet_y)
        _dx_b = interp1d(self.posit, self.eta_x)
        _dy_b = interp1d(self.posit, self.eta_y)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=UserWarning)
            self._bx_bar = quad(_bx_b, self.posit[0], self.posit[-1])[0] / self.Circu
            self._by_bar = quad(_by_b, self.posit[0], self.posit[-1])[0] / self.Circu
            self._dx_bar = quad(_dx_b, self.posit[0], self.posit[-1])[0] / self.Circu
            self._dy_bar = quad(_dy_b, self.posit[0], self.posit[-1])[0] / self.Circu


    def CoulogConst(self, emit_x, emit_y, sigma_delta, bunch_length):
        '''
        Calculates the Coulomb logarithm based on the beam parameters and optics, and the 
        constant in front of the IBS growth rate integrals.

        Args:
            emit_x (float): horizontal geometric emittance in [m].
            emit_y (float): vertical geometric emittance in [m].
            sigma_delta (float): Momentum spread \DeltaP/P
            bunch_length (float): 1 sigma bunch length in [m].

        Returns :
            N * r_c^2 * c * C_\mathrm{log}/ 12 * \pi \beta^3 \gamma^5 * \sigma_s
        '''
        Etrans = 5e8 * (self.gammar * self.EnTot - self.E_rest) * (emit_x / self._bx_bar)
        TempeV = 2.0 * Etrans
        sigxcm = 100 * np.sqrt(emit_x * self._bx_bar + (self._dx_bar * sigma_delta)**2)
        sigycm = 100 * np.sqrt(emit_y * self._by_bar + (self._dy_bar * sigma_delta)**2)
        sigtcm = 100 * bunch_length
        volume = 8.0 * np.sqrt(np.pi**3) * sigxcm * sigycm * sigtcm
        densty = self.Npart  / volume
        debyul = 743.4 * np.sqrt(TempeV / densty) / self.Ncharg
        rmincl = 1.44e-7 * self.Ncharg**2 / TempeV
        rminqm = hbar * c * 1e5 / (2.0 * np.sqrt(2e-3 * Etrans * self.E_rest))
        rmin   = max(rmincl, rminqm)
        rmax   = min(sigxcm, debyul)
        coulog = np.log(rmax / rmin)
        Ncon   = self.Npart * self.c_rad**2 * c / (12 * np.pi * self.betar**3 
                                                   * self.gammar**5 * bunch_length)
        return Ncon * coulog


    def growth_rates(self, emit_x, emit_y, sigma_delta, bunch_length):
        '''
        Computes the ``IBS`` growth rates, named :math:`T_x, T_y` and :math:`T_z` 
        in this code, starting from building all the nessesary parameters for the  
        Nagaitsev integrals.

        Args:
            emit_x (float): horizontal geometric emittance in [m].
            emit_y (float): vertical geometric emittance in [m].
            sigma_epsilon (float): Energy spread \DeltaE/E
            bunch_length (float): 1 sigma bunch length in [m].

        Returns:
            The ```IBS``` emittance growth rates Tx, Ty, Tz

        '''
        const = self.CoulogConst(emit_x, emit_y, sigma_delta, bunch_length)
        sigx: np.ndarray = np.sqrt(self.bet_x * emit_x + (self.eta_x * sigma_delta)**2)
        sigy: np.ndarray = np.sqrt(self.bet_y * emit_y + (self.eta_y * sigma_delta)**2)
        ax: np.ndarray = self.bet_x / emit_x
        ay: np.ndarray = self.bet_y / emit_y
        a_s: np.ndarray = ax * (self.eta_x**2 / self.bet_x**2 + self.phi_x**2) + 1 /  sigma_delta**2
        a1: np.ndarray  = (ax + self.gammar**2 * a_s) / 2.
        a2: np.ndarray  = (ax - self.gammar**2 * a_s) / 2.
        denom: np.ndarray = np.sqrt(a2**2 + self.gammar**2 * ax**2 * self.phi_x**2)
        #--------------------------------------------------------------------------------
        lambda_1: np.ndarray = ay
        lambda_2: np.ndarray = a1 + denom
        lambda_3: np.ndarray = a1 - denom
        #--------------------------------------------------------------------------------
        R1: np.ndarray = elliprd(1 / lambda_2, 1 / lambda_3, 1 / lambda_1) / lambda_1
        R2: np.ndarray = elliprd(1 / lambda_3, 1 / lambda_1, 1 / lambda_2) / lambda_2
        R3: np.ndarray = 3 * np.sqrt(lambda_1 * lambda_2 / lambda_3) - lambda_1 * R1 / lambda_3 - lambda_2 * R2 / lambda_3
        #--------------------------------------------------------------------------------
        Sp: np.ndarray  = (2 * R1 - R2 * (1 - 3 * a2 / denom) - R3 * (1 + 3 * a2 / denom) ) * 0.5 * self.gammar**2
        Sx: np.ndarray  = (2 * R1 - R2 * (1 + 3 * a2 / denom) - R3 * (1 - 3 * a2 / denom) ) * 0.5
        Sxp: np.ndarray = 3 * self.gammar**2 * self.phi_x**2 * ax * (R3 - R2) / denom
        #--------------------------------------------------------------------------------
        Ix_integrand = self.bet_x / (self.Circu * sigx * sigy) * (Sx + Sp * (self.eta_x**2 / self.bet_x**2 + self.phi_x**2) + Sxp)
        Iy_integrand = self.bet_y / (self.Circu * sigx * sigy) * (R2 + R3 - 2 * R1)
        Iz_integrand = Sp / (self.Circu * sigx * sigy)
        #--------------------------------------------------------------------------------
        self.Tx = float(np.sum(Ix_integrand[:-1] * np.diff(self.posit))) * const / emit_x
        self.Ty = float(np.sum(Iy_integrand[:-1] * np.diff(self.posit))) * const / emit_y
        self.Tz = float(np.sum(Iz_integrand[:-1] * np.diff(self.posit))) * const / sigma_delta**2

        return self.Tx, self.Ty, self.Tz
    
    # Run if you want to evaluate the emittance evolution using Nagaitsev's Integrals.
    def emittance_evolution(self, emit_x, emit_y, sigma_delta, 
                            bunch_length, dt):
        '''
        Analytically computes the new emittances after time dt, based on the IBS
        growth rates

        Args:
            emit_x (float): horizontal geometric emittance in [m]
            emit_y (float): vertical geometric emittance in [m]
            sigma_epsilon (float): Energy spread \DeltaE/E
            dt (float): the time interval to use

        Returns:
            The updated emittances after time dt
        '''
        new_emit_x = emit_x * np.exp(dt * self.Tx)
        new_emit_y = emit_y * np.exp(dt * self.Ty)
        new_bunch_length = bunch_length * np.exp(dt * self.Tz / 2)
        new_sigma_delta = sigma_delta * np.exp(dt * self.Tz / 2)

        return new_emit_x, new_emit_y, new_sigma_delta, new_bunch_length
    
    def emittance_evolution_2D(self, emit_x, emit_y, dt):
        '''
        Analytically computes the new emittances after time dt, based on the IBS
        growth rates

        Args:
            emit_x (float): horizontal geometric emittance in [m].
            emit_y (float): vertical geometric emittance in [m].
            dt (float): the time interval to use.

        Returns:
            The updated emittances after time dt
        '''
        new_emit_x = emit_x * np.exp(dt * self.Tx)
        new_emit_y = emit_y * np.exp(dt * self.Ty)

        return new_emit_x, new_emit_y


    def profile_density(self, profile, beam):
        beam_dt = beam.dt[beam.id > 0]
        bunch_length_rms = np.std(beam_dt)
        factor_distribution = 2 * np.sqrt(np.pi) * bunch_length_rms

        counts_normed = profile.n_macroparticles / np.sum(profile.n_macroparticles) / profile.bin_size

        Rho_normed = bm.interp_const_bin(beam_dt, profile.bin_centers, counts_normed * factor_distribution)
        kick_factor_normed = np.mean(Rho_normed)

        return Rho_normed


    # ! ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ !
    # ! ~~~~~~~~~~~~~~~~ Simple Kick ~~~~~~~~~~~~~~~~~~ !
    # ! ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ !
    def calculate_longitudinal_kick(self, emit_x, emit_y, beam):
        index_alive = beam.id > 0
        Sig_zeta  = np.std(beam.dt[index_alive] * c * self.betar)
        Sig_delta = np.std(beam.dE[index_alive] / (self.EnTot * 1e9) / self.betar**2)
        
        self.growth_rates(emit_x, emit_y, Sig_delta, Sig_zeta)
        if self.Tz < 0 : self.Tz = 0

        self.DSz =  np.sqrt(2 * self.Tz / self.frev) * Sig_delta * self.betar**2 * (self.EnTot * 1e9)# * self.betar**2)

        return self.DSz
    
    # Run !EVERY TURN! to apply the simple kick. Needs adjustment if it is not every turn    
    def track(self, profile, beam):
        rho = self.profile_density(profile, beam)

        Dkick_p = bm.random_normal(loc = 0, scale = self.DSz,
                                   size = beam.dE[beam.id > 0].shape[0]) * np.sqrt(rho)
 
        beam.dE[beam.id > 0] += Dkick_p