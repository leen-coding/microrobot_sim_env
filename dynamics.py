# dynamics.py
import numpy as np
from dataclasses import dataclass
from utils import _normalize

@dataclass
class MicroRobotParams:
    # magnetic
    B0_mT: float = 5.0
    m_mag: float = 4.08e-3
    m_mag_coff: float = 0.1

    # fluid + geometry
    eta: float = 0.34
    n_turns: float = 2.0
    R_helix: float = 1e-3
    theta_rad: float = np.deg2rad(60.9)
    lam: float = 3.5e-3
    r_fil: float = 3e-4
    d_head: float = 2.5e-3

    # optional drift / noise
    drift: np.ndarray = None
    process_noise_std: float = 0.0

class MicroRobotDynamics:
    """
    Pure dynamics: state update only. No rendering, no external dependencies.
    state: [x,y,z] in meters
    action: [kx,ky,kz,f_hz]
    """
    def __init__(self, params: MicroRobotParams):
        self.p = params
        if self.p.drift is None:
            self.p.drift = np.zeros(3, dtype=float)

        self.B0 = self.p.B0_mT * 1e-3  # Tesla

    def step(self, state, action, dt):
        v_vec, info = self.velocity(state, action, dt)
        next_state = state + v_vec * dt
        return next_state, info

    def velocity(self, state, action, dt):
        k = np.array(action[:3], dtype=float)
        f_hz = float(action[3])

        k_hat, kn = _normalize(k)
        if kn < 1e-12 or f_hz <= 0.0:
            return np.zeros(3), {
                "k_hat": np.array([0.0,0.0,1.0]),
                "v_scalar": 0.0,
                "omega_cmd": 0.0,
                "omega_eff": 0.0,
                "f_step": 0.0,
                "beta": 0.0,
                "gamma": 0.0,
                "tau_max": self.p.m_mag * self.B0 * self.p.m_mag_coff,
            }

        a, b, c, xi_perp, xi_para = self._abc_coeffs()

        # head drags
        psi_v = 3.0 * np.pi * self.p.eta * self.p.d_head
        psi_omega = np.pi * self.p.eta * (self.p.d_head ** 3)

        denom = a + psi_v
        if abs(denom) < 1e-18:
            denom = 1e-18

        beta = -b / denom
        gamma = (c + psi_omega) - (b ** 2) / denom

        omega_cmd = 2.0 * np.pi * f_hz
        tau_max = self.p.m_mag * self.B0 * self.p.m_mag_coff

        if gamma <= 0:
            omega_eff = omega_cmd
            f_step = np.inf
            sync = True
        else:
            omega_step = tau_max / gamma
            omega_eff = min(omega_cmd, omega_step)
            f_step = omega_step / (2.0 * np.pi)
            sync = (omega_cmd <= omega_step)

        v_scalar = beta * omega_eff
        v_vec = v_scalar * k_hat + self.p.drift


        # process noise (optional)
        if self.p.process_noise_std > 0:
            v_vec = v_vec + np.random.randn(3) * (self.p.process_noise_std / np.sqrt(dt))

        info = {
            "k_hat": k_hat.copy(),
            "v_scalar": float(v_scalar),
            "omega_cmd": float(omega_cmd),
            "omega_eff": float(omega_eff),
            "f_step": float(f_step) if np.isfinite(f_step) else f_step,
            "beta": float(beta),
            "gamma": float(gamma),
            "tau_max": float(tau_max),
            "sync": bool(sync),
        }
        return v_vec, info

    def _abc_coeffs(self):
        s = np.sin(self.p.theta_rad)
        if s < 1e-12:
            s = 1e-12

        arg = 0.36 * self.p.lam / (self.p.r_fil * s)
        arg = max(arg, 1.001)

        ln_term = np.log(arg)
        xi_perp = (4.0 * np.pi * self.p.eta) / (ln_term + 0.5)
        xi_para = (2.0 * np.pi * self.p.eta) / (ln_term)

        cth = np.cos(self.p.theta_rad)

        parameter_a = (xi_para * (cth ** 2) + xi_perp * (s ** 2)) / s
        parameter_b = (xi_para - xi_perp) * cth
        parameter_c = (xi_perp * (cth ** 2) + xi_para * (s ** 2)) / s

        a = 2.0 * np.pi * self.p.n_turns * self.p.R_helix * parameter_a
        b = 2.0 * np.pi * self.p.n_turns * (self.p.R_helix ** 2) * parameter_b
        c = 2.0 * np.pi * self.p.n_turns * (self.p.R_helix ** 3) * parameter_c

        return a, b, c, xi_perp, xi_para
