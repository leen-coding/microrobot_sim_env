# dynamics.py
import numpy as np
from dataclasses import dataclass, field
from typing import Callable, Optional, Dict, Any
from math_utils import _normalize


@dataclass
class MagneticParams:
    B0_mT: float = 8.0
    m_mag: float = 4.08e-3
    m_scale: float = 0.1  # scale for effective magnetic moment


@dataclass
class HelixParams:
    n_turns: float = 2.0
    R_helix: float = 1e-3
    theta_rad: float = np.deg2rad(60.9)
    lam: float = 3.5e-3
    r_fil: float = 3e-4
    d_head: float = 2.5e-3
    mass_kg: float = 0.043e-3
    volume_m3: float = 13.378e-9


@dataclass
class EnvParams:
    eta: float = 0.34
    fluid_density_kg_m3: float = 970.0
    gravity: float = 9.80665
    apply_gravity: bool = True


@dataclass
class NoiseParams:
    drift: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=float))
    process_noise_std: float = 0.0  # velocity noise std (m/s)


@dataclass
class ExternalForceParams:
    # force_fn signature: force_fn(state, k_hat, omega_eff, **kwargs) -> array-like or None
    force_fn: Optional[Callable[..., np.ndarray]] = None
    kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MicroRobotParams:
    mag: MagneticParams = field(default_factory=MagneticParams)
    helix: HelixParams = field(default_factory=HelixParams)
    env: EnvParams = field(default_factory=EnvParams)
    noise: NoiseParams = field(default_factory=NoiseParams)
    ext: ExternalForceParams = field(default_factory=ExternalForceParams)


class MicroRobotDynamics:
    """
    Pure dynamics: state update only. No rendering, no external dependencies.
    state: [x,y,z] in meters
    action: [kx,ky,kz,f_hz]
    """

    def __init__(self, params: MicroRobotParams, rng: Optional[np.random.Generator] = None):
        self.p = params
        self.rng = rng

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
                "k_hat": np.array([0.0, 0.0, 1.0]),
                "v_scalar": 0.0,
                "omega_cmd": 0.0,
                "omega_eff": 0.0,
                "f_step": 0.0,
                "beta": 0.0,
                "gamma": 0.0,
                "tau_max": self._tau_max(),
                "sync": True,
            }

        # (1) propulsion model
        beta, gamma, omega_eff, f_step, sync = self._propulsion(f_hz)

        # (2) base swimming velocity along k_hat
        v_scalar = beta * omega_eff
        v_vec = v_scalar * k_hat + self.p.noise.drift

        # (3) gravity + buoyancy
        v_vec, gb_info = self._apply_gravity_buoyancy(v_vec, dt)

        # (4) external force
        v_vec, ext_info = self._apply_external_force(state, k_hat, omega_eff, v_vec, dt)

        # (5) process noise
        v_vec = self._apply_process_noise(v_vec, dt)

        info = {
            "k_hat": k_hat.copy(),
            "v_scalar": float(v_scalar),
            "v_vec": v_vec.copy().tolist(),
            "omega_cmd": float(2.0 * np.pi * f_hz),
            "omega_eff": float(omega_eff),
            "f_step": float(f_step) if np.isfinite(f_step) else f_step,
            "beta": float(beta),
            "gamma": float(gamma),
            "tau_max": float(self._tau_max()),
            "sync": bool(sync),
            **gb_info,
            **ext_info,
        }
        return v_vec, info

    def _tau_max(self) -> float:
        B0_T = self.p.mag.B0_mT * 1e-3
        return self.p.mag.m_mag * B0_T * self.p.mag.m_scale

    def _propulsion(self, f_hz):
        a, b, c, _, _ = self._abc_coeffs()

        # head drags
        helix = self.p.helix
        env = self.p.env
        psi_v = 3.0 * np.pi * env.eta * helix.d_head
        psi_omega = np.pi * env.eta * (helix.d_head ** 3)

        denom = a + psi_v
        if abs(denom) < 1e-18:
            denom = 1e-18

        beta = -b / denom
        gamma = (c + psi_omega) - (b ** 2) / denom

        omega_cmd = 2.0 * np.pi * f_hz
        tau_max = self._tau_max()

        if gamma <= 0:
            omega_eff = omega_cmd
            f_step = np.inf
            sync = True
        else:
            omega_step = tau_max / gamma
            omega_eff = min(omega_cmd, omega_step)
            f_step = omega_step / (2.0 * np.pi)
            sync = (omega_cmd <= omega_step)

        return beta, gamma, omega_eff, f_step, sync

    def _apply_gravity_buoyancy(self, v_vec, dt):
        env = self.p.env
        if not env.apply_gravity:
            return v_vec, {"a_gravity": 0.0, "a_buoyancy": 0.0, "a_gb": 0.0}

        mass = float(self.p.helix.mass_kg)
        if mass <= 1e-12:
            mass = 1e-12

        a_buoyancy = (env.fluid_density_kg_m3 * self.p.helix.volume_m3 * env.gravity) / mass
        a_gb = -env.gravity + a_buoyancy
        v_vec = v_vec + np.array([0.0, 0.0, a_gb * dt], dtype=float)

        return v_vec, {
            "a_gravity": float(-env.gravity),
            "a_buoyancy": float(a_buoyancy),
            "a_gb": float(a_gb),
        }

    def _apply_external_force(self, state, k_hat, omega_eff, v_vec, dt):
        ext = self.p.ext
        if ext.force_fn is None:
            return v_vec, {"F_ext": None, "a_ext": [0.0, 0.0, 0.0], "v_ext": [0.0, 0.0, 0.0]}

        F_ext = ext.force_fn(state=state, k_hat=k_hat, omega_eff=omega_eff, **ext.kwargs)
        if F_ext is None:
            return v_vec, {"F_ext": None, "a_ext": [0.0, 0.0, 0.0], "v_ext": [0.0, 0.0, 0.0]}

        F_ext = np.asarray(F_ext, dtype=float)
        mass = float(self.p.helix.mass_kg)
        if mass <= 1e-12:
            mass = 1e-12
        a_ext = F_ext / mass
        v_ext = a_ext * dt
        v_vec = v_vec + v_ext

        return v_vec, {
            "F_ext": F_ext.tolist(),
            "a_ext": a_ext.tolist(),
            "v_ext": v_ext.tolist(),
        }

    def _apply_process_noise(self, v_vec, dt):
        sigma = float(self.p.noise.process_noise_std)
        if sigma <= 0:
            return v_vec

        scale = sigma * np.sqrt(dt)
        if self.rng is None:
            return v_vec + np.random.randn(3) * scale
        return v_vec + self.rng.normal(0.0, scale, size=3)

    def _abc_coeffs(self):
        helix = self.p.helix
        env = self.p.env
        s = np.sin(helix.theta_rad)
        if s < 1e-12:
            s = 1e-12

        arg = 0.36 * helix.lam / (helix.r_fil * s)
        arg = max(arg, 1.001)

        ln_term = np.log(arg)
        xi_perp = (4.0 * np.pi * env.eta) / (ln_term + 0.5)
        xi_para = (2.0 * np.pi * env.eta) / (ln_term)

        cth = np.cos(helix.theta_rad)

        parameter_a = (xi_para * (cth ** 2) + xi_perp * (s ** 2)) / s
        parameter_b = (xi_para - xi_perp) * cth
        parameter_c = (xi_perp * (cth ** 2) + xi_para * (s ** 2)) / s

        a = 2.0 * np.pi * helix.n_turns * helix.R_helix * parameter_a
        b = 2.0 * np.pi * helix.n_turns * (helix.R_helix ** 2) * parameter_b
        c = 2.0 * np.pi * helix.n_turns * (helix.R_helix ** 3) * parameter_c

        return a, b, c, xi_perp, xi_para
