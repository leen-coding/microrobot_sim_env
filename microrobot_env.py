"""
This script is to create a simulation env for microrobot in 2D world.

Action definition (3D):
    action[0] = dir_x   (desired swimming direction in world x-y plane)
    action[1] = dir_y
    action[2] = f_hz    (rotating magnetic field frequency, Hz)

State definition (2D):
    state = [x, y]      (position in meters)

Dynamics:
    Use the helical-propeller propulsion matrix in the provided paper.

    [f]   [a  b][v]
    [τ] = [b  c][ω]                                  (Eq. 7)

    With head drag (paper Eq. 13):
    [f]   [a+ψ_v     b ][v]
    [τ] = [ b     c+ψ_ω][ω]

    In the magnetic-torque swimming case, it is convenient to use (paper Eq. 15):
    [v]   [ α   β ][f]
    [τ] = [-β   γ ][ω]

    If we assume no external non-fluidic force along the axis (f = 0):
        v = β ω
        τ = γ ω

    Synchronous regime: ω ≈ 2π f_hz, until step-out where τ required exceeds τ_max.
"""

import numpy as np


class MicroRobotEnv:
    def __init__(self):
        # --- simulation ---
        self.dt = 0.02  # seconds (50 Hz integration)
        self.state = np.zeros(2, dtype=float)  # [x, y] in meters

        # Action: [dir_x, dir_y, f_hz]
        self.action_space = np.zeros(3, dtype=float)

        # --- magnetic field ---
        self.B0_mT = 5.0
        self.B0 = self.B0_mT * 1e-3  # Tesla

        # Magnetic dipole moment magnitude (A·m^2).
        # N52 r = 0.75, h = 2mm
        self.m_mag = 4.08 * 1e-3  # A·m^2 (placeholder; tune/identify from experiments)

        # --- fluid + geometry (SI units) ---
        self.eta = 0.34        # Pa·s (water at room temp)
        self.n_turns = 2       # number of helix turns, n
        self.R_helix = 1 * 1e-3    # helix radius (meters)  (paper uses n0 or similar)
        self.theta = np.deg2rad(60.9)  # pitch angle θ (radians)
        self.lam = 3.5 * 1e-3      # helix pitch / wavelength λ (meters)
        self.r_fil = 3 * 1e-4      # filament radius r (meters)

        # Head (spherical) diameter d (meters), for ψ_v and ψ_ω in the paper
        self.d_head = 2.5 * 1e-3

        # Optional simple drift/noise (set to 0 for deterministic)
        self.drift = np.zeros(2)        # m/s
        self.process_noise_std = 0.0    # m/sqrt(s)

        self.reset()

    def reset(self):
        self.state[:] = 0.0
        return self.state.copy()

    def step(self, action):
        v_vec, info = self.robot_kinematics(action)
        self.state = self.state + v_vec * self.dt
        return self.state.copy(), info

    def robot_kinematics(self, action):
        """
        Compute 2D translational velocity from action = [dir_x, dir_y, f_hz],
        based on the helical propeller dynamics in the provided paper.

        Returns:
            v_vec: np.ndarray shape (2,), m/s
            info: dict with intermediate values (v_scalar, omega_eff, f_step, etc.)
        """
        dir_x, dir_y, f_hz = float(action[0]), float(action[1]), float(action[2])

        # Normalize direction (if zero, no motion)
        d = np.array([dir_x, dir_y], dtype=float)
        dn = np.linalg.norm(d)
        if dn < 1e-12 or f_hz <= 0.0:
            return np.zeros(2), {
                "v_scalar": 0.0, "omega_cmd": 0.0, "omega_eff": 0.0,
                "f_step": 0.0, "beta": 0.0, "gamma": 0.0
            }
        d /= dn

        # --- build propulsion coefficients a, b, c (paper Eq. 8–10) ---
        a, b, c, xi_perp, xi_para = self._abc_coeffs()

        # --- head drag terms ψ_v and ψ_ω (paper Eq. 13–14; ψ_v referenced as (5)) ---
        # For a sphere of diameter d: translation drag = 3π η d ; rotation drag = π η d^3
        psi_v = 3.0 * np.pi * self.eta * self.d_head
        psi_omega = np.pi * self.eta * (self.d_head ** 3)

        # --- convert to (α, β, γ) form (paper Eq. 16) ---
        denom = (a + psi_v)
        # Guard against degenerate parameters
        if abs(denom) < 1e-18:
            denom = np.sign(denom) * 1e-18 if denom != 0 else 1e-18

        beta = -b / denom
        gamma = (c + psi_omega) - (b ** 2) / denom  # paper Eq. 16

        # --- commanded angular speed from magnetic field rotation ---
        omega_cmd = 2.0 * np.pi * f_hz  # rad/s

        # --- step-out limit: τ_required = gamma * omega, τ_max ≈ m * B0 ---
        tau_max = self.m_mag * self.B0  # N·m (A·m^2 * T = N·m)
        # Guard against non-physical gamma
        if gamma <= 0:
            # If gamma becomes non-positive due to bad parameters, fall back to no step-out limit
            omega_eff = omega_cmd
            omega_step = np.inf
            f_step = np.inf
        else:
            omega_step = tau_max / gamma
            omega_eff = min(omega_cmd, omega_step)
            f_step = omega_step / (2.0 * np.pi)

        # --- forward speed (f = 0 case): v = beta * omega ---
        v_scalar = beta * omega_eff

        # --- add drift and process noise ---
        v_vec = v_scalar * d + self.drift
        if self.process_noise_std > 0:
            v_vec = v_vec + np.random.randn(2) * (self.process_noise_std / np.sqrt(self.dt))

        info = {
            "v_scalar": float(v_scalar),
            "omega_cmd": float(omega_cmd),
            "omega_eff": float(omega_eff),
            "omega_step": float(omega_step) if np.isfinite(omega_step) else omega_step,
            "f_step": float(f_step) if np.isfinite(f_step) else f_step,
            "beta": float(beta),
            "gamma": float(gamma),
            "a": float(a), "b": float(b), "c": float(c),
            "psi_v": float(psi_v), "psi_omega": float(psi_omega),
            "xi_perp": float(xi_perp), "xi_para": float(xi_para),
            "tau_max": float(tau_max),
        }
        return v_vec, info

    def _abc_coeffs(self):
        """
        Compute propulsion matrix coefficients a, b, c (paper Eq. 8–10),
        using Lighthill-type RFT viscous coefficients for a thin filament element.

        Returns:
            a, b, c, xi_perp, xi_para
        """
        # Viscous coefficients (paper excerpt; highlighted equations)
        # xi_perp = 4π η / ( ln(0.36 λ/(r sinθ)) + 0.5 )
        # xi_para = 2π η / ( ln(0.36 λ/(r sinθ)) )
        s = np.sin(self.theta)
        if s < 1e-12:
            s = 1e-12

        arg = 0.36 * self.lam / (self.r_fil * s)
        # Guard: arg must be > 1 for ln(arg) positive; if not, still compute but clamp
        arg = max(arg, 1.001)

        ln_term = np.log(arg)
        xi_perp = (4.0 * np.pi * self.eta) / (ln_term + 0.5)
        xi_para = (2.0 * np.pi * self.eta) / (ln_term)

        cth = np.cos(self.theta)

        # a,b,c (paper Eq. 8–10) with helix radius R_helix and turns n_turns
        # Note: paper uses "n" (#turns) and "n0" (helix radius). Here: n_turns, R_helix.
        parameter_a = (xi_para * (cth ** 2) + xi_perp * (s ** 2)) / s
        parameter_b = (xi_para - xi_perp) * cth
        parameter_c = (xi_perp * (cth ** 2) + xi_para * (s ** 2)) / s
        a = 2.0 * np.pi * self.n_turns * self.R_helix * parameter_a
        b = 2.0 * np.pi * self.n_turns * (self.R_helix ** 2) * parameter_b
        c = 2.0 * np.pi * self.n_turns * (self.R_helix ** 3) * parameter_c

        return a, b, c, xi_perp, xi_para

    def render(self):
        # Minimal placeholder; integrate with matplotlib if needed
        print(f"state(x,y) = {self.state}")


import matplotlib.pyplot as plt

def test_run():
    env = MicroRobotEnv()
    traj = []
    
    # 模拟一个圆周运动：改变方向，保持频率
    f_test = 5.0  
    for t in range(200):
        angle = 0.05 * t
        action = [np.cos(angle), np.sin(angle), f_test]
        state, info = env.step(action)
        traj.append(state.copy())
    
    traj = np.array(traj)
    
    # 绘图
    plt.figure(figsize=(8, 6))
    plt.plot(traj[:, 0] * 1000, traj[:, 1] * 1000, 'b-o', label='Trajectory')
    plt.xlabel('X (mm)')
    plt.ylabel('Y (mm)')
    plt.title(f'Microrobot Simulation (f={f_test}Hz, Step-out={info["f_step"]:.1f}Hz)')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    test_run()