# env.py
import numpy as np
from dynamics import MicroRobotDynamics, MicroRobotParams

class MicroRobotEnv:
    def __init__(self, dt=0.02, params=None, renderer=None):
        self.dt = dt
        self.params = params if params is not None else MicroRobotParams()
        self.dyn = MicroRobotDynamics(self.params)

        # renderer is optional (BulletRenderer)
        self.renderer = renderer

        self.state = np.zeros(3, dtype=float)
        self.phi_spin = 0.0
        self.last_omega_eff = 0.0

    def reset(self, state0=None):
        self.state[:] = 0.0 if state0 is None else np.array(state0, dtype=float)
        self.phi_spin = 0.0
        self.last_omega_eff = 0.0
        return self.state.copy()

    def step(self, action):
        next_state, info = self.dyn.step(self.state, action, self.dt)

        # update internal
        self.state = next_state

        # update spin phase for visualization only
        self.last_omega_eff = float(info.get("omega_eff", 0.0))
        self.phi_spin += self.last_omega_eff * self.dt

        # optional render
        if self.renderer is not None:
            self.renderer.update(
                state_xyz=self.state,
                k_hat=info.get("k_hat", [0,0,1]),
                phi_spin=self.phi_spin
            )

        return self.state.copy(), info

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
