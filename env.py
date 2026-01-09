# env.py
import numpy as np
from dynamics import MicroRobotDynamics, MicroRobotParams

class MicroRobotEnv:
    def __init__(self, dt=0.02, params=None, renderer=None,
                 ground_z: float = 0.0, init_z_raise: float = 5e-3,
                 clamp_to_ground: bool = True):
        self.dt = dt
        self.params = params if params is not None else MicroRobotParams()
        self.dyn = MicroRobotDynamics(self.params)

        # renderer is optional (BulletRenderer)
        self.renderer = renderer

        # ground handling
        self.ground_z = float(ground_z)
        # default small raise above ground to avoid immediate contact
        self.init_z_raise = float(init_z_raise)
        self.clamp_to_ground = bool(clamp_to_ground)

        self.state = np.zeros(3, dtype=float)
        self.phi_spin = 0.0
        self.last_omega_eff = 0.0

    def reset(self, state0=None):
        # default start slightly above ground to avoid falling through
        if state0 is None:
            z0 = self.ground_z + self.init_z_raise
            self.state[:] = np.array([0.0, 0.0, z0], dtype=float)
        else:
            s = np.array(state0, dtype=float)
            # ensure initial z is at least ground_z + init_z_raise
            if s.shape[0] >= 3:
                if s[2] < self.ground_z + self.init_z_raise:
                    s[2] = self.ground_z + self.init_z_raise
            else:
                # pad to 3
                s = np.array([s[0], s[1] if s.shape[0] > 1 else 0.0, self.ground_z + self.init_z_raise], dtype=float)
            self.state[:] = s
        self.phi_spin = 0.0
        self.last_omega_eff = 0.0
        return self.state.copy()

    def step(self, action):
        """Advance environment by one step.

        Returns (next_state, info). If clamp_to_ground is True the z
        coordinate will not go below `ground_z` and `info["ground_contact"]`
        will be set when contact occurs.
        """
        # If renderer supports physics, apply the velocity to the dynamic body
        # and read back the base position after stepping the simulation. Otherwise
        # use the pure dynamics integrator and teleport the rendered body.
        ground_contact = False
        if self.renderer is not None and getattr(self.renderer, "use_physics", False):
            # compute velocity (do not advance state here)
            v_vec, info = self.dyn.velocity(self.state, action, self.dt)
            # apply as base linear velocity
            # convert to meters/sec (v_vec already in m/s)
            try:
                self.renderer.apply_velocity(v_vec.tolist())
            except Exception:
                # fallback: ignore physics velocity application
                pass

            # call renderer.update to step simulation (it steps internally)
            self.renderer.update(state_xyz=self.state, k_hat=info.get("k_hat", [0,0,1]), phi_spin=self.phi_spin)

            # read back actual base position from pybullet and use as state
            pos, _ = self.renderer.get_base_position()
            next_state = np.array(pos, dtype=float)

            # clamp to ground if enabled
            if self.clamp_to_ground and next_state.shape[0] >= 3 and next_state[2] < self.ground_z:
                next_state[2] = self.ground_z
                ground_contact = True
        else:
            next_state, info = self.dyn.step(self.state, action, self.dt)

            # optionally clamp to ground so robot does not fall below floor
            if self.clamp_to_ground and next_state.shape[0] >= 3:
                if next_state[2] < self.ground_z:
                    next_state[2] = self.ground_z
                    ground_contact = True

            # commit state
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

        # expose ground contact in info
        info = dict(info)
        info["ground_contact"] = ground_contact

        return self.state.copy(), info

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
