# env.py
import copy
import numpy as np
from dynamics import MicroRobotDynamics, MicroRobotParams
from randomization import apply_episode_randomization
from recorder import TrajectoryRecorder

class MicroRobotEnv:
    def __init__(
        self,
        dt=0.02,
        params=None,
        renderer=None,
        ground_z: float = 0,
        clamp_to_ground: bool = True,
        episode_randomization=None,
        rng=None,
        recorder: TrajectoryRecorder = None,
    ):
        self.dt = float(dt)

        # keep a clean baseline, avoid cross-episode parameter drift
        base = params if params is not None else MicroRobotParams()
        self.base_params = copy.deepcopy(base)
        self.params = copy.deepcopy(self.base_params)

        self.rng = rng if rng is not None else np.random.default_rng()
        self.episode_randomization = episode_randomization

        self.dyn = MicroRobotDynamics(self.params, rng=self.rng)

        self.renderer = renderer
        self.ground_z = float(ground_z)
        self.clamp_to_ground = bool(clamp_to_ground)

        self.state = np.zeros(3, dtype=float)
        self.phi_spin = 0.0
        self.last_k_hat = np.array([0.0, 0.0, 1.0], dtype=float)

        self.recorder = recorder if recorder is not None else TrajectoryRecorder(enabled=False)

    def seed(self, seed: int):
        self.rng = np.random.default_rng(int(seed))
        self.dyn.rng = self.rng
        return seed

    def reset(self, state0=None):
        # restore baseline params then randomize for this episode
        self.params = copy.deepcopy(self.base_params)
        self.dyn.p = self.params

        sampled = apply_episode_randomization(self.params, self.rng, self.episode_randomization)

        # init state
        if state0 is None:
            z0 = self.ground_z
            self.state[:] = (0.0, 0.0, z0)
        else:
            s = np.array(state0, dtype=float).reshape(-1)
            self.state[:] = s

        self.phi_spin = 0.0
        self.last_k_hat[:] = (0.0, 0.0, 1.0)

        self.recorder.on_reset(self.state, params=self.params, randomization=sampled)
        return self.state.copy()

    def step(self, action):
        next_state, info = self.dyn.step(self.state, action, self.dt)

        ground_contact = False
        if self.clamp_to_ground and next_state[2] < self.ground_z:
            next_state[2] = self.ground_z
            ground_contact = True

        self.state = next_state

        # use dynamics' k_hat for consistency
        k_hat = info.get("k_hat", np.array([0.0, 0.0, 1.0], dtype=float))
        self.last_k_hat = np.asarray(k_hat, dtype=float)

        omega_eff = float(info.get("omega_eff", 0.0))
        self.phi_spin += omega_eff * self.dt

        info = dict(info)
        info["ground_contact"] = ground_contact

        self.recorder.on_step(self.state, action, info)
        return self.state.copy(), info

    def render(self):
        if self.renderer is None:
            return
        self.renderer.update(
            state_xyz=self.state,
            k_hat=self.last_k_hat,
            phi_spin=self.phi_spin,
        )

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None

    # convenience passthroughs
    def enable_logging(self, enabled=True, clear=True):
        self.recorder.enabled = bool(enabled)
        if clear and self.recorder.enabled:
            self.recorder.reset()

    def get_trajectory(self):
        return self.recorder.get_trajectory()

    def get_episode_meta(self):
        return self.recorder.get_episode_meta()
