import numpy as np
import gymnasium as gym
from gymnasium import spaces

from env import MicroRobotEnv
from dynamics import MicroRobotParams


class MicroRobotGymEnv(gym.Env):
    """Minimal Gym wrapper around MicroRobotEnv with a circle-tracking task.

    Observations: [x,y,z, ex,ey,ez] as float32 array.
    Actions: [kx,ky,kz,f_hz] as continuous Box.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        dt=0.02,
        params: MicroRobotParams = None,
        renderer=None,
        render_mode=None,
        circle_radius=0.05,
        circle_center=(0.0, 0.0),
        circle_z=0.0,
        circle_period=10.0,
        max_steps=500,
        k_max=1.0,
        f_max=10.0,
    ):
        super().__init__()
        self.env = MicroRobotEnv(dt=dt, params=params, renderer=renderer)
        self.dt = float(dt)
        self.circle_radius = float(circle_radius)
        self.circle_center = np.asarray(circle_center, dtype=float).reshape(2)
        self.circle_z = float(circle_z)
        self.circle_period = float(circle_period)
        self.max_steps = int(max_steps)
        self.k_max = float(k_max)
        self.f_max = float(f_max)
        self.t = 0.0
        self.step_count = 0
        # action: two directional components and frequency
        self.action_space = spaces.Box(low=np.array([-1.0, -1.0, -1.0, 0.0], dtype=np.float32),
                                       high=np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
                                       dtype=np.float32)

        # observation: position + tracking error
        obs_high = np.array([np.finfo(np.float32).max] * 6, dtype=np.float32)
        self.observation_space = spaces.Box(low=-obs_high, high=obs_high, dtype=np.float32)

        self.render_mode = render_mode

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
        self.t = 0.0
        self.step_count = 0
        state = self.env.reset()
        obs = self._build_obs(state)
        info = {}
        return obs.astype(np.float32), info

    def step(self, action):
        action = np.asarray(action, dtype=float)
        action = self._scale_action(action)
        state, info = self.env.step(action)
        self.t += self.dt
        self.step_count += 1

        ref = self._reference(self.t)
        err = state - ref
        obs = self._build_obs(state)

        # reward: track reference position
        reward = -float(np.linalg.norm(err))
        terminated = False
        truncated = self.step_count >= self.max_steps

        info = dict(info)
        info["ref"] = ref.tolist()
        info["err"] = err.tolist()
        return obs.astype(np.float32), reward, terminated, truncated, info

    def render(self, mode="human"):
        # call env.render which is decoupled from dynamics
        self.env.render()

    def close(self):
        self.env.close()

    def _reference(self, t):
        omega = (2.0 * np.pi) / max(self.circle_period, 1e-6)
        x = self.circle_center[0] + self.circle_radius * np.cos(omega * t)
        y = self.circle_center[1] + self.circle_radius * np.sin(omega * t)
        z = self.circle_z
        return np.array([x, y, z], dtype=float)

    def _build_obs(self, state):
        ref = self._reference(self.t)
        err = state - ref
        return np.concatenate([state, err], axis=0)

    def _scale_action(self, action):
        k = np.clip(action[:3], -1.0, 1.0) * self.k_max
        f = np.clip(action[3], 0.0, 1.0) * self.f_max
        return np.array([k[0], k[1], k[2], f], dtype=float)
