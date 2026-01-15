import numpy as np
import gymnasium as gym
from gymnasium import spaces

from env import MicroRobotEnv
from dynamics import MicroRobotParams


class MicroRobotGymEnv(gym.Env):
    """Minimal Gym wrapper around MicroRobotEnv with a circle-tracking task.

    Observations: [x,y,z, ex,ey,ez] as float32 array.
    Actions: delta control [dkx,dky,dkz,df] in [-1,1].
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
        circle_period=5.0,
        max_steps=500,
        k_rate_rad_s=np.deg2rad(30.0),
        f_rate_hz_s=5.0,
        f_min=1.0,
        f_max=10.0,
        w_rad=2.0,
        w_pos=0.0,
        w_near=1.0,
        w_tan=0.5,
        w_smooth=0.05,
    ):
        super().__init__()
        self.env = MicroRobotEnv(dt=dt, params=params, renderer=renderer)
        self.dt = float(dt)
        self.circle_radius = float(circle_radius)
        self.circle_center = np.asarray(circle_center, dtype=float).reshape(2)
        self.circle_z = float(circle_z)
        self.circle_period = float(circle_period)
        self.max_steps = int(max_steps)
        self.k_rate_rad_s = float(k_rate_rad_s)
        self.f_rate_hz_s = float(f_rate_hz_s)
        self.f_min = float(f_min)
        self.f_max = float(f_max)
        self.w_rad = float(w_rad)
        self.w_pos = float(w_pos)
        self.w_near = float(w_near)
        self.w_tan = float(w_tan)
        self.w_smooth = float(w_smooth)
        self.t = 0.0
        self.step_count = 0
        self.k_cmd = np.array([1.0, 0.0, 0.0], dtype=float)
        self.f_cmd = 0.0
        # action: two directional components and frequency
        self.action_space = spaces.Box(low=np.array([-1.0, -1.0, -1.0, -1.0], dtype=np.float32),
                                       high=np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
                                       dtype=np.float32)

        # observation: position + tracking error
        obs_high = np.array([np.finfo(np.float32).max] * 6, dtype=np.float32)
        self.observation_space = spaces.Box(low=-obs_high, high=obs_high, dtype=np.float32)

        self.render_mode = render_mode

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.t = 0.0
        self.step_count = 0
        self.k_cmd[:] = (1.0, 0.0, 0.0)
        self.f_cmd = 0.0
        state = self.env.reset()
        obs = self._build_obs(state)
        info = {}
        return obs.astype(np.float32), info

    def step(self, action):
        action = np.asarray(action, dtype=float)
        d_k = np.clip(action[:3], -1.0, 1.0)
        d_f = float(np.clip(action[3], -1.0, 1.0))

        self.k_cmd = self._update_k(self.k_cmd, d_k, self.k_rate_rad_s, self.dt)
        self.f_cmd = np.clip(self.f_cmd + self.f_rate_hz_s * d_f * self.dt, self.f_min, self.f_max)

        ctrl = np.array([self.k_cmd[0], self.k_cmd[1], self.k_cmd[2], self.f_cmd], dtype=float)
        state, info = self.env.step(ctrl)
        self.t += self.dt
        self.step_count += 1

        ref = self._reference(self.t)
        err = state - ref
        obs = self._build_obs(state)

        v_vec = np.asarray(info.get("v_vec", [0.0, 0.0, 0.0]), dtype=float)
        reward, shaped = self._reward(state, ref, v_vec, d_k, d_f)
        terminated = False
        truncated = self.step_count >= self.max_steps

        info = dict(info)
        info["ref"] = ref.tolist()
        info["err"] = err.tolist()
        info.update(shaped)
        info["k_cmd"] = self.k_cmd.tolist()
        info["f_cmd"] = float(self.f_cmd)
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

    def _update_k(self, k_cmd, d_k, k_rate, dt):
        # d_k 表示希望朝哪个方向转（policy 输出）
        dk_norm = np.linalg.norm(d_k)
        if dk_norm < 1e-9:
            return k_cmd
        target = d_k / dk_norm  # 目标方向（单位向量）

        # 当前 k_cmd -> target 的夹角
        k0 = k_cmd / max(np.linalg.norm(k_cmd), 1e-9)
        dot = float(np.clip(np.dot(k0, target), -1.0, 1.0))
        ang = float(np.arccos(dot))

        # 每步最多转 Δθ
        dtheta = float(k_rate * dt)
        if ang < 1e-9:
            return k0
        if ang <= dtheta:
            return target

        # Rodrigues 旋转：绕 axis = k0 × target 旋转 dtheta
        axis = np.cross(k0, target)
        axis_n = np.linalg.norm(axis)
        if axis_n < 1e-9:
            # 反向或数值问题：随便选一条正交轴
            axis = np.cross(k0, np.array([1.0, 0.0, 0.0]))
            axis_n = np.linalg.norm(axis)
            if axis_n < 1e-9:
                axis = np.cross(k0, np.array([0.0, 1.0, 0.0]))
                axis_n = max(np.linalg.norm(axis), 1e-9)
        axis = axis / axis_n

        # k_new = k0*cos + (axis×k0)*sin + axis*(axis·k0)*(1-cos)
        c = np.cos(dtheta)
        s = np.sin(dtheta)
        k_new = k0 * c + np.cross(axis, k0) * s + axis * (np.dot(axis, k0)) * (1 - c)
        return k_new / max(np.linalg.norm(k_new), 1e-9)


    def _reward(self, state, ref, v_vec, d_k, d_f):
        p = np.asarray(state, dtype=float)
        c_xy = self.circle_center
        p_xy = p[:2]
        r = float(np.linalg.norm(p_xy - c_xy))
        e_rad = abs(r - self.circle_radius)
        e_pos = float(np.linalg.norm(p - ref))
        delta = p_xy - c_xy
        delta_norm = np.linalg.norm(delta)
        if delta_norm > 1e-9:
            p_near = c_xy + (self.circle_radius * delta / delta_norm)
        else:
            p_near = c_xy + np.array([self.circle_radius, 0.0], dtype=float)
        e_near = float(np.linalg.norm(p_xy - p_near))

        omega = (2.0 * np.pi) / max(self.circle_period, 1e-6)
        t_hat = np.array([-np.sin(omega * self.t), np.cos(omega * self.t), 0.0], dtype=float)
        v_tan = float(np.dot(v_vec, t_hat))

        smooth = float(np.dot(d_k, d_k) + d_f * d_f)
        reward = (
            -self.w_rad * e_rad
            - self.w_near * e_near
            - self.w_pos * e_pos
            + self.w_tan * v_tan
            - self.w_smooth * smooth
        )

        shaped = {"e_rad": float(e_rad), "e_near": float(e_near), "e_pos": float(e_pos), "v_tan": float(v_tan)}
        return float(reward), shaped
