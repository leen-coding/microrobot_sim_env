import numpy as np
import pybullet as p
import pybullet_data
import time
from pathlib import Path

def _normalize(v, eps=1e-12):
    n = np.linalg.norm(v)
    if n < eps:
        return v * 0.0, 0.0
    return v / n, n

def quat_mul(q1, q2):
    # q = q1 ⊗ q2, both in (x,y,z,w)
    x1,y1,z1,w1 = q1
    x2,y2,z2,w2 = q2
    return np.array([
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
        w1*w2 - x1*x2 - y1*y2 - z1*z2
    ], dtype=float)

def quat_from_axis_angle(axis, angle):
    axis, n = _normalize(axis)
    if n < 1e-12:
        return np.array([0,0,0,1.0], dtype=float)
    s = np.sin(angle/2.0)
    return np.array([axis[0]*s, axis[1]*s, axis[2]*s, np.cos(angle/2.0)], dtype=float)

def quat_from_two_vectors(a, b):
    """
    quaternion that rotates unit vector a to unit vector b. returns (x,y,z,w)
    """
    a, na = _normalize(a)
    b, nb = _normalize(b)
    if na < 1e-12 or nb < 1e-12:
        return np.array([0,0,0,1.0], dtype=float)

    v = np.cross(a, b)
    c = float(np.dot(a, b))

    if c < -0.999999:
        # 180 deg: pick an orthogonal axis
        axis = np.array([1.0, 0.0, 0.0])
        if abs(a[0]) > 0.9:
            axis = np.array([0.0, 1.0, 0.0])
        axis = axis - a * np.dot(a, axis)
        axis, _ = _normalize(axis)
        return np.array([axis[0], axis[1], axis[2], 0.0], dtype=float)

    s = np.sqrt((1.0 + c) * 2.0)
    q = np.array([v[0]/s, v[1]/s, v[2]/s, 0.5*s], dtype=float)
    # normalize quaternion
    q = q / (np.linalg.norm(q) + 1e-12)
    return q


class MicroRobotEnv:
    def __init__(self, render=True, urdf_path=None, gui_dt=1/240):
        self.render_enabled = render
        self.gui_dt = gui_dt

        # --- simulation ---
        self.dt = 0.02
        self.state = np.zeros(3, dtype=float)  # [x, y, z] meters
        self.trajectory = []

        # Action: [k_x, k_y, k_z, f_hz]
        self.action_space = np.zeros(4, dtype=float)

        # --- magnetic field ---
        self.B0_mT = 5.0
        self.B0 = self.B0_mT * 1e-3

        self.m_mag = 4.08e-3
        self.m_mag_coff = 0.1

        # --- fluid + geometry ---
        self.eta = 0.34
        self.n_turns = 2
        self.R_helix = 1e-3
        self.theta = np.deg2rad(60.9)
        self.lam = 3.5e-3
        self.r_fil = 3e-4
        self.d_head = 2.5e-3

        self.drift = np.zeros(3)
        self.process_noise_std = 0.0

        self.z_soft_k = 0.0
        self.z_ref = 0.0

        # --- rendering state ---
        self._pb_connected = False
        self.robot_id = None
        self.body_axis = np.array([1.0, 0.0, 0.0])  # 机器人“前进轴”在本体坐标中的方向（按需改成 [1,0,0] 或 [0,1,0]）
        self.phi_spin = 0.0                         # 可选：用于可视化自转
        self.last_omega_eff = 0.0

        if self.render_enabled:
            if urdf_path is None:
                raise ValueError("render=True 需要提供 urdf_path")
            self._setup_pybullet(urdf_path)

        self.reset()

    def _setup_pybullet(self, urdf_path):
        cid = p.connect(p.GUI)
        self._pb_connected = True
        p.resetSimulation()
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.loadURDF("plane.urdf")
        p.setGravity(0, 0, 0)

        urdf_path = str(urdf_path)
        urdf_dir = str(Path(urdf_path).parent)
        p.setAdditionalSearchPath(urdf_dir)

        self.robot_id = p.loadURDF(
            urdf_path,
            basePosition=[0, 0, 0],
            baseOrientation=[0, 0, 0, 1],
            useFixedBase=True,
            flags=p.URDF_USE_INERTIA_FROM_FILE
        )

        # 相机：根据你模型尺度可调
        p.resetDebugVisualizerCamera(
            cameraDistance=0.08,
            cameraYaw=40,
            cameraPitch=-30,
            cameraTargetPosition=[0, 0, 0]
        )

    def close(self):
        if self._pb_connected and p.isConnected():
            p.disconnect()
        self._pb_connected = False

    def reset(self):
        self.state[:] = 0.0
        self.trajectory.clear()
        self.phi_spin = 0.0
        self.last_omega_eff = 0.0
        if self.render_enabled:
            self.render(action=np.array([0, 0, 1, 0], dtype=float))
        return self.state.copy()

    def step(self, action):
        v_vec, info = self.robot_kinematics(action)
        self.state = self.state + v_vec * self.dt
        self.trajectory.append(self.state.copy())

        # 保存自转角（仅用于渲染更像“在转”）
        self.last_omega_eff = float(info.get("omega_eff", 0.0))
        self.phi_spin += self.last_omega_eff * self.dt

        if self.render_enabled:
            self.render(action)
        return self.state.copy(), info

    def robot_kinematics(self, action):
        kx, ky, kz, f_hz = float(action[0]), float(action[1]), float(action[2]), float(action[3])

        k = np.array([kx, ky, kz], dtype=float)
        k_hat, kn = _normalize(k)
        if kn < 1e-12 or f_hz <= 0.0:
            return np.zeros(3), {
                "v_scalar": 0.0, "omega_cmd": 0.0, "omega_eff": 0.0,
                "f_step": 0.0, "beta": 0.0, "gamma": 0.0
            }

        a, b, c, xi_perp, xi_para = self._abc_coeffs()

        psi_v = 3.0 * np.pi * self.eta * self.d_head
        psi_omega = np.pi * self.eta * (self.d_head ** 3)

        denom = (a + psi_v)
        if abs(denom) < 1e-18:
            denom = np.sign(denom) * 1e-18 if denom != 0 else 1e-18

        beta = -b / denom
        gamma = (c + psi_omega) - (b ** 2) / denom

        omega_cmd = 2.0 * np.pi * f_hz

        tau_max = self.m_mag * self.B0 * self.m_mag_coff
        if gamma <= 0:
            omega_eff = omega_cmd
            omega_step = np.inf
            f_step = np.inf
        else:
            omega_step = tau_max / gamma
            omega_eff = min(omega_cmd, omega_step)
            f_step = omega_step / (2.0 * np.pi)

        v_scalar = beta * omega_eff
        v_vec = v_scalar * k_hat + self.drift

        if self.z_soft_k > 0:
            v_vec[2] += -self.z_soft_k * (self.state[2] - self.z_ref)

        if self.process_noise_std > 0:
            v_vec = v_vec + np.random.randn(3) * (self.process_noise_std / np.sqrt(self.dt))

        info = {
            "v_scalar": float(v_scalar),
            "omega_cmd": float(omega_cmd),
            "omega_eff": float(omega_eff),
            "omega_step": float(omega_step) if np.isfinite(omega_step) else omega_step,
            "f_step": float(f_step) if np.isfinite(f_step) else f_step,
            "beta": float(beta),
            "gamma": float(gamma),
            "tau_max": float(tau_max),
            "k_hat": k_hat.copy(),
        }
        return v_vec, info

    def _abc_coeffs(self):
        s = np.sin(self.theta)
        if s < 1e-12:
            s = 1e-12

        arg = 0.36 * self.lam / (self.r_fil * s)
        arg = max(arg, 1.001)

        ln_term = np.log(arg)
        xi_perp = (4.0 * np.pi * self.eta) / (ln_term + 0.5)
        xi_para = (2.0 * np.pi * self.eta) / (ln_term)

        cth = np.cos(self.theta)

        parameter_a = (xi_para * (cth ** 2) + xi_perp * (s ** 2)) / s
        parameter_b = (xi_para - xi_perp) * cth
        parameter_c = (xi_perp * (cth ** 2) + xi_para * (s ** 2)) / s

        a = 2.0 * np.pi * self.n_turns * self.R_helix * parameter_a
        b = 2.0 * np.pi * self.n_turns * (self.R_helix ** 2) * parameter_b
        c = 2.0 * np.pi * self.n_turns * (self.R_helix ** 3) * parameter_c

        return a, b, c, xi_perp, xi_para

    def render(self, action=None):
        if not (self._pb_connected and p.isConnected() and self.robot_id is not None):
            return

        # 取磁场旋转轴 k_hat 作为“推进方向”
        if action is None:
            k_hat = np.array([0.0, 0.0, 1.0])
        else:
            k = np.array(action[:3], dtype=float)
            k_hat, kn = _normalize(k)
            if kn < 1e-12:
                k_hat = np.array([0.0, 0.0, 1.0])

        # 1) 先把 body_axis 对齐到 k_hat
        q_align = quat_from_two_vectors(self.body_axis, k_hat)

        # 2) 可选：绕 k_hat 再转一个自转角（视觉更像螺旋在转）
        q_spin = quat_from_axis_angle(k_hat, self.phi_spin)

        # 合成：先对齐，再自转（顺序很关键）
        q = quat_mul(q_spin, q_align)

        r_LI = np.array([0.01067, -0.00298, -0.00001])  # 你的 inertial origin（米）
        R = np.array(p.getMatrixFromQuaternion(q)).reshape(3,3)
        pos_com = self.state + R @ r_LI
        p.resetBasePositionAndOrientation(self.robot_id, pos_com.tolist(), q.tolist())
        p.stepSimulation()
        time.sleep(self.gui_dt)


from pathlib import Path
import numpy as np

if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent
    urdf_path = base_dir / "robot_model" / "robot.urdf"

    env = MicroRobotEnv(render=True, urdf_path=urdf_path)

    try:
        f_hz = 6.0
        for t in range(2000):
            # 示例：让旋转轴做一个缓慢的圆锥扫描
            ang = 0.01 * t
            k = np.array([np.cos(ang), np.sin(ang), 0.3])
            action = [k[0], k[1], k[2], f_hz]

            state, info = env.step(action)

    except KeyboardInterrupt:
        pass
    finally:
        env.close()
