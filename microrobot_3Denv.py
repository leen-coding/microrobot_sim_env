import numpy as np
import matplotlib.pyplot as plt
import pyvista as pv
class MicroRobotEnv:
    def __init__(self, render=True):
        self.render_enabled = render
        self.trajectory = []
        if render:
            self._setup_pyvista()

        # --- simulation ---
        self.dt = 0.02
        self.state = np.zeros(3, dtype=float)  # [x, y, z] in meters

        # Action: [k_x, k_y, k_z, f_hz]
        self.action_space = np.zeros(4, dtype=float)

        # --- magnetic field ---
        self.B0_mT = 5.0
        self.B0 = self.B0_mT * 1e-3  # Tesla

        # Magnetic moment model: tau_max = m_mag * B0 * m_mag_coff
        self.m_mag = 4.08e-3
        self.m_mag_coff = 0.1

        # --- fluid + geometry ---
        self.eta = 0.34
        self.n_turns = 2
        self.R_helix = 1e-3
        self.theta = np.deg2rad(60.9)
        self.lam = 3.5e-3
        self.r_fil = 3e-4

        # Head diameter
        self.d_head = 2.5e-3

        # Optional drift/noise (3D)
        self.drift = np.zeros(3)
        self.process_noise_std = 0.0

        # Optional: keep robot near a plane (e.g., imaging plane). Set to 0 to disable.
        self.z_soft_k = 0.0       # (1/s)  z restoring strength
        self.z_ref = 0.0          # meters z reference

        self.reset()
        
    def _setup_pyvista(self):
        """初始化 PyVista 渲染器"""
        self.plotter = pv.Plotter(title="3D Microrobot Simulation (PyVista)")
        self.plotter.set_background("white")  # 设置背景色
        
        # 1. 创建机器人头部网格 (球体)
        self.head_mesh = pv.Sphere(radius=self.d_head/2)
        self.head_actor = self.plotter.add_mesh(self.head_mesh, color="steelblue", smooth_shading=True)
        
        # 2. 创建螺旋尾部网格 (简单的圆柱体模拟)
        self.tail_mesh = pv.Cylinder(radius=self.r_fil*2, height=0.005)
        self.tail_actor = self.plotter.add_mesh(self.tail_mesh, color="gray")

        # 3. 创建轨迹线
        self.traj_poly = pv.PolyData()
        self.traj_actor = self.plotter.add_mesh(self.traj_poly, color="cyan", line_width=2, opacity=0.6)

        # 添加坐标轴和网格
        self.plotter.add_axes()
        self.plotter.show_grid()
        self.plotter.show(interactive_update=True)


    def reset(self):
        self.state[:] = 0.0
        self.trajectory.clear()
        return self.state.copy()

    def step(self, action):
        v_vec, info = self.robot_kinematics(action)
        self.state = self.state + v_vec * self.dt
        self.trajectory.append(self.state.copy())

        if self.render_enabled:
            self.render(action)
        return self.state.copy(), info

    def robot_kinematics(self, action):
        """
        action = [k_x, k_y, k_z, f_hz]
        k is the rotation axis of the rotating magnetic field in world frame.
        We assume the robot aligns with k (implicit attitude), so swimming direction = k_hat.
        """
        kx, ky, kz, f_hz = float(action[0]), float(action[1]), float(action[2]), float(action[3])

        k = np.array([kx, ky, kz], dtype=float)
        kn = np.linalg.norm(k)
        if kn < 1e-12 or f_hz <= 0.0:
            return np.zeros(3), {
                "v_scalar": 0.0, "omega_cmd": 0.0, "omega_eff": 0.0,
                "f_step": 0.0, "beta": 0.0, "gamma": 0.0
            }
        k_hat = k / kn

        # --- propulsion coefficients ---
        a, b, c, xi_perp, xi_para = self._abc_coeffs()

        # --- head drag ---
        psi_v = 3.0 * np.pi * self.eta * self.d_head
        psi_omega = np.pi * self.eta * (self.d_head ** 3)

        # --- alpha,beta,gamma form ---
        denom = (a + psi_v)
        if abs(denom) < 1e-18:
            denom = np.sign(denom) * 1e-18 if denom != 0 else 1e-18

        beta = -b / denom
        gamma = (c + psi_omega) - (b ** 2) / denom

        # --- commanded angular speed ---
        omega_cmd = 2.0 * np.pi * f_hz

        # --- step-out limit ---
        tau_max = self.m_mag * self.B0 * self.m_mag_coff
        if gamma <= 0:
            omega_eff = omega_cmd
            omega_step = np.inf
            f_step = np.inf
        else:
            omega_step = tau_max / gamma
            omega_eff = min(omega_cmd, omega_step)
            f_step = omega_step / (2.0 * np.pi)

        # --- forward speed along k_hat ---
        v_scalar = beta * omega_eff
        v_vec = v_scalar * k_hat + self.drift

        # Optional: softly constrain z around z_ref (useful if your experiments are near a plane)
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
            "a": float(a), "b": float(b), "c": float(c),
            "psi_v": float(psi_v), "psi_omega": float(psi_omega),
            "xi_perp": float(xi_perp), "xi_para": float(xi_para),
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
        """
        简化渲染：仍用 x-y 平面显示轨迹；z 不显示。
        如需 3D 可视化，建议另写一个 matplotlib 3D plot。
        """
        self.ax.clear()
        self.ax.set_facecolor('white')
        self.ax.grid(True, which='both', color='#e0e0e0', linestyle='-')

        if len(self.trajectory) > 2:
            traj_pts = np.array(self.trajectory)
            self.ax.plot(traj_pts[:, 0], traj_pts[:, 1], color='cyan', alpha=0.5, lw=1)

        x, y, z = self.state

        # draw heading projection in XY based on k_hat
        if action is not None:
            k = np.array(action[:3], dtype=float)
            kn = np.linalg.norm(k)
            if kn > 1e-12:
                k_hat = k / kn
                dx, dy = k_hat[0], k_hat[1]
                scale = 0.003
                self.ax.plot([x, x + dx * scale], [y, y + dy * scale],
                             color='#555555', lw=3, solid_capstyle='round')
                head = plt.Circle((x, y), self.d_head / 2, color='#2c3e50')
                self.ax.add_artist(head)

        limit = 0.03
        self.ax.set_xlim(x - limit, x + limit)
        self.ax.set_ylim(y - limit, y + limit)

        self.ax.set_aspect('equal')
        self.ax.set_title("Top View (XY): Helical Microrobot (3D state, implicit attitude)", fontsize=10)
        plt.pause(0.001)
