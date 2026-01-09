import numpy as np
import matplotlib.pyplot as plt
from base_code.microrobot_env import MicroRobotEnv

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

def analyze_fv_curve(env):
    frequencies = np.linspace(0, 20, 200)  # 扫描 0 到 80 Hz
    velocities = []
    step_out_f = 0
    
    # 临时关闭渲染以加快计算
    orig_render = env.render_enabled
    env.render_enabled = False
    
    for f in frequencies:
        # 假设朝 X 方向前进
        _, info = env.robot_kinematics([1, 0, f])
        velocities.append(info["v_scalar"] * 1000)  # 转换为 mm/s
        step_out_f = info["f_step"]

    plt.figure(figsize=(8, 5))
    plt.plot(frequencies, velocities, label="Swimming Velocity", color='royalblue', lw=2)
    plt.axvline(x=step_out_f, color='red', linestyle='--', label=f'Step-out ({step_out_f:.1f} Hz)')
    
    plt.title("Microrobot Frequency-Velocity Response", fontsize=12)
    plt.xlabel("Frequency (Hz)", fontsize=10)
    plt.ylabel("Velocity (mm/s)", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()

    env.render_enabled = orig_render

if __name__ == "__main__":
    env = MicroRobotEnv(render=False)
    analyze_fv_curve(env)