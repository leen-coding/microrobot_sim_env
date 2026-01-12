import numpy as np
import matplotlib.pyplot as plt
from env import MicroRobotEnv
from math_utils import _normalize, quat_mul, quat_from_axis_angle, quat_from_two_vectors

def analyze_fv_curve(env):
    frequencies = np.linspace(0, 20, 200)  # 扫描 0 到 80 Hz
    velocities = []
    step_out_f = 0

    state = np.zeros(3, dtype=float)
    for f in frequencies:
        # 假设朝 X 方向前进
        _, info = env.dyn.velocity(state, [1, 0, 0, f], env.dt)
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

if __name__ == "__main__":
    env = MicroRobotEnv()
    analyze_fv_curve(env)
