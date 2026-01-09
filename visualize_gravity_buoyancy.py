# visualize_gravity_buoyancy.py
import numpy as np
import matplotlib.pyplot as plt
import os
from env import MicroRobotEnv
from dynamics import MicroRobotParams

def run_sim(params, action, dt=0.01, T=5.0):
    env = MicroRobotEnv(dt=dt, params=params)
    state = env.reset(state0=[0.0, 0.0, 0.0])
    steps = int(T / dt)
    traj = np.zeros((steps, 3), dtype=float)
    vz = np.zeros(steps, dtype=float)
    a_gb = np.zeros(steps, dtype=float)
    omega_eff = np.zeros(steps, dtype=float)

    for i in range(steps):
        state, info = env.step(action)
        traj[i] = state
        vz[i] = info.get("v_noise", [0,0,0])[2] if info.get("v_noise") is not None else 0.0
        # prefer reported vertical velocity change components: v_noise + a_gb*dt
        a_gb[i] = info.get("a_gb", 0.0)
        omega_eff[i] = info.get("omega_eff", 0.0)

    return {
        "t": np.linspace(dt, T, steps),
        "traj": traj,
        "vz": np.gradient(traj[:,2], dt),
        "a_gb": a_gb,
        "omega_eff": omega_eff,
        "env": env,
    }


def main():
    dt = 0.01
    T = 5.0
    # action: kx,ky,kz, f_hz
    # use propulsion aligned with +z so we can see competition with gravity
    action = [0.0, 0.0, 1.0, 20.0]

    # params with gravity/buoyancy enabled
    params_g = MicroRobotParams()
    params_g.apply_gravity = True

    # params with gravity+buoyancy + magnetic gradient force
    params_g_mag = MicroRobotParams()
    params_g_mag.apply_gravity = True

    # define an external force callback that models magnetic gradient force: Fz = m_mag * dBdz
    def mag_gradient_force(state, dBdz=0.0):
        # state unused currently, but provided for generality
        Fz = float(params_g_mag.m_mag) * float(dBdz)
        return np.array([0.0, 0.0, Fz], dtype=float)

    # attach callback and kwargs to params
    params_g_mag.external_force_callback = mag_gradient_force
    # choose a representative gradient (T/m). Tune this to see visible effect.
    params_g_mag.external_force_kwargs = {"dBdz": 100.0}  # 100 T/m as example

    # params without gravity/buoyancy
    params_no_g = MicroRobotParams()
    params_no_g.apply_gravity = False

    res_g = run_sim(params_g, action, dt=dt, T=T)
    res_g_mag = run_sim(params_g_mag, action, dt=dt, T=T)
    res_no_g = run_sim(params_no_g, action, dt=dt, T=T)

    # plotting
    fig, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=True)

    axes[0].plot(res_g["t"], res_g["traj"][:,2], label="z (gravity+buoyancy)")
    axes[0].plot(res_g_mag["t"], res_g_mag["traj"][:,2], label="z (gravity+buoyancy+mag grad)")
    axes[0].plot(res_no_g["t"], res_no_g["traj"][:,2], label="z (no gravity)")
    axes[0].set_ylabel("z (m)")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(res_g["t"], res_g["vz"], label="vz (gravity+buoyancy)")
    axes[1].plot(res_g_mag["t"], res_g_mag["vz"], label="vz (gravity+buoyancy+mag grad)")
    axes[1].plot(res_no_g["t"], res_no_g["vz"], label="vz (no gravity)")
    axes[1].set_ylabel("vz (m/s)")
    axes[1].legend()
    axes[1].grid(True)

    axes[2].plot(res_g["t"], res_g["a_gb"], label="a_gb (gravity-buoyancy)")
    axes[2].plot(res_g_mag["t"], res_g_mag["a_gb"], label="a_gb (gravity-buoyancy) + mag grad")
    axes[2].set_ylabel("a_gb (m/s^2)")
    axes[2].set_xlabel("time (s)")
    axes[2].legend()
    axes[2].grid(True)

    out_png = os.path.join(os.getcwd(), "gravity_buoyancy_compare.png")
    fig.tight_layout()
    fig.savefig(out_png)
    print("Saved figure to:", out_png)

    # Print final summary
    print("Final z (with gravity):", res_g["traj"][-1,2])
    print("Final z (no gravity):", res_no_g["traj"][-1,2])
    print("Max a_gb:", np.max(np.abs(res_g["a_gb"])))

    plt.show()

if __name__ == '__main__':
    main()
