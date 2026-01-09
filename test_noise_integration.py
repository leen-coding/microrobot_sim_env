import numpy as np
import pandas as pd

from compute_rotating_average_force import compute_average_force_rotating_dipole
from dynamics import MicroRobotParams, MicroRobotDynamics


def mag_noise_cb(state, df, axis_hat=(0,0,1), B0=0.01, magnetic_moment=1e-6, n_phase=12):
    x, y, z = float(state[0]), float(state[1]), float(state[2])
    return compute_average_force_rotating_dipole(df, x, y, z,
                                                axis_hat=axis_hat,
                                                B0=B0,
                                                magnetic_moment=magnetic_moment,
                                                n_phase=n_phase)


def main():
    try:
        df = pd.read_parquet("actuation_matrices_45deg.parquet")
    except Exception as e:
        print("Failed to load parquet:", e)
        return

    params = MicroRobotParams(
        external_force_callback=mag_noise_cb,
        external_force_kwargs={'df': df, 'axis_hat': (0, 0, 1), 'B0': 0.01, 'magnetic_moment': 1e-6, 'n_phase': 12},
        mass_kg=0.043e-3
    )

    dyn = MicroRobotDynamics(params)
    state = np.array([0.0, 0.0, -0.19])
    action = np.array([0.0, 0.0, 1.0, 10.0])

    next_state, info = dyn.step(state, action, dt=0.01)

    print("next_state:", next_state)
    print("info keys:", list(info.keys()))
    for k, v in info.items():
        print(f"{k}: {v}")


if __name__ == '__main__':
    main()
