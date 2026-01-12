import numpy as np
from dynamics import MicroRobotParams, MicroRobotDynamics
from external_force_model import build_force_model_from_params


def main():
    params = MicroRobotParams()
    try:
        params.ext.force_fn = build_force_model_from_params(
            params,
            "actuation_matrices_45deg.parquet",
            n_phase=12,
            h=1e-3,
        )
    except Exception as e:
        print("Failed to load parquet:", e)
        return

    dyn = MicroRobotDynamics(params)
    state = np.array([0.0, 0.0, -0.22])
    action = np.array([0.0, 0.0, 1.0, 10.0])

    next_state, info = dyn.step(state, action, dt=0.01)

    print("next_state:", next_state)
    print("info keys:", list(info.keys()))
    for k, v in info.items():
        print(f"{k}: {v}")


if __name__ == '__main__':
    main()
