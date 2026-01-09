import time
from pathlib import Path

import numpy as np
import pandas as pd
import pybullet as p

from env import MicroRobotEnv
from renderer_bullet import BulletRenderer
from dynamics import MicroRobotParams
from compute_rotating_average_force import build_precomputed_field, make_cached_force_callback


def _normalize(v, eps=1e-12):
    n = float(np.linalg.norm(v))
    if n < eps:
        return v * 0.0, 0.0
    return v / n, n


def main():
    base_dir = Path(__file__).resolve().parent
    urdf_path = base_dir / "robot_model" / "robot.urdf"

    renderer = BulletRenderer(
        urdf_path=urdf_path,
        body_axis=np.array([1, 0, 0]),
        r_LI=np.array([0.01067, -0.00298, -0.00001]),
        use_gui=True,
        gravity=(0, 0, 0),
    )

    parquet_path = base_dir / "actuation_matrices_45deg.parquet"
    cb = None
    if parquet_path.exists():
        df = pd.read_parquet(parquet_path)
        precomp = build_precomputed_field(df, theta=45.0)
        cb = make_cached_force_callback(precomp, grid_resolution=1e-4)

    params = MicroRobotParams()
    if cb is not None:
        grid_th = 1e-4

        def make_throttled(cb_fn, grid_th):
            state_last = {"key": None, "res": None}

            def throttled(state, **kwargs):
                key = (
                    round(state[0] / grid_th),
                    round(state[1] / grid_th),
                    round(state[2] / grid_th),
                )
                if key == state_last["key"] and state_last["res"] is not None:
                    return state_last["res"]
                res = cb_fn(state, **kwargs)
                state_last["key"] = key
                state_last["res"] = res
                return res

            return throttled

        params.external_force_callback = make_throttled(cb, grid_th)
        params.external_force_kwargs = {
            "axis_hat": (0, 0, 1),
            "B0": 0.01,
            "magnetic_moment": 1e-6,
            "n_phase": 18,
            "h": 1e-3,
        }

    env = MicroRobotEnv(dt=0.02, renderer=renderer, params=params)
    env.reset()

    k = np.array([0.0, 0.0, 1.0], dtype=float)
    f_hz = 2.0
    k_step = 0.05
    f_step = 0.2

    print("Teleop controls:")
    print("  W/S: +Y/-Y, A/D: -X/+X, R/F: +Z/-Z")
    print("  Z/X: -f/+f, SPACE: reset k, C: stop (f=0)")
    print("  ESC or Q: quit")

    try:
        while True:
            events = p.getKeyboardEvents()

            if (events.get(p.B3G_SPACE) or 0) & p.KEY_WAS_TRIGGERED:
                break
            if (events.get(ord("q")) or 0) & p.KEY_WAS_TRIGGERED:
                break

            if (events.get(ord("w")) or 0) & p.KEY_IS_DOWN:
                k += np.array([0.0, k_step, 0.0])
            if (events.get(ord("s")) or 0) & p.KEY_IS_DOWN:
                k += np.array([0.0, -k_step, 0.0])
            if (events.get(ord("a")) or 0) & p.KEY_IS_DOWN:
                k += np.array([-k_step, 0.0, 0.0])
            if (events.get(ord("d")) or 0) & p.KEY_IS_DOWN:
                k += np.array([k_step, 0.0, 0.0])
            if (events.get(ord("r")) or 0) & p.KEY_IS_DOWN:
                k += np.array([0.0, 0.0, k_step])
            if (events.get(ord("f")) or 0) & p.KEY_IS_DOWN:
                k += np.array([0.0, 0.0, -k_step])

            if (events.get(ord("z")) or 0) & p.KEY_IS_DOWN:
                f_hz = max(0.0, f_hz - f_step)
            if (events.get(ord("x")) or 0) & p.KEY_IS_DOWN:
                f_hz = f_hz + f_step

            if (events.get(ord(" ")) or 0) & p.KEY_WAS_TRIGGERED:
                k = np.array([0.0, 0.0, 1.0], dtype=float)
            if (events.get(ord("c")) or 0) & p.KEY_WAS_TRIGGERED:
                f_hz = 0.0

            k_hat, kn = _normalize(k)
            if kn < 1e-12:
                k_hat = np.array([0.0, 0.0, 1.0], dtype=float)

            if params.external_force_kwargs is not None:
                params.external_force_kwargs["axis_hat"] = k_hat.tolist()

            action = [k_hat[0], k_hat[1], k_hat[2], f_hz]
            env.step(action)
            time.sleep(1.0 / 60.0)
    finally:
        env.close()


if __name__ == "__main__":
    main()
