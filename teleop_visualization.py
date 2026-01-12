import time
from pathlib import Path

import numpy as np
import pybullet as p

from env import MicroRobotEnv
from renderer_bullet import BulletRenderer
from dynamics import MicroRobotParams
from external_force_model import build_force_model_from_params


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

    params = MicroRobotParams()
    parquet_path = base_dir / "actuation_matrices_45deg.parquet"
    if parquet_path.exists():
        params.ext.force_fn = build_force_model_from_params(params, parquet_path)


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

            action = [k_hat[0], k_hat[1], k_hat[2], f_hz]
            state, info = env.step(action)
            if renderer is not None:
                renderer.update(
                    state_xyz=state,
                    k_hat=info.get("k_hat", [0, 0, 1]),
                    phi_spin=env.phi_spin,
                )
            time.sleep(1.0 / 60.0)
    finally:
        env.close()


if __name__ == "__main__":
    main()
