import argparse

import numpy as np

from gym_wrapper import MicroRobotGymEnv
from recorder import TrajectoryRecorder
from pathlib import Path
from external_force_model import build_force_model_from_params
base_dir = Path(__file__).resolve().parent
from dynamics import MicroRobotParams
from renderer_bullet import BulletRenderer
urdf_path = base_dir / "robot_model" / "robot.urdf"
renderer = BulletRenderer(
    urdf_path=urdf_path,
    body_axis=np.array([1,0,0]),                 # 你已确认螺旋轴为 STL 的 x
    r_LI=np.array([0.01067,-0.00298,-0.00001]),  # URDF inertial origin（米）
    use_gui=True,
    gravity=(0,0,0),                              # 仅渲染，建议 0
)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="ppo_circle")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--dt", type=float, default=0.02)
    parser.add_argument("--radius", type=float, default=0.05)
    parser.add_argument("--period", type=float, default=10.0)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--deterministic", action="store_true")
    args = parser.parse_args()

    try:
        from stable_baselines3 import PPO
    except Exception as exc:
        raise SystemExit(
            "stable-baselines3 is required. Install with: pip install stable-baselines3"
        ) from exc
        
    params = MicroRobotParams()
    parquet_path = base_dir / "actuation_matrices_45deg.parquet"
    if parquet_path.exists():
        params.ext.force_fn = build_force_model_from_params(params, parquet_path)
    env = MicroRobotGymEnv(
        dt=args.dt,
        renderer=renderer,
        params=params,
        circle_radius=args.radius,
        circle_center=(0.0, 0.0),
        circle_z=0.0,
        circle_period=args.period,
        max_steps=args.max_steps,
        
    )

    model = PPO.load(args.model_path)

    rng = np.random.default_rng(args.seed)
    metrics_all = []

    for ep in range(args.episodes):
        obs, _ = env.reset(seed=int(rng.integers(0, 1_000_000)))
        recorder = TrajectoryRecorder(enabled=True)
        recorder.on_reset(env.env.state, params=env.env.params)

        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=args.deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            recorder.on_step(env.env.state, action, info)
            done = terminated or truncated
            if env.step_count % 50 == 0:
                print(f"[step {env.step_count}] action={action} f_cmd={info.get('f_cmd')}")

        # endpoint error: distance to last reference point
        target_xyz = info.get("ref", None)
        metrics = recorder.finalize_episode(target_xyz=target_xyz, dt=args.dt)
        metrics_all.append(metrics)
        print(f"Episode {ep}: {metrics}")

    # aggregate
    if metrics_all:
        avg_speed = np.mean([m["avg_speed"] for m in metrics_all if m["avg_speed"] is not None])
        sync_ratio = np.mean([m["sync_ratio"] for m in metrics_all if m["sync_ratio"] is not None])
        endpoint_error = np.mean([m["endpoint_error"] for m in metrics_all if m["endpoint_error"] is not None])
        print("\nAggregate:")
        print(f"  avg_speed: {avg_speed:.6f}")
        print(f"  sync_ratio: {sync_ratio:.6f}")
        print(f"  endpoint_error: {endpoint_error:.6f}")

    env.close()


if __name__ == "__main__":
    main()
