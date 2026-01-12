import argparse

import numpy as np

from gym_wrapper import MicroRobotGymEnv
from recorder import TrajectoryRecorder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="ppo_circle")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--dt", type=float, default=0.02)
    parser.add_argument("--radius", type=float, default=0.05)
    parser.add_argument("--period", type=float, default=10.0)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    try:
        from stable_baselines3 import PPO
    except Exception as exc:
        raise SystemExit(
            "stable-baselines3 is required. Install with: pip install stable-baselines3"
        ) from exc

    env = MicroRobotGymEnv(
        dt=args.dt,
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
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            recorder.on_step(env.env.state, action, info)
            done = terminated or truncated

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
