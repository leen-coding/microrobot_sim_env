import argparse
import time

import numpy as np

from gym_wrapper import MicroRobotGymEnv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--total-timesteps", type=int, default=200000)
    parser.add_argument("--dt", type=float, default=0.02)
    parser.add_argument("--radius", type=float, default=0.05)
    parser.add_argument("--period", type=float, default=10.0)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save-path", type=str, default="ppo_circle")
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

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        seed=args.seed,
        n_steps=2048,
        batch_size=64,
        learning_rate=3e-4,
        gamma=0.99,
    )

    start = time.time()
    from stable_baselines3.common.callbacks import BaseCallback

    class ActionRewardStatsCallback(BaseCallback):
        def __init__(self, log_every=1000):
            super().__init__()
            self.log_every = int(log_every)
            self.actions = []
            self.rewards = []

        def _on_step(self) -> bool:
            actions = self.locals.get("actions", None)
            rewards = self.locals.get("rewards", None)
            if actions is not None:
                self.actions.append(np.asarray(actions))
            if rewards is not None:
                self.rewards.append(np.asarray(rewards))
            if self.n_calls % self.log_every == 0:
                self._log_stats()
            return True

        def _log_stats(self):
            if not self.actions and not self.rewards:
                return
            if self.actions:
                acts = np.concatenate(self.actions, axis=0)
                mean = np.mean(acts, axis=0)
                std = np.std(acts, axis=0)
                print(f"[stats] action mean={mean} std={std}")
            if self.rewards:
                r = np.concatenate(self.rewards, axis=0)
                print(f"[stats] reward mean={np.mean(r):.6f} std={np.std(r):.6f}")
            self.actions.clear()
            self.rewards.clear()

    callback = ActionRewardStatsCallback(log_every=1000)
    model.learn(total_timesteps=args.total_timesteps, callback=callback)
    elapsed = time.time() - start
    print(f"Training finished in {elapsed:.1f}s")

    model.save(args.save_path)
    print(f"Saved model to {args.save_path}")

    env.close()


if __name__ == "__main__":
    main()
