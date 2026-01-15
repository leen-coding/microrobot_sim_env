import argparse
import time

import numpy as np

from dynamics import MicroRobotParams
from pathlib import Path
from external_force_model import build_force_model_from_params
base_dir = Path(__file__).resolve().parent
from gym_wrapper import MicroRobotGymEnv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--total-timesteps", type=int, default=2000000)
    parser.add_argument("--dt", type=float, default=0.02)
    parser.add_argument("--radius", type=float, default=0.05)
    parser.add_argument("--period", type=float, default=15.0)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save-path", type=str, default="ppo_circle")
    parser.add_argument("--tensorboard-log", type=str, default=None)
    parser.add_argument("--tb-log-name", type=str, default="ppo_circle")
    parser.add_argument("--log-every", type=int, default=1000)
    args = parser.parse_args()

    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.monitor import Monitor
        from stable_baselines3.common.vec_env import DummyVecEnv
    except Exception as exc:
        raise SystemExit(
            "stable-baselines3 is required. Install with: pip install stable-baselines3"
        ) from exc

    params = MicroRobotParams()
    parquet_path = base_dir / "actuation_matrices_45deg.parquet"
    if parquet_path.exists():
        params.ext.force_fn = build_force_model_from_params(params, parquet_path)
    
    def make_env():
        base_env = MicroRobotGymEnv(
            dt=args.dt,
            params=params,
            renderer=None,
            render_mode=None,
            circle_radius=args.radius,
            circle_center=(0.0, 0.0),
            circle_z=0.0,
            circle_period=args.period,
            max_steps=args.max_steps,
            k_rate_rad_s=np.deg2rad(30.0),
            f_rate_hz_s=5.0,
            f_min=1.0,
            f_max=10.0,
            w_rad=2.0,
            w_pos=0.0,
            w_near=1.0,
            w_tan=0.5,
            w_smooth=0.05,
        )
        return Monitor(base_env)

    env = DummyVecEnv([make_env])

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        seed=args.seed,
        n_steps=2048,
        batch_size=64,
        learning_rate=3e-4,
        gamma=0.99,
        tensorboard_log=args.tensorboard_log,
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
                for idx, (m, s) in enumerate(zip(mean, std)):
                    self.logger.record(f"stats/action_mean_{idx}", float(m))
                    self.logger.record(f"stats/action_std_{idx}", float(s))
            if self.rewards:
                r = np.concatenate(self.rewards, axis=0)
                self.logger.record("stats/reward_mean", float(np.mean(r)))
                self.logger.record("stats/reward_std", float(np.std(r)))
            self.actions.clear()
            self.rewards.clear()

    callback = ActionRewardStatsCallback(log_every=args.log_every)
    model.learn(
        total_timesteps=args.total_timesteps,
        callback=callback,
        tb_log_name=args.tb_log_name,
    )
    elapsed = time.time() - start
    print(f"Training finished in {elapsed:.1f}s")

    model.save(args.save_path)
    print(f"Saved model to {args.save_path}")

    env.close()


if __name__ == "__main__":
    main()
