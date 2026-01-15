# recorder.py
import numpy as np
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


def snapshot_params(params) -> Dict[str, Any]:
    """Stable params snapshot for episode metadata."""
    p = params
    return {
        "mag": {"B0_mT": float(p.mag.B0_mT), 
        "m_mag": float(p.mag.m_mag), 
        "m_scale": float(p.mag.m_scale)},
        "helix": {
            "n_turns": float(p.helix.n_turns),
            "R_helix": float(p.helix.R_helix),
            "theta_rad": float(p.helix.theta_rad),
            "lam": float(p.helix.lam),
            "r_fil": float(p.helix.r_fil),
            "d_head": float(p.helix.d_head),
            "mass_kg": float(p.helix.mass_kg),
            "volume_m3": float(p.helix.volume_m3),
        },
        "env": {
            "eta": float(p.env.eta),
            "fluid_density_kg_m3": float(p.env.fluid_density_kg_m3),
            "gravity": float(p.env.gravity),
            "apply_gravity": bool(p.env.apply_gravity),
        },
        "noise": {
            "drift": np.asarray(p.noise.drift, dtype=float).tolist(),
            "process_noise_std": float(p.noise.process_noise_std),
        },
    }


@dataclass
class TrajectoryRecorder:
    enabled: bool = False
    store_params_in_meta: bool = True

    trajectory: List[Dict[str, Any]] = field(default_factory=list)
    episode_meta: Dict[str, Any] = field(default_factory=dict)

    def reset(self):
        self.trajectory.clear()
        self.episode_meta.clear()

    def on_reset(self, state_xyz, *, params=None, randomization=None, extra_meta: Optional[Dict[str, Any]] = None):
        if not self.enabled:
            return
        self.trajectory.clear()

        meta: Dict[str, Any] = {}
        if self.store_params_in_meta and params is not None:
            meta["params"] = snapshot_params(params)
        if randomization is not None:
            meta["randomization"] = dict(randomization)
        if extra_meta:
            meta.update(extra_meta) #字典合并

        self.episode_meta = meta

    def on_step(self, state_xyz, action, info: Dict[str, Any]):
        if not self.enabled:
            return
        self.trajectory.append({
            "state": np.asarray(state_xyz, dtype=float).tolist(),
            "action": np.asarray(action, dtype=float).tolist(),
            "info": dict(info),
        })

    def get_trajectory(self):
        return list(self.trajectory)

    def get_episode_meta(self):
        return dict(self.episode_meta)

    def to_dataframe(self):
        """Optional helper; only imports pandas when called."""
        import pandas as pd
        return pd.DataFrame(self.trajectory)

    def finalize_episode(self, *, target_xyz=None, dt: Optional[float] = None):
        """Compute and store episode-level metrics in episode_meta."""
        metrics = self.compute_episode_metrics(target_xyz=target_xyz, dt=dt)
        self.episode_meta["metrics"] = metrics
        return metrics

    def compute_episode_metrics(self, *, target_xyz=None, dt: Optional[float] = None) -> Dict[str, Any]:
        '''
        在 Python 函数参数列表中的这个单独的 *，被称为**强制关键字参数（Keyword-Only Arguments）**分割符。

        它的意思是：* 后面的所有参数，在调用函数时必须显式地写出参数名，不能只传数值。
        '''

        if not self.trajectory:
            return {
                "avg_speed": None,
                "sync_ratio": None,
                "endpoint_error": None,
            }

        v_scalars = []
        sync_flags = []
        states = []
        for step in self.trajectory:
            info = step.get("info", {})
            if "v_scalar" in info and info["v_scalar"] is not None:
                v_scalars.append(abs(float(info["v_scalar"])))
            if "sync" in info:
                sync_flags.append(bool(info["sync"]))
            states.append(np.asarray(step.get("state", [0.0, 0.0, 0.0]), dtype=float))

        avg_speed = None
        if v_scalars:
            avg_speed = float(np.mean(v_scalars))
        elif dt is not None and len(states) > 1:
            diffs = np.diff(np.vstack(states), axis=0)
            speeds = np.linalg.norm(diffs, axis=1) / float(dt)
            avg_speed = float(np.mean(speeds)) if speeds.size > 0 else None

        sync_ratio = None
        if sync_flags:
            sync_ratio = float(np.mean(sync_flags))

        endpoint_error = None
        if target_xyz is not None and states:
            target = np.asarray(target_xyz, dtype=float).reshape(3)
            endpoint_error = float(np.linalg.norm(states[-1] - target))

        return {
            "avg_speed": avg_speed,
            "sync_ratio": sync_ratio,
            "endpoint_error": endpoint_error,
        }
