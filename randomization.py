# randomization.py
import numpy as np
from typing import Any, Callable, Dict, Optional, Tuple, Union

Spec = Union[None, Dict[str, Any], Callable[[Any, np.random.Generator], Any]]


def _sample_uniform(rng: np.random.Generator, value):
    if isinstance(value, (list, tuple)) and len(value) == 2:
        low, high = float(value[0]), float(value[1])
        return float(rng.uniform(low, high))
    return float(value)


def _sample_drift(rng: np.random.Generator, value):
    """
    drift spec supports:
      - [dx,dy,dz]
      - {"dist":"uniform","low":[...],"high":[...]}
      - {"dist":"normal","mean":[...],"std":[...]}
      - (low, high) -> uniform scalar range for each axis
    """
    if isinstance(value, (list, tuple)) and len(value) == 3:
        return np.array(value, dtype=float)

    if isinstance(value, dict):
        dist = value.get("dist", "normal")
        if dist == "uniform":
            low = np.asarray(value.get("low", [0.0, 0.0, 0.0]), dtype=float)
            high = np.asarray(value.get("high", [0.0, 0.0, 0.0]), dtype=float)
            return rng.uniform(low, high)
        mean = np.asarray(value.get("mean", [0.0, 0.0, 0.0]), dtype=float)
        std = np.asarray(value.get("std", [0.0, 0.0, 0.0]), dtype=float)
        return rng.normal(mean, std)

    if isinstance(value, (list, tuple)) and len(value) == 2:
        low, high = float(value[0]), float(value[1])
        return rng.uniform(low, high, size=3)

    return np.zeros(3, dtype=float)


def apply_episode_randomization(params, rng: np.random.Generator, spec: Spec) -> Dict[str, Any]:
    """
    Apply episode-level domain randomization IN-PLACE on params.

    Returns a dict describing the sampled values for logging / reproducibility.
    """
    if spec is None:
        return {}

    # callable: user fully controls
    if callable(spec):
        out = spec(params, rng)
        # allow callable to return sampled dict; if None, return empty
        return {} if out is None else dict(out)

    if not isinstance(spec, dict):
        return {}

    sampled: Dict[str, Any] = {}

    if "eta" in spec:
        params.env.eta = _sample_uniform(rng, spec["eta"])
        sampled["eta"] = float(params.env.eta)

    if "m_scale" in spec:
        params.mag.m_scale = _sample_uniform(rng, spec["m_scale"])
        sampled["m_scale"] = float(params.mag.m_scale)

    if "drift" in spec:
        params.noise.drift = _sample_drift(rng, spec["drift"])
        sampled["drift"] = np.asarray(params.noise.drift, dtype=float).tolist()

    # You can extend here later: theta_rad, r_fil, B0_mT, etc.
    return sampled
