import numpy as np
from functools import lru_cache
from typing import Optional, Tuple
from compute_rotating_average_force import compute_average_force_rotating_dipole, build_precomputed_field


def load_precomputed_field(parquet_path, theta: float = 45.0):
    import pandas as pd

    df = pd.read_parquet(parquet_path)
    return build_precomputed_field(df, theta=theta)


def build_force_model_from_parquet(
    parquet_path,
    *,
    fixed_xy=(0.0, 0.0),
    fixed_z=0.21,
    B0_T=0.005,
    m_eff=1e-6,
    n_phase=18,
    h=1e-3,
    grid=1e-4,
    cache_size=4096,
    suppress_errors=True,
):
    if not parquet_path:
        return None
    precomp = load_precomputed_field(parquet_path, theta=45.0)
    return ExternalForceModel(
        precomp,
        fixed_xy=fixed_xy,
        fixed_z=fixed_z,
        B0_T=B0_T,
        m_eff=m_eff,
        n_phase=n_phase,
        h=h,
        grid=grid,
        cache_size=cache_size,
        suppress_errors=suppress_errors,
    )


def build_force_model_from_params(
    params,
    parquet_path,
    *,
    fixed_xy=(0.0, 0.0),
    fixed_z=0.21,
    n_phase=18,
    h=1e-3,
    grid=1e-4,
    cache_size=4096,
    suppress_errors=True,
    B0_T=None,
    m_eff=None,
):
    if params is None:
        return None
    if B0_T is None:
        B0_T = params.mag.B0_mT * 1e-3
    if m_eff is None:
        m_eff = params.mag.m_mag * params.mag.m_scale
    return build_force_model_from_parquet(
        parquet_path,
        fixed_xy=fixed_xy,
        fixed_z=fixed_z,
        B0_T=B0_T,
        m_eff=m_eff,
        n_phase=n_phase,
        h=h,
        grid=grid,
        cache_size=cache_size,
        suppress_errors=suppress_errors,
    )


class ExternalForceModel:
    """
    Compute magnetic gradient force using a precomputed field.

    By default this is evaluated at a fixed position.
    """

    def __init__(
        self,
        precomp,
        grid: float = 1e-4,
        fixed_pos: Optional[Tuple[float, float, float]] = None,
        fixed_xy: Tuple[float, float] = (0.0, 0.0),
        fixed_z: float = 0.21,
        B0_T: float = 0.005,
        m_eff: float = 4.08e-3 * 0.06,
        n_phase: int = 18,
        h: float = 1e-3,
        cache_size: int = 4096,
        suppress_errors: bool = True,
    ):
        self.precomp = precomp
        self.grid = float(grid)
        self.fixed_pos = None if fixed_pos is None else tuple(fixed_pos)
        self.fixed_xy = tuple(fixed_xy)
        self.fixed_z = float(fixed_z)
        self.B0_T = float(B0_T)
        self.m_eff = float(m_eff)
        self.n_phase = int(n_phase)
        self.h = float(h)
        self.suppress_errors = bool(suppress_errors)
        self._cached = self._make_cache(cache_size) if precomp is not None else None

    def _q(self, v: float) -> float:
        return round(v / self.grid) * self.grid

    def _make_cache(self, cache_size):
        @lru_cache(maxsize=cache_size)
        def f(xq, yq, zq, axis_hat, B0, m_eff, n_phase, h):
            res = compute_average_force_rotating_dipole(
                self.precomp,
                xq,
                yq,
                zq,
                axis_hat=axis_hat,
                B0=B0,
                magnetic_moment=m_eff,
                n_phase=n_phase,
                h=h,
            )
            if isinstance(res, dict) and "F_mean" in res:
                return tuple(np.asarray(res["F_mean"], dtype=float))
            return tuple(np.asarray(res, dtype=float))

        return f

    def __call__(self, state=None, k_hat=(0.0, 0.0, 1.0), omega_eff=None, **_kwargs):
        if self._cached is None:
            return None

        try:
            if self.fixed_pos is not None:
                x, y, z = self.fixed_pos
            elif self.fixed_xy is not None and self.fixed_z is not None:
                x, y = self.fixed_xy
                z = self.fixed_z
            elif state is not None:
                x, y, z = float(state[0]), float(state[1]), float(state[2])
            else:
                x, y, z = 0.0, 0.0, 0.0

            xq, yq, zq = self._q(x), self._q(y), self._q(z)
            F = self._cached(
                xq,
                yq,
                zq,
                tuple(k_hat),
                float(self.B0_T),
                float(self.m_eff),
                int(self.n_phase),
                float(self.h),
            )
            return np.array(F, dtype=float)
        except Exception:
            if self.suppress_errors:
                return None
            raise
