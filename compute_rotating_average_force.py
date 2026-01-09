"""
计算旋转平均力脚本
给定：
  1. 磁场大小 B0（Tesla）
  2. 旋转轴方向 axis_hat（单位向量）
  3. 作用位置 (x, y, z)
  4. 被控机器人磁矩大小 |m|（A·m²）
  
从 45° 的离线数据库中提取雅可比矩阵，计算旋转一圈时的平均力
假设：磁矩方向与磁场方向始终一致（锁相）
"""

import numpy as np
import pandas as pd
from numpy.linalg import norm
from functools import lru_cache
from typing import Callable, Tuple, Optional, Dict


PARQUET_45DEG = "actuation_matrices_45deg.parquet"


def row_to_A(row: pd.Series) -> np.ndarray:
    """从数据框行提取 3x3 雅可比矩阵"""
    return np.array([
        [row["A11"], row["A12"], row["A13"]],
        [row["A21"], row["A22"], row["A23"]],
        [row["A31"], row["A32"], row["A33"]],
    ], dtype=float)


def find_nearest_row(df_t: pd.DataFrame, x: float, y: float, z: float) -> pd.Series:
    """在给定数据框中找最近的网格点"""
    if df_t.empty:
        raise ValueError("Empty DataFrame passed to find_nearest_row.")
    d2 = (df_t["x"]-x)**2 + (df_t["y"]-y)**2 + (df_t["z"]-z)**2
    idx = d2.idxmin()
    return df_t.loc[idx]


class PrecomputedField:
    """
    Precompute numeric arrays + nearest-neighbor index for faster queries.
    Provides cached spatial derivatives of A via central diff or local fit.
    """
    def __init__(self, df: pd.DataFrame, theta: float = 45.0, grid_resolution: Optional[float] = None):
        df_t = df[df["theta"] == float(theta)].copy()
        if df_t.empty:
            raise ValueError(f"No data for theta={theta} in database")

        self.coords = df_t[["x", "y", "z"]].to_numpy(dtype=float)
        a_cols = ["A11","A12","A13","A21","A22","A23","A31","A32","A33"]
        self.A = df_t[a_cols].to_numpy(dtype=float).reshape(-1, 3, 3)
        self.theta = float(theta)

        if grid_resolution is None:
            grid_resolution = self._infer_grid_resolution()
        self.grid_resolution = float(grid_resolution)

        self.index_map = self._build_index_map()
        self.deriv_cache: Dict[Tuple[int, float], Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        self.kdtree = self._build_kdtree()

    def _infer_grid_resolution(self) -> float:
        res = []
        for axis in range(3):
            vals = np.unique(self.coords[:, axis])
            if len(vals) < 2:
                continue
            diffs = np.diff(vals)
            diffs = diffs[diffs > 0]
            if len(diffs) > 0:
                res.append(float(np.min(diffs)))
        if not res:
            return 1.0
        return float(min(res))

    def _quantize(self, v: float) -> float:
        return round(v / self.grid_resolution) * self.grid_resolution

    def _build_index_map(self) -> Dict[Tuple[float, float, float], int]:
        idx_map: Dict[Tuple[float, float, float], int] = {}
        for i, (x, y, z) in enumerate(self.coords):
            key = (self._quantize(x), self._quantize(y), self._quantize(z))
            idx_map[key] = i
        return idx_map

    def _build_kdtree(self):
        try:
            from scipy.spatial import cKDTree  # type: ignore
        except Exception:
            return None
        return cKDTree(self.coords)

    def nearest_index(self, x: float, y: float, z: float) -> int:
        if self.kdtree is not None:
            _, idx = self.kdtree.query([x, y, z], k=1)
            return int(idx)
        target = np.array([x, y, z], dtype=float)
        d2 = np.sum((self.coords - target) ** 2, axis=1)
        return int(np.argmin(d2))

    def _neighbor_index(self, x: float, y: float, z: float) -> Optional[int]:
        key = (self._quantize(x), self._quantize(y), self._quantize(z))
        return self.index_map.get(key, None)

    def _local_fit_gradients(self, idx: int, k: int = 27) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        center = self.coords[idx]
        if self.kdtree is not None:
            _, inds = self.kdtree.query(center, k=min(k, len(self.coords)))
            inds = np.atleast_1d(inds)
        else:
            d2 = np.sum((self.coords - center) ** 2, axis=1)
            kk = min(k, len(self.coords))
            inds = np.argpartition(d2, kk - 1)[:kk]

        neigh = self.coords[inds]
        dx = neigh[:, 0] - center[0]
        dy = neigh[:, 1] - center[1]
        dz = neigh[:, 2] - center[2]
        X = np.stack([np.ones(len(neigh)), dx, dy, dz], axis=1)
        Y = self.A[inds].reshape(len(neigh), 9)
        beta, *_ = np.linalg.lstsq(X, Y, rcond=None)
        dA_dx = beta[1].reshape(3, 3)
        dA_dy = beta[2].reshape(3, 3)
        dA_dz = beta[3].reshape(3, 3)
        return dA_dx, dA_dy, dA_dz

    def get_dA(self, idx: int, h: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        key = (int(idx), float(h))
        cached = self.deriv_cache.get(key)
        if cached is not None:
            return cached

        x0, y0, z0 = self.coords[idx]
        x_p = self._neighbor_index(x0 + h, y0, z0)
        x_m = self._neighbor_index(x0 - h, y0, z0)
        y_p = self._neighbor_index(x0, y0 + h, z0)
        y_m = self._neighbor_index(x0, y0 - h, z0)
        z_p = self._neighbor_index(x0, y0, z0 + h)
        z_m = self._neighbor_index(x0, y0, z0 - h)

        if None not in (x_p, x_m, y_p, y_m, z_p, z_m):
            dA_dx = (self.A[x_p] - self.A[x_m]) / (2.0 * h)
            dA_dy = (self.A[y_p] - self.A[y_m]) / (2.0 * h)
            dA_dz = (self.A[z_p] - self.A[z_m]) / (2.0 * h)
        else:
            dA_dx, dA_dy, dA_dz = self._local_fit_gradients(idx)

        self.deriv_cache[key] = (dA_dx, dA_dy, dA_dz)
        return dA_dx, dA_dy, dA_dz


def build_precomputed_field(df: pd.DataFrame, theta: float = 45.0, grid_resolution: Optional[float] = None) -> PrecomputedField:
    return PrecomputedField(df, theta=theta, grid_resolution=grid_resolution)


def get_A_at_45deg(df: pd.DataFrame, x: float, y: float, z: float):
    """
    从数据库中获取位置 (x,y,z) 处 45° 的雅可比矩阵
    返回：A(3x3), 实际使用的坐标(x_used,y_used,z_used)
    """
    df_45 = df[df["theta"] == 45.0].copy()
    if df_45.empty:
        raise ValueError("No data for theta=45.0 in database")
    
    row = find_nearest_row(df_45, x, y, z)
    A = row_to_A(row)
    x_used, y_used, z_used = float(row["x"]), float(row["y"]), float(row["z"])
    return A, (x_used, y_used, z_used)


def solve_currents_for_direction(A: np.ndarray, u_hat, B0=0.001, Imax=None):
    """
    给定目标磁场方向 u_hat 和大小 B0，求解电流
    目标：A i ≈ B0 * u_hat
    
    返回：
      i_star: 求得的电流 (3,)
      B: 实际实现的磁场向量 A @ i_star
      err: 方向/幅值误差的范数
    """
    u = np.asarray(u_hat, float)
    u = u / norm(u)
    b = B0 * u

    # 最小二乘解
    i_star, *_ = np.linalg.lstsq(A, b, rcond=None)

    # 可选：电流限幅
    if Imax is not None:
        i_star = np.clip(i_star, -Imax, Imax)

    B = A @ i_star
    err = float(norm(B - b))
    return i_star, B, err


def central_diff_or_fit(df_t: pd.DataFrame, x0, y0, z0, h, colname, axis):
    """
    在 (x0,y0,z0) 处对给定列做中央差分或局部拟合
    返回该列在给定轴上的空间导数
    """
    # 尝试中央差分
    if axis == "x":
        r_p = df_t[(np.isclose(df_t["y"], y0)) & (np.isclose(df_t["z"], z0)) & (np.isclose(df_t["x"], x0 + h))]
        r_m = df_t[(np.isclose(df_t["y"], y0)) & (np.isclose(df_t["z"], z0)) & (np.isclose(df_t["x"], x0 - h))]
    elif axis == "y":
        r_p = df_t[(np.isclose(df_t["x"], x0)) & (np.isclose(df_t["z"], z0)) & (np.isclose(df_t["y"], y0 + h))]
        r_m = df_t[(np.isclose(df_t["x"], x0)) & (np.isclose(df_t["z"], z0)) & (np.isclose(df_t["y"], y0 - h))]
    else:  # "z"
        r_p = df_t[(np.isclose(df_t["x"], x0)) & (np.isclose(df_t["y"], y0)) & (np.isclose(df_t["z"], z0 + h))]
        r_m = df_t[(np.isclose(df_t["x"], x0)) & (np.isclose(df_t["y"], y0)) & (np.isclose(df_t["z"], z0 - h))]

    if len(r_p)==1 and len(r_m)==1:
        vp = float(r_p.iloc[0][colname])
        vm = float(r_m.iloc[0][colname])
        return (vp - vm) / (2*h)

    # 没有严格的 ±h 点 → 做局部线性拟合
    neigh = df_t.copy()
    neigh["dx"] = neigh["x"] - x0
    neigh["dy"] = neigh["y"] - y0
    neigh["dz"] = neigh["z"] - z0
    neigh["dist"] = np.sqrt(neigh["dx"]**2 + neigh["dy"]**2 + neigh["dz"]**2)
    neigh = neigh.nsmallest(27, "dist")  # 取 27 个最近邻点
    X = np.stack([np.ones(len(neigh)), neigh["dx"], neigh["dy"], neigh["dz"]], axis=1)
    yv = np.asarray(neigh[colname], dtype=float)
    beta, *_ = np.linalg.lstsq(X, yv, rcond=None)  # beta = [c0, cx, cy, cz]
    if axis == "x":
        return float(beta[1])
    elif axis == "y":
        return float(beta[2])
    else:
        return float(beta[3])


def compute_grad_absB(df: pd.DataFrame, x: float, y: float, z: float,
                      u_hat=(1,0,0), B0=0.01, Imax=None, h=1e-3):
    """
    计算位置 (x,y,z) 处磁场梯度 ∇|B|
    
    步骤：
      1) 从数据库获取雅可比 A
      2) 求解电流 i* 使 A i ≈ B0 u_hat
      3) 用 dA/dx,y,z 和 i* 推出 ∇|B|
    
    返回：dict 包含
      x_used, y_used, z_used: 实际使用的网格点
      i_star: 求得的电流
      B: 实际磁场向量
      Bmag: 磁场大小
      grad_absB: ∇|B| (3,)
      err: 电流求解误差
    """
    # 从数据库获取雅可比
    df_t = df[df["theta"] == 45.0].copy()
    if df_t.empty:
        raise ValueError("No data for theta=45.0 in database")

    # 找最近点
    d2 = (df_t["x"]-x)**2 + (df_t["y"]-y)**2 + (df_t["z"]-z)**2
    row = df_t.loc[d2.idxmin()]
    A = row_to_A(row)
    x0, y0, z0 = float(row["x"]), float(row["y"]), float(row["z"])

    # 求电流
    i_star, B, err = solve_currents_for_direction(A, u_hat, B0=B0, Imax=Imax)
    Bmag = float(norm(B))
    if Bmag < 1e-12:
        raise RuntimeError("|B| too small at this point")

    # 估计 dA/dx, dA/dy, dA/dz
    a_cols = ["A11","A12","A13","A21","A22","A23","A31","A32","A33"]
    dA_dx = np.zeros((3,3))
    dA_dy = np.zeros((3,3))
    dA_dz = np.zeros((3,3))
    for idx, c in enumerate(a_cols):
        gx = central_diff_or_fit(df_t, x0, y0, z0, h, c, "x")
        gy = central_diff_or_fit(df_t, x0, y0, z0, h, c, "y")
        gz = central_diff_or_fit(df_t, x0, y0, z0, h, c, "z")
        dA_dx[idx//3, idx%3] = gx
        dA_dy[idx//3, idx%3] = gy
        dA_dz[idx//3, idx%3] = gz

    # dB/dx,y,z = (dA/dx,y,z) @ i*
    dB_dx = dA_dx @ i_star
    dB_dy = dA_dy @ i_star
    dB_dz = dA_dz @ i_star
    J = np.column_stack([dB_dx, dB_dy, dB_dz])  # 3x3 梯度矩阵

    # ∇|B| = (J^T B)/|B|
    grad_absB = (J.T @ B) / Bmag

    return {
        "x_used": x0, "y_used": y0, "z_used": z0,
        "i_star": i_star, "B": B, "Bmag": Bmag, "err": err,
        "grad_absB": grad_absB,
        "J": J  # 保存梯度矩阵便于后续使用
    }


def compute_grad_absB_fast(precomp: PrecomputedField, x: float, y: float, z: float,
                           u_hat=(1,0,0), B0=0.01, Imax=None, h=1e-3):
    """
    Fast version using PrecomputedField.
    """
    idx = precomp.nearest_index(x, y, z)
    A = precomp.A[idx]
    x0, y0, z0 = precomp.coords[idx]

    i_star, B, err = solve_currents_for_direction(A, u_hat, B0=B0, Imax=Imax)
    Bmag = float(norm(B))
    if Bmag < 1e-12:
        raise RuntimeError("|B| too small at this point")

    dA_dx, dA_dy, dA_dz = precomp.get_dA(idx, h)
    dB_dx = dA_dx @ i_star
    dB_dy = dA_dy @ i_star
    dB_dz = dA_dz @ i_star
    J = np.column_stack([dB_dx, dB_dy, dB_dz])
    grad_absB = (J.T @ B) / Bmag

    return {
        "x_used": float(x0), "y_used": float(y0), "z_used": float(z0),
        "i_star": i_star, "B": B, "Bmag": Bmag, "err": err,
        "grad_absB": grad_absB,
        "J": J
    }


def compute_average_force_rotating_dipole(df_or_precomp, x: float, y: float, z: float,
                                          axis_hat=(0,0,1), B0=0.01, magnetic_moment=1e-6,
                                          n_phase=36, h=1e-3):
    """
    计算被控机器人在给定位置旋转一圈时受到的平均力 
    
    参数：
      df: 离线数据库 (DataFrame)
      x, y, z: 作用位置 (m)
      axis_hat: 旋转轴方向（单位向量或会被单位化）
      B0: 磁场大小 (Tesla)
      magnetic_moment: 磁矩大小 |m| (A·m²)
      n_phase: 旋转周期内的相位采样数
      h: 梯度估计的空间步长 (m)
    
    返回：dict 包含
      F_mean: 旋转平均力 (3,)
      F_mag_mean: 力的大小平均值 (标量)
      F_dc_axis: 沿旋转轴的直流力分量
      F_rms_perp: 正交于旋转轴的RMS力
      F_list: 所有相位的力向量 (n_phase, 3)
      phis: 相位角数组 (n_phase,)
      details: 包含更多细节的dict
    """
    a = np.asarray(axis_hat, float)
    a = a / norm(a)  # 单位化旋转轴

    # 构造旋转平面的两个正交基 e1, e2
    if np.allclose(a, [0,0,1.0]):
        e1 = np.array([1.0, 0.0, 0.0])
        e2 = np.array([0.0, 1.0, 0.0])
    else:
        tmp = np.array([1.0, 0.0, 0.0])
        if abs(np.dot(tmp, a)) > 0.9:
            tmp = np.array([0.0, 1.0, 0.0])
        e1 = tmp - np.dot(tmp, a) * a
        e1 /= norm(e1)
        e2 = np.cross(a, e1)

    # 采样旋转周期
    phis = np.linspace(0, 2*np.pi, n_phase, endpoint=False)
    F_list = []
    B_list = []
    grad_list = []

    for phi in phis:
        # 目标磁场方向（在旋转平面内）
        u_hat = np.cos(phi)*e1 + np.sin(phi)*e2

        # 计算该相位的梯度和电流
        if isinstance(df_or_precomp, PrecomputedField):
            res = compute_grad_absB_fast(df_or_precomp, x, y, z, u_hat=u_hat, B0=B0, h=h)
        else:
            res = compute_grad_absB(df_or_precomp, x, y, z, u_hat=u_hat, B0=B0, h=h)
        grad_absB = res["grad_absB"]

        # 力 = 磁矩大小 × 梯度
        # F = |m| ∇|B|（在锁相假设下，m // B，所以 F = ∇(m·B) = m·∇|B|）
        F = magnetic_moment * grad_absB
        F_list.append(F)
        B_list.append(res["B"])
        grad_list.append(grad_absB)

    F_list = np.vstack(F_list)  # (n_phase, 3)
    B_list = np.vstack(B_list)
    grad_list = np.vstack(grad_list)

    # 计算旋转平均力
    F_mean = np.mean(F_list, axis=0)  # (3,)
    F_mag = norm(F_list, axis=1)  # (n_phase,)
    F_mag_mean = np.mean(F_mag)

    # 分解为轴向和垂直分量
    F_axis_all = F_list @ a  # (n_phase,)
    F_dc_axis = np.mean(F_axis_all)  # 直流分量
    F_perp = F_list - np.outer(F_axis_all, a)  # (n_phase, 3)
    F_rms_perp = np.sqrt(np.mean(np.sum(F_perp**2, axis=1)))

    return {
        "F_mean": F_mean,
        "F_mag_mean": F_mag_mean,
        "F_dc_axis": F_dc_axis,
        "F_rms_perp": F_rms_perp,
        "F_list": F_list,
        "phis": phis,
        "axis_hat": a,
        "B_list": B_list,
        "grad_list": grad_list,
        "details": {
            "x": x, "y": y, "z": z,
            "B0": B0,
            "magnetic_moment": magnetic_moment,
            "n_phase": n_phase,
            "h": h,
            "theta": 45.0
        }
    }


def make_cached_force_callback(df_or_precomp, grid_resolution: float = 1e-4) -> Callable:
    """
    返回一个回调函数 `cb(state, **kwargs)`，它会把位置按 `grid_resolution` 四舍五入并使用 LRU 缓存
    来避免重复计算旋转平均力。

    用法示例：
      cb = make_cached_force_callback(df_or_precomp, grid_resolution=1e-4)
      res = cb(state, axis_hat=(0,0,1), B0=0.01, magnetic_moment=1e-6)

    注意：该回调会接受与 `compute_average_force_rotating_dipole` 相同的关键字参数，
    但 `df_or_precomp` 会闭包在工厂函数内部，因此不用每次传入。
    """

    def quantize_coord(v: float) -> float:
        return round(v / grid_resolution) * grid_resolution

    @lru_cache(maxsize=4096)
    def _compute_cached(xq: float, yq: float, zq: float,
                        axis_hat: Tuple[float, float, float],
                        B0: float, magnetic_moment: float, n_phase: int, h: float):
        # call the main function with quantized coords
        return compute_average_force_rotating_dipole(df_or_precomp, xq, yq, zq,
                                                    axis_hat=axis_hat,
                                                    B0=B0,
                                                    magnetic_moment=magnetic_moment,
                                                    n_phase=n_phase,
                                                    h=h)

    def cb(state, **kwargs):
        x, y, z = float(state[0]), float(state[1]), float(state[2])
        xq, yq, zq = quantize_coord(x), quantize_coord(y), quantize_coord(z)

        # extract kwargs with defaults matching compute_average_force_rotating_dipole
        axis_hat = kwargs.get('axis_hat', (0,0,1))
        B0 = kwargs.get('B0', 0.01)
        magnetic_moment = kwargs.get('magnetic_moment', 1e-6)
        n_phase = int(kwargs.get('n_phase', 36))
        h = float(kwargs.get('h', 1e-3))

        return _compute_cached(xq, yq, zq, tuple(axis_hat), float(B0), float(magnetic_moment), n_phase, h)

    # expose cache_clear for user control
    cb.cache_clear = _compute_cached.cache_clear  # type: ignore
    return cb


# ====== 用法示例 ======
if __name__ == "__main__":
    # 加载 45° 的离线数据库
    df = pd.read_parquet(PARQUET_45DEG)
    print(f"✓ 加载数据库，共 {len(df)} 条记录")
    print(f"  位置范围：x={df['x'].min():.4f}~{df['x'].max():.4f}")
    print(f"  位置范围：y={df['y'].min():.4f}~{df['y'].max():.4f}")
    print(f"  位置范围：z={df['z'].min():.4f}~{df['z'].max():.4f}")

    # 示例参数
    x, y, z = 0.0, 0.0, -0.19  # 作用位置
    axis_hat = np.array([0, 0, 1])  # Z 轴旋转
    B0 = 0.01  # 磁场大小 10 mT
    magnetic_moment = 1e-6  # 磁矩 1 μA·m²
    n_phase = 36  # 采样 36 个相位（每 10° 一个）

    print("\n" + "="*60)
    print("计算旋转平均力")
    print("="*60)
    print(f"位置: ({x}, {y}, {z}) m")
    print(f"旋转轴: {axis_hat} (单位化后)")
    print(f"磁场大小: {B0*1e3:.1f} mT")
    print(f"磁矩大小: {magnetic_moment*1e6:.3f} μA·m²")
    print(f"采样相位数: {n_phase}")

    result = compute_average_force_rotating_dipole(
        df, x, y, z,
        axis_hat=axis_hat,
        B0=B0,
        magnetic_moment=magnetic_moment,
        n_phase=n_phase,
        h=1e-3
    )

    print("\n结果:")
    print(f"  平均力向量: {result['F_mean']} N")
    print(f"  平均力大小: {result['F_mag_mean']:.6e} N")
    print(f"  轴向直流力: {result['F_dc_axis']:.6e} N")
    print(f"  垂直RMS力: {result['F_rms_perp']:.6e} N")
    print(f"\n详细信息:")
    for key, val in result['details'].items():
        print(f"  {key}: {val}")

    # 可视化力随相位的变化
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    # 力的三个分量随相位的变化
    ax = axes[0, 0]
    ax.plot(result['phis']*180/np.pi, result['F_list'][:, 0], label='Fx', marker='.')
    ax.plot(result['phis']*180/np.pi, result['F_list'][:, 1], label='Fy', marker='.')
    ax.plot(result['phis']*180/np.pi, result['F_list'][:, 2], label='Fz', marker='.')
    ax.axhline(result['F_mean'][0], color='C0', linestyle='--', alpha=0.5)
    ax.axhline(result['F_mean'][1], color='C1', linestyle='--', alpha=0.5)
    ax.axhline(result['F_mean'][2], color='C2', linestyle='--', alpha=0.5)
    ax.set_xlabel('Phase (deg)')
    ax.set_ylabel('Force (N)')
    ax.set_title('Force Components vs Rotation Phase')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 力的大小
    ax = axes[0, 1]
    F_mag = norm(result['F_list'], axis=1)
    ax.plot(result['phis']*180/np.pi, F_mag, marker='o', label='|F|')
    ax.axhline(result['F_mag_mean'], color='r', linestyle='--', label=f'Mean={result["F_mag_mean"]:.3e}')
    ax.set_xlabel('Phase (deg)')
    ax.set_ylabel('|F| (N)')
    ax.set_title('Force Magnitude vs Phase')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 轴向和垂直分量
    ax = axes[1, 0]
    F_axis_all = result['F_list'] @ result['axis_hat']
    ax.plot(result['phis']*180/np.pi, F_axis_all, marker='s', label='F_axis')
    ax.axhline(result['F_dc_axis'], color='r', linestyle='--', label=f'DC={result["F_dc_axis"]:.3e}')
    ax.set_xlabel('Phase (deg)')
    ax.set_ylabel('F_axis (N)')
    ax.set_title('Axial Force Component')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 垂直力RMS
    ax = axes[1, 1]
    F_perp = result['F_list'] - np.outer(F_axis_all, result['axis_hat'])
    F_perp_mag = norm(F_perp, axis=1)
    ax.plot(result['phis']*180/np.pi, F_perp_mag, marker='^', label='|F_perp|')
    ax.axhline(result['F_rms_perp'], color='r', linestyle='--', label=f'RMS={result["F_rms_perp"]:.3e}')
    ax.set_xlabel('Phase (deg)')
    ax.set_ylabel('|F_perp| (N)')
    ax.set_title('Perpendicular Force Component')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('rotating_force_analysis.png', dpi=150)
    print("\n✓ 已保存力分析图到 rotating_force_analysis.png")
    plt.show()
