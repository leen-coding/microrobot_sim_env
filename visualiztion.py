from env import MicroRobotEnv
from renderer_bullet import BulletRenderer
from pathlib import Path
import numpy as np
import time
import pandas as pd
from compute_rotating_average_force import make_cached_force_callback, build_precomputed_field
from dynamics import MicroRobotParams
base_dir = Path(__file__).resolve().parent
urdf_path = base_dir / "robot_model" / "robot.urdf"

renderer = BulletRenderer(
    urdf_path=urdf_path,
    body_axis=np.array([1,0,0]),                 # 你已确认螺旋轴为 STL 的 x
    r_LI=np.array([0.01067,-0.00298,-0.00001]),  # URDF inertial origin（米）
    use_gui=True,
    gravity=(0,0,0),                              # 仅渲染，建议 0
)

# --- prepare parameters with external force callback (方案 A) ---
# load actuation matrices parquet (must exist in workspace root)
parquet_path = base_dir / "actuation_matrices_45deg.parquet"
if parquet_path.exists():
    df = pd.read_parquet(parquet_path)
    precomp = build_precomputed_field(df, theta=45.0)
    cb = make_cached_force_callback(precomp, grid_resolution=1e-4)
else:
    df = None
    cb = None

# conservative params with callback attached
params = MicroRobotParams()
if cb is not None:
    # wrap cached callback with a cheap throttle: re-use last result while
    # the robot stays within the same quantized grid cell to avoid repeated
    # cache overhead for nearby positions
    grid_th = 1e-4
    def make_throttled(cb_fn, grid_th):
        state_last = {"key": None, "res": None}
        def throttled(state, **kwargs):
            key = (round(state[0]/grid_th), round(state[1]/grid_th), round(state[2]/grid_th))
            if key == state_last["key"] and state_last["res"] is not None:
                return state_last["res"]
            res = cb_fn(state, **kwargs)
            state_last["key"] = key
            state_last["res"] = res
            return res
        return throttled

    params.external_force_callback = make_throttled(cb, grid_th)
    params.external_force_kwargs = {
        # axis_hat will be updated each step to match action k direction
        "axis_hat": (0,0,1),
        "B0": 0.02,                   # 10 mT
        # use a conservative magnetic moment for test (A·m^2)
        "magnetic_moment": 1e-6,
        "n_phase": 18,
        "h": 1e-3,
    }

env = MicroRobotEnv(dt=0.02, renderer=renderer, params=params)
env.reset()

for t in range(2000):
    ang = 0.01*t
    k = np.array([np.cos(ang), np.sin(ang), 0.2])
    action = [k[0], k[1], k[2], 8.0]
    if params.external_force_kwargs is not None:
        kn = np.linalg.norm(k)
        if kn > 1e-12:
            params.external_force_kwargs["axis_hat"] = (k / kn).tolist()
    state, info = env.step(action)
    print(info.get('F_noise'))
    print(state)
    time.sleep(1/60)

env.close()
