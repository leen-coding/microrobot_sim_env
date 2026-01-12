from env import MicroRobotEnv
from renderer_bullet import BulletRenderer
from pathlib import Path
import numpy as np
import time
from dynamics import MicroRobotParams
from external_force_model import build_force_model_from_params
base_dir = Path(__file__).resolve().parent
urdf_path = base_dir / "robot_model" / "robot.urdf"

renderer = BulletRenderer(
    urdf_path=urdf_path,
    body_axis=np.array([1,0,0]),                 # 你已确认螺旋轴为 STL 的 x
    r_LI=np.array([0.01067,-0.00298,-0.00001]),  # URDF inertial origin（米）
    use_gui=True,
    gravity=(0,0,0),                              # 仅渲染，建议 0
)

# conservative params with external force model
params = MicroRobotParams()
parquet_path = base_dir / "actuation_matrices_45deg.parquet"
if parquet_path.exists():
    params.ext.force_fn = build_force_model_from_params(params, parquet_path)

env = MicroRobotEnv(dt=0.02, renderer=renderer, params=params)
env.reset()

for t in range(2000):
    ang = 0.008*t
    k = np.array([np.cos(ang), np.sin(ang), 0.2])
    action = [k[0], k[1], k[2], 7.0]
    state, info = env.step(action)
    if renderer is not None:
        renderer.update(
            state_xyz=state,
            k_hat=info.get("k_hat", [0, 0, 1]),
            phi_spin=env.phi_spin,
        )
    # print(info.get("F_ext"))
    # print(state)
    time.sleep(1/60)

env.close()
