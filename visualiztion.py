from env import MicroRobotEnv
from renderer_bullet import BulletRenderer
from pathlib import Path
import numpy as np
import time
base_dir = Path(__file__).resolve().parent
urdf_path = base_dir / "robot_model" / "robot.urdf"

renderer = BulletRenderer(
    urdf_path=urdf_path,
    body_axis=np.array([1,0,0]),                 # 你已确认螺旋轴为 STL 的 x
    r_LI=np.array([0.01067,-0.00298,-0.00001]),  # URDF inertial origin（米）
    use_gui=True,
    gravity=(0,0,0),                              # 仅渲染，建议 0
)

env = MicroRobotEnv(dt=0.02, renderer=renderer)
env.reset()

for t in range(2000):
    ang = 0.01*t
    k = np.array([np.cos(ang), np.sin(ang), 0.1])
    action = [k[0], k[1], k[2], 2.0]
    state, info = env.step(action)
    time.sleep(1/240) 

env.close()