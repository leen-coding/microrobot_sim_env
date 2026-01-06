import pybullet as p
import pybullet_data
import time
from pathlib import Path

base_dir = Path(__file__).resolve().parent
urdf_path = str(base_dir / "robot_model" / "robot.urdf")
urdf_dir  = str(Path(urdf_path).parent)

cid = p.connect(p.GUI)
p.resetSimulation()

# 可选：加载地面，设置重力（渲染用也无妨）
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.loadURDF("plane.urdf")
p.setGravity(0, 0, 0)

# 让 PyBullet 能找到 robot_model/ 下的 mesh 相对路径
p.setAdditionalSearchPath(urdf_dir)

robot_id = p.loadURDF(
    urdf_path,
    basePosition=[0, 0, 0],
    baseOrientation=[0, 0, 0, 1],
    useFixedBase=True,
    flags=p.URDF_USE_INERTIA_FROM_FILE
)

# 可选：设置相机，避免看不到模型
p.resetDebugVisualizerCamera(
    cameraDistance=0.08,
    cameraYaw=40,
    cameraPitch=-30,
    cameraTargetPosition=[0, 0, 0]
)

# 关键：保持窗口不退出
try:
    while p.isConnected():
        p.stepSimulation()
        time.sleep(1/240)  # GUI 刷新节奏
except KeyboardInterrupt:
    pass
finally:
    p.disconnect()
