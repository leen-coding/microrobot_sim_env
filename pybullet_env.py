import pybullet as p
import os

urdf_path = "C:\\Users\\Jialin\\Desktop\\microrobot_sim_env\\robot_model\\robot.urdf"
urdf_dir  = os.path.dirname(urdf_path)

p.connect(p.GUI)
p.resetSimulation()

# 让 PyBullet 能找到 robot_model/ 这类相对路径
p.setAdditionalSearchPath(urdf_dir)

robot_id = p.loadURDF(
    urdf_path,
    basePosition=[0,0,0],
    baseOrientation=[0,0,0,1],
    useFixedBase=True,
    flags=p.URDF_USE_INERTIA_FROM_FILE
)
