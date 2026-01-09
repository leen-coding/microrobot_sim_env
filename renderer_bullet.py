# renderer_bullet.py
import numpy as np
import pybullet as p
import pybullet_data
from pathlib import Path
from utils import quat_mul, quat_from_axis_angle, quat_from_two_vectors, _normalize

class BulletRenderer:
    """
    Rendering only. It never updates dynamics.
    state is interpreted as the task/reference point position (your choice).
    This renderer converts it to COM position using r_LI if needed.
    """
    def __init__(
        self,
        urdf_path,
        body_axis=np.array([1.0,0.0,0.0]),
        r_LI=np.array([0.0,0.0,0.0]),
        use_gui=True,
        camera_distance=0.08,
        camera_yaw=40,
        camera_pitch=-30,
        camera_target=(0,0,0),
        gravity=(0,0,0),
        use_inertia_from_file=True,
        ignore_collision=True,
    ):
        self.body_axis = np.array(body_axis, dtype=float)
        self.r_LI = np.array(r_LI, dtype=float)
        self.use_gui = use_gui

        cid = p.connect(p.GUI if use_gui else p.DIRECT)
        p.resetSimulation()
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.loadURDF("plane.urdf")
        p.setGravity(*gravity)

        urdf_path = str(urdf_path)
        urdf_dir = str(Path(urdf_path).parent)
        p.setAdditionalSearchPath(urdf_dir)

        flags = 0
        if use_inertia_from_file:
            flags |= p.URDF_USE_INERTIA_FROM_FILE
        if ignore_collision:
            flags |= p.URDF_IGNORE_COLLISION_SHAPES

        self.robot_id = p.loadURDF(
            urdf_path,
            basePosition=[0,0,0],
            baseOrientation=[0,0,0,1],
            useFixedBase=True,
            flags=flags
        )

        if use_gui:
            p.resetDebugVisualizerCamera(
                cameraDistance=camera_distance,
                cameraYaw=camera_yaw,
                cameraPitch=camera_pitch,
                cameraTargetPosition=list(camera_target)
            )

    def update(self, state_xyz, k_hat, phi_spin):
        # orientation from body_axis -> k_hat
        k_hat, kn = _normalize(np.array(k_hat, dtype=float))
        if kn < 1e-12:
            k_hat = np.array([0.0,0.0,1.0])

        q_align = quat_from_two_vectors(self.body_axis, k_hat)
        q_spin  = quat_from_axis_angle(k_hat, phi_spin)
        q = quat_mul(q_spin, q_align)

        R = np.array(p.getMatrixFromQuaternion(q.tolist()), dtype=float).reshape(3,3)
        pos_com = np.array(state_xyz, dtype=float) + R @ self.r_LI

        p.resetBasePositionAndOrientation(self.robot_id, pos_com.tolist(), q.tolist())
        p.stepSimulation()

    def close(self):
        if p.isConnected():
            p.disconnect()
