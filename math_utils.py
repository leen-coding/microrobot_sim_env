import numpy as np


def _normalize(v, eps=1e-12):
    n = np.linalg.norm(v)
    if n < eps:
        return v * 0.0, 0.0
    return v / n, n


def quat_mul(q1, q2):
    # q = q1 ⊗ q2, both in (x,y,z,w)
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return np.array(
        [
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ],
        dtype=float,
    )


def quat_from_axis_angle(axis, angle):
    axis, n = _normalize(axis)
    if n < 1e-12:
        return np.array([0, 0, 0, 1.0], dtype=float)
    s = np.sin(angle / 2.0)
    return np.array(
        [axis[0] * s, axis[1] * s, axis[2] * s, np.cos(angle / 2.0)],
        dtype=float,
    )


def quat_from_two_vectors(a, b):
    """
    quaternion that rotates unit vector a to unit vector b. returns (x,y,z,w)
    """
    a, na = _normalize(a)
    b, nb = _normalize(b)
    if na < 1e-12 or nb < 1e-12:
        return np.array([0, 0, 0, 1.0], dtype=float)

    v = np.cross(a, b)
    c = float(np.dot(a, b))

    if c < -0.999999:
        # 180 deg: pick an orthogonal axis
        axis = np.array([1.0, 0.0, 0.0])
        if abs(a[0]) > 0.9:
            axis = np.array([0.0, 1.0, 0.0])
        axis = axis - a * np.dot(a, axis)
        axis, _ = _normalize(axis)
        return np.array([axis[0], axis[1], axis[2], 0.0], dtype=float)

    s = np.sqrt((1.0 + c) * 2.0)
    q = np.array([v[0] / s, v[1] / s, v[2] / s, 0.5 * s], dtype=float)
    # normalize quaternion
    q = q / (np.linalg.norm(q) + 1e-12)
    return q
