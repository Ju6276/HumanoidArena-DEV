# Copyright (c) 2025. All Rights Reserved.
# License: Apache-2.0
"""Pure NumPy SONIC geometry, extracted unchanged from the HA provider."""
import numpy as np

def quat_to_rotation_6d(quat: np.ndarray) -> np.ndarray:
    """
    将四元数转换为6D旋转表示（旋转矩阵的前2列，按行展开）

    这是SONIC encoder期望的输入格式，与C++实现保持一致。
    参考: gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/src/g1_deploy_onnx_ref.cpp:514-580

    Args:
        quat: (..., 4) 四元数 [w, x, y, z]

    Returns:
        rot6d: (..., 6) 6D旋转表示 [R00, R01, R10, R11, R20, R21]
    """
    w, x, y, z = (quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3])
    r00 = 1 - 2 * (y * y + z * z)
    r10 = 2 * (x * y + w * z)
    r20 = 2 * (x * z - w * y)
    r01 = 2 * (x * y - w * z)
    r11 = 1 - 2 * (x * x + z * z)
    r21 = 2 * (y * z + w * x)
    rot6d = np.stack([r00, r01, r10, r11, r20, r21], axis=-1)
    return rot6d.astype(np.float32)

def quat_normalize_wxyz(quat: np.ndarray) -> np.ndarray:
    """Normalize quaternion in wxyz format."""
    quat = np.asarray(quat, dtype=np.float32)
    norm = np.linalg.norm(quat, axis=-1, keepdims=True)
    return (quat / np.clip(norm, 1e-12, None)).astype(np.float32)

def quat_conjugate_wxyz(quat: np.ndarray) -> np.ndarray:
    """Quaternion conjugate in wxyz format."""
    quat = np.asarray(quat, dtype=np.float32)
    out = quat.copy()
    out[..., 1:] *= -1.0
    return out.astype(np.float32)

def quat_mul_wxyz(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Quaternion multiply in wxyz format."""
    q1 = np.asarray(q1, dtype=np.float32)
    q2 = np.asarray(q2, dtype=np.float32)
    w1, x1, y1, z1 = (q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3])
    w2, x2, y2, z2 = (q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3])
    out = np.stack([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2, w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2], axis=-1)
    return quat_normalize_wxyz(out)

def compute_anchor_rot6d_wxyz(base_quat_wxyz: np.ndarray, ref_quat_wxyz: np.ndarray, heading_align_quat_wxyz: np.ndarray, use_heading_align: bool) -> np.ndarray:
    """Match SONIC deploy: compute base-relative anchor from current base and ref root quats."""
    base_quat_wxyz = quat_normalize_wxyz(base_quat_wxyz)
    ref_quat_wxyz = quat_normalize_wxyz(ref_quat_wxyz)
    aligned_ref_quat_wxyz = ref_quat_wxyz.copy()
    if use_heading_align:
        aligned_ref_quat_wxyz = quat_mul_wxyz(heading_align_quat_wxyz, ref_quat_wxyz)
    rel_quat_wxyz = quat_mul_wxyz(quat_conjugate_wxyz(base_quat_wxyz), aligned_ref_quat_wxyz)
    return quat_to_rotation_6d(rel_quat_wxyz.reshape(1, 4))[0].astype(np.float32)

def gravity_dir_from_base_quat_wxyz(quat: np.ndarray) -> np.ndarray:
    """Match SONIC deploy: gravity_dir = quat_conjugate(base_quat) rotate [0, 0, -1]."""
    quat = quat_normalize_wxyz(quat)
    gravity_world = np.array([0.0, 0.0, -1.0], dtype=np.float32)
    gravity_quat = np.array([0.0, gravity_world[0], gravity_world[1], gravity_world[2]], dtype=np.float32)
    rotated = quat_mul_wxyz(quat_mul_wxyz(quat_conjugate_wxyz(quat), gravity_quat), quat)
    return rotated[1:].astype(np.float32)