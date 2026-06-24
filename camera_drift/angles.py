"""从单应 H 估计相机旋转角 pan / tilt / roll（需求2）。

纯旋转下 H = K·R·K⁻¹。本项目无法用棋盘格标定 K：
  - 焦距 f 从 H 自标定（Hartley 全景思路，假设主点在画面中心、方形像素）；
  - K 已知后 R = K⁻¹·H·K，SVD 正交化后分解欧拉角；
  - roll 不依赖 K，可由 H 直接近似 atan2(H[1,0], H[0,0])，f 估不准时退化用它。

角度约定：R = Rz(roll) · Ry(pan) · Rx(tilt)
  pan  绕 y 轴（左右摇）, tilt 绕 x 轴（上下俯仰）, roll 绕 z 轴（画面内旋转）。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class AngleResult:
    roll_deg: float                 # 总能给（roll 不依赖 f）
    pan_deg: float | None           # 需要 f；估不准为 None
    tilt_deg: float | None
    focal_px: float | None          # 自标定出的焦距；None 表示失败
    roll_simple_deg: float          # 不依赖 K 的退化估计 atan2(H[1,0],H[0,0])


def rotation_matrix(pan_deg: float, tilt_deg: float, roll_deg: float) -> np.ndarray:
    """按 R = Rz(roll)·Ry(pan)·Rx(tilt) 约定构造旋转矩阵（主要供测试/正向构造）。"""
    a, b, c = map(math.radians, (tilt_deg, pan_deg, roll_deg))
    Rx = np.array([[1, 0, 0], [0, math.cos(a), -math.sin(a)], [0, math.sin(a), math.cos(a)]])
    Ry = np.array([[math.cos(b), 0, math.sin(b)], [0, 1, 0], [-math.sin(b), 0, math.cos(b)]])
    Rz = np.array([[math.cos(c), -math.sin(c), 0], [math.sin(c), math.cos(c), 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def _center_homography(H: np.ndarray, cx: float, cy: float) -> np.ndarray:
    """把 H 共轭到「主点在原点」的坐标系：Hc = T⁻¹·H·T。自标定公式要求主点在原点。"""
    T = np.array([[1, 0, cx], [0, 1, cy], [0, 0, 1]], dtype=np.float64)
    return np.linalg.inv(T) @ H @ T


def focal_from_homography(Hc: np.ndarray) -> float | None:
    """从已居中的单应自标定焦距 f（OpenCV focalsFromHomography 思路）。

    纯旋转 Hc = K·R·K⁻¹，K=diag(f,f,1)，利用 R 行/列正交约束解 f²，取两组解的可用者。
    """
    h = Hc.ravel()
    cands: list[float] = []

    # 解 1（由 H 的第三列约束）
    d1 = h[6] * h[7]
    d2 = (h[7] - h[6]) * (h[7] + h[6])
    v1 = -(h[0] * h[1] + h[3] * h[4]) / d1 if d1 != 0 else None
    v2 = (h[0] ** 2 + h[3] ** 2 - h[1] ** 2 - h[4] ** 2) / d2 if d2 != 0 else None
    f = _pick_focal(v1, v2, d1, d2)
    if f is not None:
        cands.append(f)

    # 解 2（由 H 的第三行约束）
    d1 = h[0] * h[3] + h[1] * h[4]
    d2 = h[0] ** 2 + h[1] ** 2 - h[3] ** 2 - h[4] ** 2
    v1 = -h[2] * h[5] / d1 if d1 != 0 else None
    v2 = (h[5] ** 2 - h[2] ** 2) / d2 if d2 != 0 else None
    f = _pick_focal(v1, v2, d1, d2)
    if f is not None:
        cands.append(f)

    if not cands:
        return None
    return float(np.median(cands))


def _pick_focal(v1, v2, d1, d2) -> float | None:
    """从两个 f² 候选里挑一个正值（分母绝对值大的更可信）。"""
    if v1 is not None and v2 is not None:
        if v1 < v2:
            v1, v2, d1, d2 = v2, v1, d2, d1
        if v1 > 0 and v2 > 0:
            return math.sqrt(v1 if abs(d1) > abs(d2) else v2)
        if v1 > 0:
            return math.sqrt(v1)
        return None
    for v in (v1, v2):
        if v is not None and v > 0:
            return math.sqrt(v)
    return None


def _euler_from_R(R: np.ndarray) -> tuple[float, float, float]:
    """R = Rz(roll)·Ry(pan)·Rx(tilt) 的逆向分解，返回 (pan, tilt, roll) 度。"""
    pan = math.atan2(-R[2, 0], math.hypot(R[0, 0], R[1, 0]))
    roll = math.atan2(R[1, 0], R[0, 0])
    tilt = math.atan2(R[2, 1], R[2, 2])
    return math.degrees(pan), math.degrees(tilt), math.degrees(roll)


def estimate_angles(H: np.ndarray, image_shape: tuple[int, int]) -> AngleResult:
    """从 H 估计 pan/tilt/roll。image_shape=(h, w)。"""
    h_img, w_img = image_shape[:2]
    cx, cy = w_img / 2.0, h_img / 2.0

    roll_simple = math.degrees(math.atan2(H[1, 0], H[0, 0]))

    Hc = _center_homography(np.asarray(H, dtype=np.float64), cx, cy)
    f = focal_from_homography(Hc)
    if f is None or f <= 0:
        return AngleResult(roll_simple, None, None, None, roll_simple)

    K = np.diag([f, f, 1.0])
    R = np.linalg.inv(K) @ Hc @ K
    # SVD 正交化，得到最接近的合法旋转矩阵
    U, _, Vt = np.linalg.svd(R)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt

    pan, tilt, roll = _euler_from_R(R)
    return AngleResult(roll, pan, tilt, f, roll_simple)
