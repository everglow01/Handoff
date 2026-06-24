"""ORB 特征检测与关键点序列化。

集中定义 ORB 检测器，保证标定阶段（step 2）与估 H 阶段（step 3）
用完全相同的参数，避免两处不一致导致匹配退化。
cv2.KeyPoint 无法直接存进 npz，这里提供与数组的互转。
"""

from __future__ import annotations

import cv2
import numpy as np

# 关键点打包成 Nx7 数组的列顺序
_KP_COLS = ("x", "y", "size", "angle", "response", "octave", "class_id")


def make_orb(nfeatures: int = 2000) -> cv2.ORB:
    """创建 ORB 检测器。step 2/3 统一从这里取，参数只在此处定义一次。"""
    return cv2.ORB_create(nfeatures=nfeatures)


def to_gray(image: np.ndarray) -> np.ndarray:
    """转灰度供 ORB 使用；已是单通道则原样返回。"""
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def detect(orb: cv2.ORB, image: np.ndarray) -> tuple[list[cv2.KeyPoint], np.ndarray]:
    """在图像上检测 ORB 特征，返回 (keypoints, descriptors)。"""
    gray = to_gray(image)
    return orb.detectAndCompute(gray, None)


def keypoints_to_array(keypoints: list[cv2.KeyPoint]) -> np.ndarray:
    """关键点列表 -> Nx7 float32 数组（可存入 npz）。"""
    if not keypoints:
        return np.empty((0, len(_KP_COLS)), dtype=np.float32)
    return np.array(
        [
            (k.pt[0], k.pt[1], k.size, k.angle, k.response, k.octave, k.class_id)
            for k in keypoints
        ],
        dtype=np.float32,
    )


def keypoints_from_array(arr: np.ndarray) -> list[cv2.KeyPoint]:
    """Nx7 数组 -> 关键点列表（逆向还原）。"""
    return [
        cv2.KeyPoint(
            x=float(r[0]),
            y=float(r[1]),
            size=float(r[2]),
            angle=float(r[3]),
            response=float(r[4]),
            octave=int(r[5]),
            class_id=int(r[6]),
        )
        for r in arr
    ]
