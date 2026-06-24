"""基于强度的 ECC 单应配准——低纹理/强光照变化时的降级路线。

ORB 在大白墙/纯地面或昼夜光照剧变时会失配。ECC（增强相关系数）不依赖
特征点，且其判据本身对亮度/对比度的仿射变化天然不变——这正是它抗光照的来源，
所以直接喂灰度图即可（比喂稀疏的梯度图更稳定，后者会让 ECC 的高斯牛顿迭代发散）。
强制 MOTION_HOMOGRAPHY（8-DoF），绝不用 EUCLIDEAN/AFFINE——后者无法表达
pan/tilt 的透视畸变。

返回的 warp 约定与 ORB 路径一致：H 把「参考帧坐标 -> 当前帧坐标」。
"""

from __future__ import annotations

import cv2
import numpy as np

from .features import to_gray


def _prep(image: np.ndarray) -> np.ndarray:
    """转灰度 float32 供 ECC 使用（ECC 内部 gaussFiltSize 再做平滑）。"""
    return to_gray(image).astype(np.float32)


def estimate_ecc(
    ref_frame: np.ndarray,
    frame: np.ndarray,
    init_H: np.ndarray | None = None,
    iterations: int = 100,
    eps: float = 1e-5,
    levels: int = 3,
) -> tuple[np.ndarray | None, float]:
    """金字塔 ECC 估计 H（ref -> current）。返回 (H, cc)；失败返回 (None, -1)。

    cc 为相关系数[-1,1]，越接近 1 越可信，作为该路径的置信度。
    init_H 用 ORB 的粗结果做初值能进一步提升收敛（没有则从单位阵起步）。
    多尺度（粗->细）大幅扩大收敛域，否则单尺度只能纠正几像素级错位。
    """
    # 各层灰度金字塔（顶层最粗）
    pyr_ref = [_prep(ref_frame)]
    pyr_cur = [_prep(frame)]
    for _ in range(levels - 1):
        pyr_ref.append(cv2.pyrDown(pyr_ref[-1]))
        pyr_cur.append(cv2.pyrDown(pyr_cur[-1]))

    H = (np.asarray(init_H, dtype=np.float64).copy()
         if init_H is not None else np.eye(3, dtype=np.float64))
    criteria = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, iterations, eps)

    cc = -1.0
    any_ok = False
    for lvl in range(levels - 1, -1, -1):  # 从最粗层到原分辨率
        s = 0.5 ** lvl
        S = np.diag([s, s, 1.0])
        W = (S @ H @ np.linalg.inv(S)).astype(np.float32)  # 把 H 缩放到当前层
        try:
            cc, W = cv2.findTransformECC(
                pyr_ref[lvl], pyr_cur[lvl], W, cv2.MOTION_HOMOGRAPHY, criteria, None, 5
            )
        except cv2.error:
            # 某层（常是过平滑的粗层）无法收敛：保持当前 H，继续更细层
            continue
        H = np.linalg.inv(S) @ W.astype(np.float64) @ S  # 还原回原分辨率
        any_ok = True
    return (H, float(cc)) if any_ok else (None, -1.0)
