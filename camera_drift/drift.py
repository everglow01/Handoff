"""偏移量检测与告警（需求1）。

用 H 把参考帧四角投影到当前帧，量它们移动了多少像素：
位移超阈值 => 「偏离标准位」告警。
H 估计失败（匹配/内点不足、画面变化过大）则单独告警人工介入——不硬套 H。
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .homography import HomographyResult


@dataclass
class DriftReport:
    ok: bool                 # H 是否可用
    max_disp_px: float       # 四角中最大位移
    mean_disp_px: float      # 四角平均位移
    alarm: bool              # 是否告警
    reason: str              # 告警/状态原因


def corner_displacements(image_shape: tuple[int, int], H: np.ndarray) -> np.ndarray:
    """参考帧四角经 H 投影后相对原位置的位移（像素）。返回长度4数组。"""
    h, w = image_shape[:2]
    corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
    moved = cv2.perspectiveTransform(corners, H).reshape(-1, 2)
    return np.linalg.norm(moved - corners.reshape(-1, 2), axis=1)


def evaluate(
    result: HomographyResult,
    image_shape: tuple[int, int],
    drift_thresh_px: float = 15.0,
    min_inliers: int = 30,
) -> DriftReport:
    """综合 H 估计结果给出偏移报告与告警决策。"""
    # 兜底：H 失败或内点过少 -> 告警人工介入，不算位移
    if not result.ok or result.H is None:
        return DriftReport(False, float("nan"), float("nan"), True,
                           "H 估计失败：画面变化过大/纹理不足，需人工介入")
    if result.num_inliers < min_inliers:
        return DriftReport(False, float("nan"), float("nan"), True,
                           f"内点过少({result.num_inliers}<{min_inliers})：匹配不可靠，需人工介入")

    disp = corner_displacements(image_shape, result.H)
    max_d, mean_d = float(disp.max()), float(disp.mean())
    if max_d > drift_thresh_px:
        return DriftReport(True, max_d, mean_d, True,
                           f"偏离标准位：最大角位移 {max_d:.1f}px > {drift_thresh_px}px")
    return DriftReport(True, max_d, mean_d, False, "正常")
