"""估计单应矩阵 H，并用它投影多边形。

这是整个项目的核心机制：当前帧与参考帧之间的映射严格是单应 H
（相机绕光心旋转、与场景深度无关）。估出 H 后，
  - 投影多边形 -> 需求3（多边形自适应）
  - 看位移/分解 H -> 需求1/2（在 step 4 实现）
本模块只负责「估 H + 投影多边形」，置信度门控与时序平滑留给 step 5。
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from . import features
from .calibration import Calibration


@dataclass
class HomographyResult:
    """一次 H 估计的结果。"""

    H: np.ndarray | None       # 3x3 单应（ref -> current），失败为 None
    num_good: int              # 比率测试后的好匹配数
    num_inliers: int           # RANSAC 内点数
    ok: bool                   # 是否成功估出 H（基础可用性，非置信度门控）
    reproj_err: float = float("nan")  # 内点平均重投影误差(px)，供置信度门控用

    @property
    def inlier_ratio(self) -> float:
        return self.num_inliers / self.num_good if self.num_good else 0.0


def estimate_homography(
    calib: Calibration,
    frame: np.ndarray,
    orb: cv2.ORB | None = None,
    ratio: float = 0.75,
    ransac_thresh: float = 3.0,
    min_matches: int = 4,
) -> HomographyResult:
    """匹配当前帧与参考帧特征，RANSAC 估出 H（ref -> current）。

    动态前景（人/车）会被 RANSAC 当外点剔除，只要静态背景特征够。
    匹配/内点不足 -> ok=False（交由调用方走告警兜底，不硬套 H）。
    """
    if orb is None:
        orb = features.make_orb(calib.nfeatures)

    # 低纹理参考帧可能没有特征 -> ORB 路径直接判失败，交由上层走 ECC
    if calib.descriptors is None or len(calib.descriptors) < min_matches:
        return HomographyResult(None, 0, 0, ok=False)

    kp_cur, des_cur = features.detect(orb, frame)
    if des_cur is None or len(kp_cur) < min_matches:
        return HomographyResult(None, 0, 0, ok=False)

    # 比率测试筛好匹配（ref 为 query，current 为 train）
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    knn = matcher.knnMatch(calib.descriptors, des_cur, k=2)
    good = [m for pair in knn if len(pair) == 2 for m, n in [pair] if m.distance < ratio * n.distance]

    if len(good) < min_matches:
        return HomographyResult(None, len(good), 0, ok=False)

    src = np.float32([calib.keypoints[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([kp_cur[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, ransac_thresh)

    if H is None:
        return HomographyResult(None, len(good), 0, ok=False)

    inliers = int(mask.sum()) if mask is not None else 0

    # 内点重投影误差：把 ref 内点经 H 投到 current，与真实匹配点比
    reproj = float("nan")
    if mask is not None and inliers > 0:
        m = mask.ravel().astype(bool)
        proj = cv2.perspectiveTransform(src[m], H)
        reproj = float(np.linalg.norm(proj - dst[m], axis=2).mean())

    return HomographyResult(H=H, num_good=len(good), num_inliers=inliers, ok=True, reproj_err=reproj)


def project_polygon(polygon: np.ndarray, H: np.ndarray) -> np.ndarray:
    """用 H 把参考帧多边形投影到当前帧坐标系。返回 Kx2 float32。"""
    poly = np.asarray(polygon, dtype=np.float32).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(poly, H).reshape(-1, 2)
