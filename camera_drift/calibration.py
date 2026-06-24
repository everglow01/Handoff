"""标定数据的生成、保存与读取。

标定阶段（装机时做一次）把以下三样东西算好并缓存：
  - 标准位置参考帧 ref_frame
  - 参考帧的 ORB 特征（点 + 描述子，预先算好，运行时直接用）
  - 多边形坐标（参考帧坐标系下）
运行阶段（step 3）只需 load_calibration 即可拿到这些。
缓存为单个 .npz 文件，不使用 pickle。
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import features


@dataclass
class Calibration:
    """一台摄像头的标定数据。"""

    ref_frame: np.ndarray          # 参考帧（原始，BGR 或灰度）
    keypoints: list                # list[cv2.KeyPoint]
    descriptors: np.ndarray        # NxM uint8（ORB）
    polygon: np.ndarray            # Kx2 float32，参考帧坐标系
    nfeatures: int                 # 生成特征时的 ORB 参数


def calibrate(
    ref_frame: np.ndarray,
    polygon: np.ndarray,
    nfeatures: int = 2000,
) -> Calibration:
    """对参考帧提特征，连同多边形打包成 Calibration。"""
    poly = np.asarray(polygon, dtype=np.float32).reshape(-1, 2)
    if len(poly) < 3:
        raise ValueError(f"多边形至少需要 3 个点，得到 {len(poly)} 个")

    orb = features.make_orb(nfeatures)
    keypoints, descriptors = features.detect(orb, ref_frame)
    if descriptors is None or len(keypoints) == 0:
        # 低纹理（白墙/纯地面）参考帧可能一个角点都没有。不硬失败：
        # 仍存下参考帧，运行时走 ECC（基于强度、无需特征点）降级。
        print("[警告] 参考帧 ORB 特征为 0（纹理过少）：将依赖 ECC 降级", file=sys.stderr)
        keypoints, descriptors = [], np.empty((0, 32), dtype=np.uint8)

    return Calibration(
        ref_frame=ref_frame,
        keypoints=keypoints,
        descriptors=descriptors,
        polygon=poly,
        nfeatures=nfeatures,
    )


def save_calibration(path: str | Path, calib: Calibration) -> Path:
    """保存为 .npz 缓存文件。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        ref_frame=calib.ref_frame,
        keypoints=features.keypoints_to_array(calib.keypoints),
        descriptors=calib.descriptors,
        polygon=calib.polygon,
        nfeatures=np.int32(calib.nfeatures),
    )
    # np.savez 会自动补 .npz 后缀
    return path if path.suffix == ".npz" else path.with_suffix(".npz")


def load_calibration(path: str | Path) -> Calibration:
    """读取 .npz 缓存文件，还原为 Calibration。"""
    with np.load(Path(path), allow_pickle=False) as data:
        return Calibration(
            ref_frame=data["ref_frame"],
            keypoints=features.keypoints_from_array(data["keypoints"]),
            descriptors=data["descriptors"],
            polygon=data["polygon"],
            nfeatures=int(data["nfeatures"]),
        )
