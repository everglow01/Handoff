"""多参考帧库——应对昼夜/光照变化（本项目#1风险）。

单张参考帧扛不住 7×24 的昼夜差异。这里维护一组在不同时段/光照下采的
参考帧（白天/黄昏/夜晚...），当前帧来时对每张 ref 各估一次 H，择优
（内点最多者）。夜间帧会自然匹配到夜间 ref，白天 ref 内点寥寥。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import homography
from .calibration import Calibration, load_calibration


@dataclass
class CalibrationBank:
    """一组参考帧标定（同一台相机、同一标准位、不同光照）。"""

    refs: list[Calibration]
    names: list[str]

    def __post_init__(self):
        if not self.refs:
            raise ValueError("参考帧库不能为空")
        if len(self.refs) != len(self.names):
            raise ValueError("refs 与 names 数量不一致")

    @classmethod
    def from_calibration(cls, calib: Calibration, name: str = "ref0") -> "CalibrationBank":
        """单张参考帧也包成库，让上层统一按库处理。"""
        return cls([calib], [name])

    @classmethod
    def load(cls, paths: list[str | Path]) -> "CalibrationBank":
        refs = [load_calibration(p) for p in paths]
        names = [Path(p).stem for p in paths]
        return cls(refs, names)


def estimate_best(
    bank: CalibrationBank,
    frame: np.ndarray,
    **kwargs,
) -> tuple[homography.HomographyResult, int]:
    """对库中每张 ref 估 H，返回内点最多的 (result, ref索引)。"""
    best_res: homography.HomographyResult | None = None
    best_idx = 0
    for i, calib in enumerate(bank.refs):
        res = homography.estimate_homography(calib, frame, **kwargs)
        if best_res is None or res.num_inliers > best_res.num_inliers:
            best_res, best_idx = res, i
    return best_res, best_idx
