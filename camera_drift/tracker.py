"""全自动多边形校正 + 保护逻辑（需求3 的安全网，含昼夜/低纹理降级）。

把逐帧的「估 H -> 投影多边形」包成有状态的跟踪器：
  估计层（应对昼夜/低纹理）：
    - 多参考帧库择优：对每张 ref 估 H 取内点最多者（昼夜）；
    - ECC 降级：ORB 置信度不足时，对梯度图做 ECC 配准（低纹理/强光照）。
  保护层（需求3 四道网）：
    1. 置信度门控：内点数/比率/重投影误差达标（ORB），或 ECC 相关系数达标，才更新；
    2. 时序平滑：多边形顶点 EMA，抹平单帧抖动；
    3. 失败回退：都不达标时保持上一次好结果，多边形不乱跳；
    4. 大漂移告警：疑似平移/被撞时校正之外额外告警，提示人工确认是否重新基准化。

> 再往上的升级（v2）：低纹理/夜间仍失配时换学习型匹配 SuperPoint+SuperGlue / LoFTR，
>   可替换 estimate 层而不动保护层——此处留作扩展点，暂不实装（需重模型依赖）。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import angles as angles_mod
from . import drift as drift_mod
from . import ecc as ecc_mod
from . import homography
from . import multiref
from .calibration import Calibration


@dataclass
class TrackerConfig:
    min_inliers: int = 40            # 置信度门控：内点数下限
    min_inlier_ratio: float = 0.5    # 置信度门控：内点/好匹配比率下限
    max_reproj_err: float = 3.0      # 置信度门控：内点重投影误差上限(px)
    smooth_alpha: float = 0.3        # EMA 系数（新样本权重，越小越平滑/越滞后）
    drift_thresh_px: float = 15.0    # 正常偏移告警阈值
    big_drift_thresh_px: float = 60.0  # 大漂移告警阈值（疑似平移/被撞）
    use_ecc: bool = True             # ORB 不达标时是否启用 ECC 降级
    ecc_min_cc: float = 0.6          # ECC 路径置信度门控：相关系数下限


@dataclass
class TrackUpdate:
    polygon: np.ndarray              # 当前应使用的多边形（平滑后/或回退的上次结果）
    confident: bool                  # 本帧 H 是否通过置信度门控
    updated: bool                    # 多边形是否被本帧更新（=confident）
    drift: drift_mod.DriftReport     # 偏移报告
    angles: angles_mod.AngleResult | None  # 角度（仅 confident 时给）
    method: str                      # 本帧采用的估计方法：orb / ecc / none
    ref_name: str                    # 择优命中的参考帧名
    alarms: list[str] = field(default_factory=list)  # 本帧所有告警


class PolygonTracker:
    """对单台摄像头做全自动多边形校正，内置估计降级与保护逻辑。"""

    def __init__(
        self,
        source: Calibration | multiref.CalibrationBank,
        config: TrackerConfig | None = None,
    ):
        # 单张参考帧也统一包成库，上层逻辑只面对 bank
        self.bank = (
            source if isinstance(source, multiref.CalibrationBank)
            else multiref.CalibrationBank.from_calibration(source)
        )
        self.cfg = config or TrackerConfig()
        # 初始即标准位，先用参考多边形作为「上次好结果」
        self.smoothed: np.ndarray = self.bank.refs[0].polygon.copy()
        self.last_good_H: np.ndarray | None = None

    def _orb_confident(self, res: homography.HomographyResult) -> bool:
        return (
            res.ok
            and res.num_inliers >= self.cfg.min_inliers
            and res.inlier_ratio >= self.cfg.min_inlier_ratio
            and not np.isnan(res.reproj_err)
            and res.reproj_err <= self.cfg.max_reproj_err
        )

    def _estimate(self, frame: np.ndarray):
        """估计层：多参考帧 ORB 择优 -> 不达标则 ECC 降级。

        返回 (H, confident, method, ref_idx)。
        """
        res, idx = multiref.estimate_best(self.bank, frame)
        if self._orb_confident(res):
            return res.H, True, "orb", idx

        if self.cfg.use_ecc:
            init = res.H if res.ok else None
            H_ecc, cc = ecc_mod.estimate_ecc(self.bank.refs[idx].ref_frame, frame, init)
            if H_ecc is not None and cc >= self.cfg.ecc_min_cc:
                return H_ecc, True, f"ecc(cc={cc:.2f})", idx

        return None, False, "none", idx

    def update(self, frame: np.ndarray) -> TrackUpdate:
        cfg = self.cfg
        H, confident, method, idx = self._estimate(frame)
        name = self.bank.names[idx]
        alarms: list[str] = []

        if not confident or H is None:
            # 保护2/3：保持上次好结果，不更新多边形
            report = drift_mod.DriftReport(
                False, float("nan"), float("nan"), True,
                "H 不可靠（ORB 失配且 ECC 未达标）：保持上次好结果，需人工介入",
            )
            alarms.append(report.reason)
            return TrackUpdate(self.smoothed.copy(), False, False, report, None, method, name, alarms)

        # 偏移量（需求1）
        disp = drift_mod.corner_displacements(frame.shape, H)
        max_d, mean_d = float(disp.max()), float(disp.mean())
        alarm = max_d > cfg.drift_thresh_px
        reason = ("正常" if not alarm
                  else f"偏离标准位：最大角位移 {max_d:.1f}px > {cfg.drift_thresh_px}px")
        report = drift_mod.DriftReport(True, max_d, mean_d, alarm, reason)
        if alarm:
            alarms.append(reason)

        # 投影 + 时序平滑（需求3）
        raw = homography.project_polygon(self.bank.refs[idx].polygon, H)
        self.smoothed = cfg.smooth_alpha * raw + (1 - cfg.smooth_alpha) * self.smoothed
        self.last_good_H = H

        # 角度（需求2）
        ang = angles_mod.estimate_angles(H, frame.shape)

        # 保护4：大漂移（疑似平移/被撞）额外告警
        if max_d > cfg.big_drift_thresh_px:
            alarms.append(
                f"大漂移 {max_d:.0f}px，疑似平移/被撞，建议人工确认是否重新基准化"
            )

        return TrackUpdate(self.smoothed.copy(), True, True, report, ang, method, name, alarms)
