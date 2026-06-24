"""估 H 主链路演示：载入标定缓存 + 当前帧 -> 估 H -> 投影多边形。

用法：
  uv run python track.py cache/cam01.npz current.jpg -o out.png

输出当前帧上投影后的多边形（绿色）与原参考多边形位置（灰色虚线对比），
并打印好匹配数 / 内点数。多边形稳稳跟住真实区域 => 第三步成立。
"""

from __future__ import annotations

import argparse
import sys

import cv2
import numpy as np

from camera_drift import angles, drift, homography
from camera_drift.calibration import load_calibration


def draw_polygon(image: np.ndarray, polygon: np.ndarray, color, thickness=2) -> np.ndarray:
    pts = polygon.reshape(-1, 1, 2).astype(np.int32)
    cv2.polylines(image, [pts], isClosed=True, color=color, thickness=thickness)
    return image


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="估 H 并投影多边形")
    ap.add_argument("cache", help="标定缓存 .npz")
    ap.add_argument("current", help="当前帧图片")
    ap.add_argument("-o", "--out", default=None, help="标注输出图路径")
    ap.add_argument("--ratio", type=float, default=0.75)
    ap.add_argument("--ransac", type=float, default=3.0)
    ap.add_argument("--drift-thresh", type=float, default=15.0, help="角位移告警阈值(px)")
    args = ap.parse_args(argv)

    calib = load_calibration(args.cache)
    frame = cv2.imread(args.current, cv2.IMREAD_COLOR)
    if frame is None:
        print(f"读不到当前帧: {args.current}", file=sys.stderr)
        return 1

    res = homography.estimate_homography(calib, frame, ratio=args.ratio, ransac_thresh=args.ransac)
    print(f"good={res.num_good} inliers={res.num_inliers} ratio={res.inlier_ratio:.2f} ok={res.ok}")

    # 需求1：偏移检测与告警（含 H 失败兜底）
    report = drift.evaluate(res, frame.shape, drift_thresh_px=args.drift_thresh)
    tag = "⚠ 告警" if report.alarm else "正常"
    print(f"[偏移] {tag}: {report.reason}")
    if report.ok:
        print(f"       最大角位移={report.max_disp_px:.1f}px 平均={report.mean_disp_px:.1f}px")

    if not res.ok:
        return 2

    # 需求2：角度估计 pan/tilt/roll
    ang = angles.estimate_angles(res.H, frame.shape)
    if ang.focal_px:
        print(f"[角度] pan={ang.pan_deg:+.2f}° tilt={ang.tilt_deg:+.2f}° "
              f"roll={ang.roll_deg:+.2f}° (f≈{ang.focal_px:.0f}px)")
    else:
        print(f"[角度] f 自标定失败 -> 退化为 roll≈{ang.roll_simple_deg:+.2f}°（pan/tilt 不可分离）")

    # 需求3：多边形投影
    poly_cur = homography.project_polygon(calib.polygon, res.H)
    print("投影后多边形:", np.round(poly_cur, 1).tolist())

    if args.out:
        canvas = frame.copy()
        draw_polygon(canvas, calib.polygon, (160, 160, 160), 1)   # 原位置（灰）
        draw_polygon(canvas, poly_cur, (0, 255, 0), 2)            # 跟随后（绿）
        cv2.imwrite(args.out, canvas)
        print(f"已保存标注图: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
