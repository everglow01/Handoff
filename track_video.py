"""视频流逐帧多边形校正：吃标定缓存 + 视频，输出多边形始终贴合的标注视频。

用法：
  # 先对首帧（标准位）标定出缓存（见 calibrate.py），再处理整段视频
  uv run python track_video.py cache/cam01.npz 输入.mp4 -o 输出.mp4

  # 每 5 帧才重估一次 H（中间帧复用上次好结果，省算力；契合「低频跑偏移检测」）
  uv run python track_video.py cache/cam01.npz 输入.mp4 -o 输出.mp4 --every 5

  # 实时预览（按 q 退出）
  uv run python track_video.py cache/cam01.npz 输入.mp4 --show

内部用 PolygonTracker（有状态、逐帧）：EMA 时序平滑 + 置信度门控 +
失败回退（某帧估歪时保持上次结果，多边形不乱跳）+ 大漂移告警。
绿色=校正后多边形，灰色细线=原参考位置。叠加文字用英文（OpenCV 字体不支持中文）。
"""

from __future__ import annotations

import argparse
import sys

import cv2
import numpy as np

from camera_drift.calibration import load_calibration
from camera_drift.tracker import PolygonTracker, TrackUpdate


def draw_polygon(image: np.ndarray, polygon: np.ndarray, color, thickness=2) -> None:
    pts = polygon.reshape(-1, 1, 2).astype(np.int32)
    cv2.polylines(image, [pts], isClosed=True, color=color, thickness=thickness)


def overlay_hud(canvas: np.ndarray, upd: TrackUpdate, frame_idx: int) -> None:
    """左上角叠加状态文字（英文，OpenCV Hershey 字体不支持中文）。"""
    alarmed = bool(upd.alarms)
    head_color = (0, 0, 255) if alarmed else (0, 200, 0)
    status = "ALARM" if alarmed else "OK"
    d = upd.drift
    drift_s = f"{d.max_disp_px:.0f}px" if d.ok else "n/a"
    lines = [
        f"frame {frame_idx}  [{status}]  method={upd.method}  ref={upd.ref_name}",
        f"drift_max={drift_s}  confident={upd.confident}",
    ]
    if upd.angles is not None and upd.angles.focal_px:
        a = upd.angles
        lines.append(f"pan={a.pan_deg:+.1f} tilt={a.tilt_deg:+.1f} roll={a.roll_deg:+.1f}")
    elif upd.angles is not None:
        lines.append(f"roll~={upd.angles.roll_simple_deg:+.1f} (pan/tilt n/a)")

    y = 24
    for i, text in enumerate(lines):
        color = head_color if i == 0 else (255, 255, 255)
        cv2.putText(canvas, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
        cv2.putText(canvas, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
        y += 26


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="视频逐帧多边形校正并输出标注视频")
    ap.add_argument("cache", help="标定缓存 .npz（对首帧/标准位标定得到）")
    ap.add_argument("video", help="输入视频路径")
    ap.add_argument("-o", "--out", default=None, help="输出标注视频路径（.mp4）")
    ap.add_argument("--every", type=int, default=1,
                    help="每 N 帧重估一次 H，中间帧复用上次结果（默认 1=逐帧）")
    ap.add_argument("--show", action="store_true", help="实时预览窗口（q 退出）")
    args = ap.parse_args(argv)

    if not args.out and not args.show:
        print("需指定 -o 输出视频 或 --show 预览（至少其一）", file=sys.stderr)
        return 1
    if args.every < 1:
        print("--every 必须 >= 1", file=sys.stderr)
        return 1

    calib = load_calibration(args.cache)
    tracker = PolygonTracker(calib)
    ref_polygon = calib.polygon

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"读不到视频: {args.video}", file=sys.stderr)
        return 1

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = None
    if args.out:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.out, fourcc, fps, (width, height))
        if not writer.isOpened():
            print(f"无法创建输出视频: {args.out}", file=sys.stderr)
            cap.release()
            return 1

    frame_idx = 0
    alarm_frames = 0
    last_upd: TrackUpdate | None = None
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            # --every：仅在该重估的帧调用 tracker；其余帧复用上次结果（含平滑后的多边形）
            if frame_idx % args.every == 0 or last_upd is None:
                last_upd = tracker.update(frame)
                if last_upd.alarms:
                    alarm_frames += 1
                    print(f"[帧 {frame_idx}] " + " | ".join(last_upd.alarms))

            canvas = frame.copy()
            draw_polygon(canvas, ref_polygon, (160, 160, 160), 1)       # 原位置（灰）
            draw_polygon(canvas, last_upd.polygon, (0, 255, 0), 2)      # 跟随后（绿）
            overlay_hud(canvas, last_upd, frame_idx)

            if writer is not None:
                writer.write(canvas)
            if args.show:
                cv2.imshow("track_video (q to quit)", canvas)
                if (cv2.waitKey(1) & 0xFF) == ord("q"):
                    break

            frame_idx += 1
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if args.show:
            cv2.destroyAllWindows()

    print(f"完成：处理 {frame_idx} 帧，其中 {alarm_frames} 次重估触发告警。")
    if args.out:
        print(f"已保存标注视频: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
