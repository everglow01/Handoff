"""标定脚本：对一张参考帧提特征、记录多边形，存成缓存文件。

用法：
  # 多边形来自 JSON 文件（[[x,y],[x,y],...]，参考帧坐标系）
  uv run python calibrate.py ref.jpg --polygon-file poly.json -o cache/cam01.npz

  # 多边形内联
  uv run python calibrate.py ref.jpg --polygon "100,100 380,100 380,380 100,380"

  # 交互画：左键加点，回车/右键完成，esc 取消
  uv run python calibrate.py ref.jpg --draw -o cache/cam01.npz
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

from camera_drift import calibration


def _parse_inline(text: str) -> np.ndarray:
    pts = [tuple(map(float, tok.split(","))) for tok in text.split()]
    return np.array(pts, dtype=np.float32)


def _load_polygon_file(path: str) -> np.ndarray:
    return np.array(json.loads(Path(path).read_text()), dtype=np.float32)


def _draw_polygon(image: np.ndarray) -> np.ndarray:
    """交互画多边形：左键加点，回车/右键完成，esc 取消。"""
    pts: list[tuple[int, int]] = []
    win = "draw polygon (L-click add, Enter/R-click done, Esc cancel)"

    def on_mouse(event, x, y, flags, _):
        if event == cv2.EVENT_LBUTTONDOWN:
            pts.append((x, y))
        elif event == cv2.EVENT_RBUTTONDOWN and len(pts) >= 3:
            cv2.setWindowProperty(win, cv2.WND_PROP_AUTOSIZE, -2)  # 触发退出标记

    cv2.namedWindow(win)
    cv2.setMouseCallback(win, on_mouse)
    while True:
        canvas = image.copy()
        if canvas.ndim == 2:
            canvas = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)
        for i, p in enumerate(pts):
            cv2.circle(canvas, p, 4, (0, 255, 0), -1)
            if i > 0:
                cv2.line(canvas, pts[i - 1], p, (0, 255, 0), 2)
        if len(pts) >= 3:
            cv2.line(canvas, pts[-1], pts[0], (0, 200, 0), 1)
        cv2.imshow(win, canvas)
        key = cv2.waitKey(20) & 0xFF
        if key == 13 and len(pts) >= 3:   # Enter
            break
        if key == 27:                     # Esc
            pts = []
            break
        if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
            break
    cv2.destroyWindow(win)
    if len(pts) < 3:
        raise SystemExit("已取消或点数不足，未生成多边形")
    return np.array(pts, dtype=np.float32)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="生成摄像头标定缓存")
    ap.add_argument("ref_image", help="标准位置参考帧图片路径")
    ap.add_argument("-o", "--out", default=None, help="缓存输出路径（默认 cache/<图名>.npz）")
    ap.add_argument("--polygon-file", help="多边形 JSON 文件 [[x,y],...]")
    ap.add_argument("--polygon", help='内联多边形 "x,y x,y ..."')
    ap.add_argument("--draw", action="store_true", help="交互画多边形")
    ap.add_argument("--nfeatures", type=int, default=2000, help="ORB 特征数")
    args = ap.parse_args(argv)

    frame = cv2.imread(args.ref_image, cv2.IMREAD_COLOR)
    if frame is None:
        print(f"读不到图片: {args.ref_image}", file=sys.stderr)
        return 1

    if args.polygon_file:
        polygon = _load_polygon_file(args.polygon_file)
    elif args.polygon:
        polygon = _parse_inline(args.polygon)
    elif args.draw:
        polygon = _draw_polygon(frame)
    else:
        print("需提供 --polygon-file / --polygon / --draw 之一", file=sys.stderr)
        return 1

    calib = calibration.calibrate(frame, polygon, nfeatures=args.nfeatures)

    out = args.out or str(Path("cache") / (Path(args.ref_image).stem + ".npz"))
    saved = calibration.save_calibration(out, calib)
    print(f"已保存标定缓存: {saved}")
    print(f"  参考帧: {calib.ref_frame.shape} | 特征: {len(calib.keypoints)} | 多边形: {len(calib.polygon)} 点")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
