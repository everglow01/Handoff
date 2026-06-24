"""环境冒烟检查：验证 v1 单应估计工具链是否可用。

构造一张合成纹理图，对它施加一个已知单应变换，再用
ORB + BFMatcher + findHomography(RANSAC) 把变换估回来，
最后用 perspectiveTransform 投影一个多边形。
全链路跑通 => 第一步（工程骨架 + OpenCV 环境）完成。
"""

import cv2
import numpy as np

from camera_drift import __version__


def _make_textured_image(size: int = 480) -> np.ndarray:
    """生成一张有丰富纹理的灰度图，保证 ORB 能稳定提特征。"""
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, size=(size, size), dtype=np.uint8)
    # 叠加一些几何结构，让特征更可重复
    for _ in range(40):
        p1 = tuple(rng.integers(0, size, size=2).tolist())
        p2 = tuple(rng.integers(0, size, size=2).tolist())
        cv2.line(img, p1, p2, int(rng.integers(0, 256)), 2)
    return img


def run_smoke_check() -> None:
    print(f"camera-drift v{__version__}")
    print(f"OpenCV {cv2.__version__}, NumPy {np.__version__}")

    ref = _make_textured_image()

    # 已知的"真值"单应：绕画面中心做一点旋转 + 透视，模拟相机偏移
    h, w = ref.shape
    center = (w / 2, h / 2)
    rot = cv2.getRotationMatrix2D(center, angle=8.0, scale=1.0)
    H_true = np.vstack([rot, [0, 0, 1]]).astype(np.float64)
    H_true[2, 0] = 1e-4  # 一点点透视，逼近 pan/tilt 产生的梯形畸变
    cur = cv2.warpPerspective(ref, H_true, (w, h))

    # ORB + 比率测试 + RANSAC 估 H
    orb = cv2.ORB_create(nfeatures=2000)
    k1, d1 = orb.detectAndCompute(ref, None)
    k2, d2 = orb.detectAndCompute(cur, None)

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    knn = matcher.knnMatch(d1, d2, k=2)
    good = [m for m, n in knn if m.distance < 0.75 * n.distance]

    src = np.float32([k1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([k2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    H_est, mask = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)

    inliers = int(mask.sum())
    print(f"ORB 特征: ref={len(k1)} cur={len(k2)} | good={len(good)} | inliers={inliers}")

    # 用估出的 H 投影一个多边形，看是否落在合理范围内
    poly_ref = np.float32([[100, 100], [380, 100], [380, 380], [100, 380]]).reshape(-1, 1, 2)
    poly_cur = cv2.perspectiveTransform(poly_ref, H_est)

    # 与真值变换对比，验证估计精度
    poly_true = cv2.perspectiveTransform(poly_ref, H_true)
    err = float(np.linalg.norm(poly_cur - poly_true, axis=2).mean())
    print(f"多边形投影平均误差: {err:.3f} px")

    assert inliers >= 50, "内点过少，工具链可能有问题"
    assert err < 2.0, "多边形投影误差过大，H 估计不准"
    print("OK: v1 单应估计工具链跑通。")


if __name__ == "__main__":
    run_smoke_check()
