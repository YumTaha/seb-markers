from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

WINDOW = "Canny Pipeline"

STEP_NAMES = [
    "0: Input",
    "1: Gaussian",
    "2: Gradients (Gx / Gy / Magnitude)",
    "3: Non-maximum suppression",
    "4: Hysteresis thresholding",
]


def gaussian_smooth(gray: np.ndarray, sigma: float) -> np.ndarray:
    ksize = int(6 * sigma + 1) | 1
    return cv2.GaussianBlur(gray, (ksize, ksize), sigma)


def compute_gradients(
    smooth: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    gx = cv2.Sobel(smooth, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(smooth, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(gx**2 + gy**2)
    angle = np.arctan2(gy, gx)
    return gx, gy, mag, angle


def non_maximum_suppression(mag: np.ndarray, angle: np.ndarray) -> np.ndarray:
    deg = np.rad2deg(angle) % 180
    d = np.zeros(mag.shape, dtype=np.uint8)
    d[(deg >= 22.5) & (deg < 67.5)] = 1
    d[(deg >= 67.5) & (deg < 112.5)] = 2
    d[(deg >= 112.5) & (deg < 157.5)] = 3

    p = np.pad(mag, 1)
    neighbors = [
        (p[1:-1, :-2], p[1:-1, 2:]),
        (p[:-2, 2:], p[2:, :-2]),
        (p[:-2, 1:-1], p[2:, 1:-1]),
        (p[:-2, :-2], p[2:, 2:]),
    ]
    out = np.zeros_like(mag)
    for i, (a, b) in enumerate(neighbors):
        mask = (d == i) & (mag >= a) & (mag >= b)
        out[mask] = mag[mask]
    return out


def hysteresis_threshold(nms: np.ndarray, t_low: float, t_high: float) -> np.ndarray:
    strong = (nms >= t_high).astype(np.uint8) * 255
    weak = ((nms >= t_low) & (nms < t_high)).astype(np.uint8) * 255
    combined = np.maximum(strong, weak)
    n, labels = cv2.connectedComponents(combined)
    out = np.zeros_like(nms, dtype=np.uint8)
    for lbl in range(1, n):
        mask = labels == lbl
        if strong[mask].any():
            out[mask] = 1
    return out


def _norm_u8(arr: np.ndarray) -> np.ndarray:
    mn, mx = arr.min(), arr.max()
    if mx == mn:
        return np.zeros_like(arr, dtype=np.uint8)
    return ((arr - mn) / (mx - mn) * 255).astype(np.uint8)


def _to_bgr(gray: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def _overlay_label(img: np.ndarray, text: str) -> np.ndarray:
    out = img.copy()
    cv2.putText(out, text, (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(out, text, (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2, cv2.LINE_AA)
    return out


def compute_steps(
    image: np.ndarray, sigma: float, t_low: float, t_high: float
) -> list[np.ndarray]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    smooth = gaussian_smooth(gray, sigma)
    gx, gy, mag, angle = compute_gradients(smooth)
    nms = non_maximum_suppression(mag, angle)
    edges = hysteresis_threshold(nms, t_low, t_high)

    h, w = image.shape[:2]
    sub_h = h // 3

    sub_gx = cv2.resize(_to_bgr(_norm_u8(np.abs(gx))), (w, sub_h))
    sub_gy = cv2.resize(_to_bgr(_norm_u8(np.abs(gy))), (w, sub_h))
    sub_mag = cv2.resize(_to_bgr(_norm_u8(mag)), (w, h - 2 * sub_h))

    grad_label = np.zeros((20, w, 3), dtype=np.uint8)
    for i, (text, col) in enumerate([("Gx", sub_h), ("Gy", 2 * sub_h), ("Mag", h)]):
        cv2.putText(sub_gx if i == 0 else sub_gy if i == 1 else sub_mag,
                    text, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 200), 1)
    step2 = np.vstack([sub_gx, sub_gy, sub_mag])

    step4 = image.copy()
    step4[edges == 1] = (0, 0, 255)

    return [
        image.copy(),
        _to_bgr(smooth),
        step2,
        _to_bgr(_norm_u8(nms)),
        step4,
    ]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("input", type=Path)
    p.add_argument("--width", type=int, default=900, help="display width in pixels (default 900)")
    args = p.parse_args()

    src = cv2.imread(str(args.input))
    if src is None:
        print(f"Cannot read: {args.input}", file=sys.stderr)
        sys.exit(1)

    h, w = src.shape[:2]
    if w > args.width:
        scale = args.width / w
        src = cv2.resize(src, (args.width, int(h * scale)))

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.createTrackbar("Step  (0-4)", WINDOW, 4, 4, lambda _: None)
    cv2.createTrackbar("sigma x10  ", WINDOW, 14, 50, lambda _: None)
    cv2.createTrackbar("t_low      ", WINDOW, 20, 255, lambda _: None)
    cv2.createTrackbar("t_high     ", WINDOW, 50, 255, lambda _: None)

    prev_params: tuple | None = None
    steps: list[np.ndarray] = []
    frame: np.ndarray | None = None

    print("Trackbars: Step(0-4), sigma, t_low, t_high  |  S = save  |  Q/Esc = quit")

    while True:
        step_idx = cv2.getTrackbarPos("Step  (0-4)", WINDOW)
        sigma = max(cv2.getTrackbarPos("sigma x10  ", WINDOW), 1) / 10.0
        t_low = cv2.getTrackbarPos("t_low      ", WINDOW)
        t_high = cv2.getTrackbarPos("t_high     ", WINDOW)

        params = (sigma, t_low, t_high)
        if params != prev_params:
            steps = compute_steps(src, sigma, t_low, t_high)
            prev_params = params

        idx = max(0, min(step_idx, 4))
        new_frame = _overlay_label(steps[idx], STEP_NAMES[idx])

        if frame is None or not np.array_equal(new_frame, frame):
            frame = new_frame
            cv2.imshow(WINDOW, frame)

        key = cv2.waitKey(30) & 0xFF
        if key == ord("s") and frame is not None:
            out = Path(args.input).stem + f"_step{idx}.png"
            cv2.imwrite(out, frame)
            print(f"Saved: {out}")
        elif key in (27, ord("q")):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
