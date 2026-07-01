"""Bake-off for the colored backing paper: which color segments most reliably.

Locate the table ONCE, then at several color-analysis exposures grab a frame, warp
with the same homography, and segment each candidate color. Reports detected area +
blob count per color per exposure (stability), and saves an overlay per exposure so we
can see false positives (tools/foam leaking into a color range).
"""
from __future__ import annotations
import cv2, numpy as np, time, os, sys
from lens import load_lens, undistort
from locate_table import load_board_calib, detect_union, locate, ctl, DICT

CANON_W = 1400
EXPO = [25, 60, 110, 180, 300]
# candidate pastel ranges (OpenCV HSV: H 0-180)
RANGES = {
    "pink":   [((150, 55, 70), (172, 255, 255))],
    "green":  [((38, 30, 70), (85, 255, 255))],
    "yellow": [((22, 30, 130), (36, 200, 255))],
}
DRAW = {"pink": (255, 0, 255), "green": (0, 255, 0), "yellow": (0, 255, 255)}

def color_mask(hsv, ranges):
    m = np.zeros(hsv.shape[:2], np.uint8)
    for lo, hi in ranges:
        m = cv2.bitwise_or(m, cv2.inRange(hsv, lo, hi))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5,5), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((9,9), np.uint8))
    return m

def main():
    lens = load_lens(); board, _ = load_board_calib()
    det = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(DICT),
                                  cv2.aruco.DetectorParameters())
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    if not cap.isOpened(): print("[error] no camera"); sys.exit(1)

    # locate table ONCE
    markers = detect_union(cap, det, lens)
    res = locate(markers, board)
    if not res.get("ok"): print(f"[fail] {res.get('reason')}"); sys.exit(1)
    corners = res["corners"].astype(np.float32)
    w_top = np.linalg.norm(corners[1]-corners[0]); h_l = np.linalg.norm(corners[3]-corners[0])
    W = CANON_W; H = int(round(W * (h_l / w_top)))
    M = cv2.getPerspectiveTransform(corners, np.array([[0,0],[W,0],[W,H],[0,H]], np.float32))
    print(f"located (reproj {res['reproj']:.2f}px), canonical {W}x{H}\n")

    ts = time.strftime("%H%M%S")
    outdir = f"testing/paperbake_{ts}"; os.makedirs(outdir, exist_ok=True)
    print(f"{'expo':>5} " + "".join(f"{c+'(area/blobs)':>20}" for c in RANGES))
    area_hist = {c: [] for c in RANGES}
    for ev in EXPO:
        ctl(auto_exposure=1, gain=0, exposure_time_absolute=ev)
        for _ in range(6): cap.read()
        ok, frame = cap.read()
        if not ok: continue
        canon = cv2.warpPerspective(undistort(frame, lens), M, (W, H))
        cv2.imwrite(f"{outdir}/canon_ev{ev:03d}.png", canon)
        hsv = cv2.cvtColor(canon, cv2.COLOR_BGR2HSV)
        overlay = canon.copy()
        line = f"{ev:>5} "
        for c, ranges in RANGES.items():
            m = color_mask(hsv, ranges)
            nb, _, stats, _ = cv2.connectedComponentsWithStats(m)
            blobs = [i for i in range(1, nb) if stats[i, cv2.CC_STAT_AREA] > 500]
            area = int(sum(stats[i, cv2.CC_STAT_AREA] for i in blobs))
            area_hist[c].append(area)
            overlay[m > 0] = DRAW[c]
            line += f"{f'{area}/{len(blobs)}':>20}"
        cv2.imwrite(f"{outdir}/overlay_ev{ev:03d}.png", overlay)
        print(line)
    cap.release()
    print("\narea stability across exposures (want: high & steady, blob count = # papers):")
    for c in RANGES:
        a = area_hist[c]
        print(f"  {c:7s}: {a}  (min {min(a)}, max {max(a)})")
    print(f"\n[saved] {outdir}/ (canon + overlay per exposure)")

if __name__ == "__main__":
    main()
