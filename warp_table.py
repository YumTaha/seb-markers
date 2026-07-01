"""Warp the located table to a flat top-down 'canonical' image.

Once flattened, tool slots are fixed rectangles in this image regardless of how
the physical table sits in the camera view. Foundation for slot/presence analysis.

    .venv/bin/python warp_table.py     # saves testing/warp_<ts>.png
"""
from __future__ import annotations
import cv2, numpy as np, time, os, sys
from lens import load_lens
from locate_table import load_board_calib, detect_union, locate, ctl, DICT

CANON_W = 1200   # canonical width in px; height derived from table aspect

def main():
    lens = load_lens()
    board_calib, _ = load_board_calib()
    detector = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(DICT),
                                       cv2.aruco.DetectorParameters())
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    if not cap.isOpened():
        print("[error] cannot open camera"); sys.exit(1)
    markers = detect_union(cap, detector, lens)
    ctl(exposure_time_absolute=90)
    for _ in range(6): cap.read()
    ok, frame = cap.read(); cap.release()
    from lens import undistort
    frame = undistort(frame, lens)

    res = locate(markers, board_calib)
    if not res.get("ok"):
        print(f"[fail] {res.get('reason')}"); sys.exit(1)
    corners = res["corners"].astype(np.float32)   # TL,TR,BR,BL in image

    # estimate table aspect from the derived corners
    w_top = np.linalg.norm(corners[1]-corners[0]); w_bot = np.linalg.norm(corners[2]-corners[3])
    h_l = np.linalg.norm(corners[3]-corners[0]); h_r = np.linalg.norm(corners[2]-corners[1])
    aspect = (w_top+w_bot) / (h_l+h_r)
    W = CANON_W; H = int(round(W/aspect))
    dst = np.array([[0,0],[W,0],[W,H],[0,H]], np.float32)
    M = cv2.getPerspectiveTransform(corners, dst)
    canon = cv2.warpPerspective(frame, M, (W, H))

    os.makedirs("testing", exist_ok=True)
    out = f"testing/warp_{time.strftime('%H%M%S')}.png"
    cv2.imwrite(out, canon)
    print(f"[warp] canonical {W}x{H} (aspect {aspect:.2f}), reproj {res['reproj']:.2f}px -> {out}")

if __name__ == "__main__":
    main()
