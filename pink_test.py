"""Paired present/absent test for pink backing paper.

Warps to a FIXED canonical size so pink pieces land at identical table coords in
both captures. Usage:
    .venv/bin/python pink_test.py capture present   # tools ON the pink pieces
    (remove the tools covering pink)
    .venv/bin/python pink_test.py capture absent     # pink exposed
    .venv/bin/python pink_test.py compare            # measure pink% drop per piece
"""
from __future__ import annotations
import cv2, numpy as np, sys, os
from lens import load_lens, undistort
from locate_table import load_board_calib, detect_union, locate, ctl, DICT

CANON = (1400, 826)                      # fixed so both captures align in table coords
PINK = ((148, 30, 60), (174, 255, 255))   # hue gate keeps foam/tools out; low S floor catches washed pink
DIR = "testing/pinktest"

def pink_mask(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    m = cv2.inRange(hsv, PINK[0], PINK[1])
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5,5), np.uint8))
    return cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((9,9), np.uint8))

def grab(ev=60):
    lens = load_lens(); board, _ = load_board_calib()
    det = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(DICT),
                                  cv2.aruco.DetectorParameters())
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    markers = detect_union(cap, det, lens)
    ctl(exposure_time_absolute=ev)
    for _ in range(6): cap.read()
    ok, frame = cap.read(); cap.release()
    res = locate(markers, board)
    if not res.get("ok"): print(f"[fail] {res['reason']}"); sys.exit(1)
    corners = res["corners"].astype(np.float32)
    M = cv2.getPerspectiveTransform(corners, np.array(
        [[0,0],[CANON[0],0],[CANON[0],CANON[1]],[0,CANON[1]]], np.float32))
    return cv2.warpPerspective(undistort(frame, lens), M, CANON), res["reproj"]

def main():
    os.makedirs(DIR, exist_ok=True)
    mode = sys.argv[1] if len(sys.argv) > 1 else "compare"
    if mode == "capture":
        label = sys.argv[2]
        canon, reproj = grab()
        cv2.imwrite(f"{DIR}/{label}.png", canon)
        print(f"[capture] {label}: reproj {reproj:.2f}px -> {DIR}/{label}.png")
        return
    # compare
    present = cv2.imread(f"{DIR}/present.png"); absent = cv2.imread(f"{DIR}/absent.png")
    if present is None or absent is None:
        print("[error] need both present.png and absent.png"); sys.exit(1)
    mp, ma = pink_mask(present), pink_mask(absent)
    # locate pink pieces from the ABSENT frame (all pink visible)
    n, lbl, stats, cent = cv2.connectedComponentsWithStats(ma)
    vis = present.copy()
    print(f"{'#':>2} {'cx':>5} {'cy':>5} {'absent%':>8} {'present%':>9}  verdict")
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < 800: continue
        x,y,w,h = stats[i,0],stats[i,1],stats[i,2],stats[i,3]
        box = (slice(y,y+h), slice(x,x+w))
        a_pct = 100*np.count_nonzero(ma[box])/(w*h)
        p_pct = 100*np.count_nonzero(mp[box])/(w*h)
        covered = p_pct < 0.4*a_pct       # pink dropped to <40% of exposed
        verdict = "COVERED (tool present)" if covered else "exposed (no tool)"
        col = (0,255,0) if covered else (0,180,255)
        cv2.rectangle(vis,(x,y),(x+w,y+h),col,2)
        cv2.putText(vis,f"{a_pct:.0f}->{p_pct:.0f}",(x,y-6),cv2.FONT_HERSHEY_SIMPLEX,0.5,col,2)
        print(f"{i:>2} {cent[i][0]:>5.0f} {cent[i][1]:>5.0f} {a_pct:>7.1f}% {p_pct:>8.1f}%  {verdict}")
    cv2.imwrite(f"{DIR}/compare.png", vis)
    print(f"[saved] {DIR}/compare.png")

if __name__ == "__main__":
    main()
