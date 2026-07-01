"""Probe: grab a flattened (canonical) view and locate the colored paper blobs.

Reports each saturated blob's hue/sat/value/area/centroid + a color guess, and saves
an overlay so we can confirm which blobs are the yellow/green/pink papers (vs the
table's own colored tools) before running the full bake-off.
"""
from __future__ import annotations
import cv2, numpy as np, time, os, sys
from lens import load_lens, undistort
from locate_table import load_board_calib, detect_union, locate, ctl, DICT

CANON_W = 1400

def grab_warped(exposure=90):
    lens = load_lens(); board, _ = load_board_calib()
    det = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(DICT),
                                  cv2.aruco.DetectorParameters())
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    if not cap.isOpened(): print("[error] no camera"); sys.exit(1)
    markers = detect_union(cap, det, lens)
    ctl(exposure_time_absolute=exposure)
    for _ in range(6): cap.read()
    ok, frame = cap.read(); cap.release()
    frame = undistort(frame, lens)
    res = locate(markers, board)
    if not res.get("ok"):
        print(f"[fail] locate: {res.get('reason')}"); sys.exit(1)
    corners = res["corners"].astype(np.float32)
    w_top = np.linalg.norm(corners[1]-corners[0]); h_l = np.linalg.norm(corners[3]-corners[0])
    W = CANON_W; H = int(round(W * (h_l / w_top)))
    M = cv2.getPerspectiveTransform(corners, np.array([[0,0],[W,0],[W,H],[0,H]], np.float32))
    return cv2.warpPerspective(frame, M, (W, H)), res

def guess_color(h, s, v):
    if s < 60: return "grey/foam?"
    if h < 10 or h >= 172: return "red"
    if h < 20: return "orange"
    if h < 35: return "yellow"
    if h < 90: return "green"
    if h < 130: return "blue"
    if h < 172: return "pink/magenta"
    return "?"

def main():
    canon, res = grab_warped()
    os.makedirs("testing", exist_ok=True)
    ts = time.strftime("%H%M%S")
    cv2.imwrite(f"testing/paper_ref_{ts}.png", canon)
    hsv = cv2.cvtColor(canon, cv2.COLOR_BGR2HSV)
    sat, val = hsv[:,:,1], hsv[:,:,2]
    mask = ((sat > 80) & (val > 50)).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5,5), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9,9), np.uint8))
    n, lbl, stats, cent = cv2.connectedComponentsWithStats(mask)
    overlay = canon.copy()
    print(f"reprojection {res['reproj']:.2f}px; saturated blobs (area>800):")
    print(f"{'#':>2} {'area':>6} {'cx':>5} {'cy':>5} {'H':>4} {'S':>4} {'V':>4}  guess")
    idx = 0
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < 800: continue
        m = (lbl == i)
        H = int(np.median(hsv[:,:,0][m])); S = int(np.median(sat[m])); V = int(np.median(val[m]))
        cx, cy = cent[i]
        idx += 1
        g = guess_color(H, S, V)
        x,y,w,h = stats[i,0],stats[i,1],stats[i,2],stats[i,3]
        cv2.rectangle(overlay,(x,y),(x+w,y+h),(0,255,0),2)
        cv2.putText(overlay,f"{idx}:{g}",(x,y-6),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),2)
        print(f"{idx:>2} {area:>6} {cx:>5.0f} {cy:>5.0f} {H:>4} {S:>4} {V:>4}  {g}")
    cv2.imwrite(f"testing/paper_probe_{ts}.png", overlay)
    print(f"[saved] testing/paper_ref_{ts}.png  testing/paper_probe_{ts}.png")

if __name__ == "__main__":
    main()
