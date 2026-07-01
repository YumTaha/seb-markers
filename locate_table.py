"""Runtime: locate the table from the calibrated markers.

Pipeline:  undistort -> detect markers (grayscale, exposure-bracket union)
        -> fit ONE homography (normalized table <-> image) from every detected
           marker's 4 corners using board_calib.json
        -> derive the 4 table corners, and VALIDATE (enough markers, low
           reprojection error, all corners in-frame, sane quad).

Run headless (saves testing/locate_<ts>.png + prints a verdict):
    .venv/bin/python locate_table.py
"""
from __future__ import annotations
import cv2, numpy as np, subprocess, json, time, os, sys
from lens import load_lens, undistort

CAM = 0
DICT = cv2.aruco.DICT_4X4_50
BRACKET = [40, 90, 180]          # exposures to union marker detections over
GAIN = 0
NORM_CORNERS = np.array([[0,0],[1,0],[1,1],[0,1]], np.float32)
# validation thresholds
MIN_MARKERS = 2
MAX_REPROJ  = 12.0               # px
EDGE_MARGIN = 5                  # px; derived corners must be within frame + this

def ctl(**kw):
    for k, v in kw.items():
        subprocess.run(["v4l2-ctl","-d",f"/dev/video{CAM}","-c",f"{k}={v}"], capture_output=True)

def load_board_calib(path="board_calib.json"):
    d = json.load(open(path))
    markers = {int(k): np.array(v, np.float32) for k, v in d["markers"].items()}
    return markers, d["image_size"]

def detect_union(cap, detector, lens):
    """Union marker detections across the exposure bracket. Returns {id: 4x2 img corners}."""
    ctl(auto_exposure=1, gain=GAIN)
    found = {}
    for ev in BRACKET:
        ctl(exposure_time_absolute=ev)
        for _ in range(6): cap.read()
        ok, f = cap.read()
        if not ok: continue
        f = undistort(f, lens)
        c, ids, _ = detector.detectMarkers(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY))
        if ids is None: continue
        for k, i in enumerate(ids.flatten()):
            found.setdefault(int(i), c[k][0])   # keep first exposure that sees it
    return found

def locate(markers_img, board_calib):
    """Fit homography norm->img from all shared markers. Returns dict of results."""
    src, dst = [], []            # normalized, image
    for mid, img_quad in markers_img.items():
        if mid in board_calib:
            src.append(board_calib[mid]); dst.append(img_quad)
    n = len(src)
    if n < MIN_MARKERS:
        return {"ok": False, "reason": f"only {n} calibrated markers seen (<{MIN_MARKERS})", "n": n}
    src = np.vstack(src).astype(np.float32); dst = np.vstack(dst).astype(np.float32)
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
    if H is None:
        return {"ok": False, "reason": "homography failed", "n": n}
    # reprojection error
    proj = cv2.perspectiveTransform(src.reshape(1,-1,2), H).reshape(-1,2)
    reproj = float(np.sqrt(((proj - dst)**2).sum(axis=1)).mean())
    corners = cv2.perspectiveTransform(NORM_CORNERS.reshape(1,-1,2), H).reshape(-1,2)
    return {"ok": True, "n": n, "H": H, "reproj": reproj, "corners": corners}

def validate(res, img_size):
    if not res.get("ok"):
        return False, [res.get("reason","fail")]
    w, h = img_size
    problems = []
    if res["reproj"] > MAX_REPROJ:
        problems.append(f"high reprojection error {res['reproj']:.1f}px (>{MAX_REPROJ})")
    cs = res["corners"]
    for name, (x, y) in zip(["TL","TR","BR","BL"], cs):
        if not (-EDGE_MARGIN <= x <= w+EDGE_MARGIN and -EDGE_MARGIN <= y <= h+EDGE_MARGIN):
            problems.append(f"corner {name} out of frame ({x:.0f},{y:.0f}) -> table not fully captured")
    if cv2.contourArea(cs.astype(np.float32)) < 0.15 * w * h:
        problems.append("table quad too small / degenerate")
    if not cv2.isContourConvex(cs.astype(np.float32)):
        problems.append("table quad not convex")
    return (len(problems) == 0), problems

def main():
    lens = load_lens()
    board_calib, calib_size = load_board_calib()
    detector = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(DICT),
                                       cv2.aruco.DetectorParameters())
    cap = cv2.VideoCapture(CAM, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    if not cap.isOpened():
        print("[error] cannot open camera"); sys.exit(1)

    markers_img = detect_union(cap, detector, lens)
    # grab one clean display frame (mid exposure), undistorted
    ctl(exposure_time_absolute=90)
    for _ in range(6): cap.read()
    ok, disp = cap.read(); cap.release()
    disp = undistort(disp, lens)

    print(f"[detect] markers seen: {sorted(markers_img)}  (calibrated: {sorted(board_calib)})")
    res = locate(markers_img, board_calib)
    passed, problems = validate(res, calib_size)

    # draw
    for mid, quad in markers_img.items():
        cv2.polylines(disp, [quad.astype(np.int32)], True, (0,255,0), 2)
        c = quad.mean(axis=0).astype(int)
        cv2.putText(disp, str(mid), tuple(c), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
    if res.get("ok"):
        cs = res["corners"].astype(np.int32)
        col = (0,255,0) if passed else (0,180,255)
        cv2.polylines(disp, [cs], True, col, 3)
        for name,(x,y) in zip(["TL","TR","BR","BL"], res["corners"]):
            cv2.circle(disp,(int(x),int(y)),10,col,-1)
            cv2.putText(disp,name,(int(x)+12,int(y)),cv2.FONT_HERSHEY_SIMPLEX,0.9,col,2)
        print(f"[locate] used {res['n']} markers, reprojection error {res['reproj']:.2f} px")
        print("[corners] " + "  ".join(f"{n}=({x:.0f},{y:.0f})"
              for n,(x,y) in zip(["TL","TR","BR","BL"], res["corners"])))
    verdict = "PASS" if passed else "FAIL"
    print(f"[verdict] {verdict}" + ("" if passed else ":  " + "; ".join(problems)))
    cv2.putText(disp, verdict, (20,60), cv2.FONT_HERSHEY_SIMPLEX, 1.6,
                (0,255,0) if passed else (0,0,255), 3)
    os.makedirs("testing", exist_ok=True)
    out = f"testing/locate_{time.strftime('%H%M%S')}.png"
    cv2.imwrite(out, disp); print(f"[saved] {out}")

if __name__ == "__main__":
    main()
