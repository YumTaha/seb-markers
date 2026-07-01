"""Click-calibration: record each marker's position relative to the table.

Run this FROM THE RDP DESKTOP (lxterminal) so the window opens in your session:

    cd ~/seb-markers && .venv/bin/python calibrate_board.py

It grabs a clean frame (auto-picking an exposure that sees the markers), then you
click the 4 TRUE TABLE CORNERS in order:  TL -> TR -> BR -> BL.
Keys:  r = reset clicks,  s = save,  q = quit without saving.

Output: board_calib.json
  { "markers": { "<id>": [[x,y]*4 normalized table coords of the marker's corners] },
    "image_size": [w,h], "table_corners_image": [[x,y]*4],
    "exposure": <ev>, "marker_color": "yellow" }
At runtime: detect markers, match their image corners to these stored normalized
coords, fit one homography, and derive the 4 table corners.
"""
from __future__ import annotations
import os
os.environ.setdefault("QT_QPA_PLATFORM", "xcb")
import cv2, numpy as np, subprocess, json, sys
from lens import load_lens, undistort

CAM = 0
EXPO_TRY = [60, 90, 130, 180, 250]   # pick the one detecting the most markers
GAIN = 0
DICT = cv2.aruco.DICT_4X4_50
CORNER_NAMES = ["TL", "TR", "BR", "BL"]
NORM_CORNERS = np.array([[0,0],[1,0],[1,1],[0,1]], np.float32)

def ctl(**kw):
    for k, v in kw.items():
        subprocess.run(["v4l2-ctl", "-d", f"/dev/video{CAM}", "-c", f"{k}={v}"],
                       capture_output=True)

def grab_best():
    cap = cv2.VideoCapture(CAM, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    if not cap.isOpened():
        print("[error] cannot open camera"); sys.exit(1)
    det = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(DICT),
                                  cv2.aruco.DetectorParameters())
    lens = load_lens()
    print("[lens] " + ("loaded camera_calib.json — undistorting" if lens is not None
                       else "WARNING: no camera_calib.json, frames NOT undistorted"))
    ctl(auto_exposure=1, gain=GAIN)
    best = None
    for ev in EXPO_TRY:
        ctl(exposure_time_absolute=ev)
        for _ in range(8): cap.read()
        ok, f = cap.read()
        if not ok: continue
        f = undistort(f, lens)            # straighten BEFORE detect/click
        c, ids, _ = det.detectMarkers(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY))
        n = 0 if ids is None else len(ids)
        print(f"  ev{ev}: {n} markers")
        if best is None or n > best[0]:
            best = (n, ev, f, c, ids)
    cap.release()
    return best  # (n, ev, frame, corners, ids)

def main():
    n, ev, frame, corners, ids = grab_best()
    if ids is None or n < 4:
        print(f"[warn] only {n} markers detected — reposition/adjust and retry "
              f"(need all 4 visible).")
    markers = {int(i): corners[k][0] for k, i in enumerate(ids.flatten())} if ids is not None else {}
    print(f"[grab] ev{ev}, markers detected: {sorted(markers)}")

    clicks = []
    disp = frame.copy()
    win = "calibrate: click TL, TR, BR, BL"

    def redraw():
        d = frame.copy()
        # draw markers
        if ids is not None:
            cv2.aruco.drawDetectedMarkers(d, corners, ids)
        for j, p in enumerate(clicks):
            cv2.circle(d, tuple(int(v) for v in p), 8, (0,255,0), -1)
            cv2.putText(d, CORNER_NAMES[j], (int(p[0])+10,int(p[1])),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2)
        if len(clicks) == 4:
            cv2.polylines(d, [np.array(clicks, np.int32)], True, (0,255,255), 2)
        nxt = CORNER_NAMES[len(clicks)] if len(clicks) < 4 else "DONE - press s to save"
        cv2.putText(d, f"click: {nxt}", (20,40), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0,255,255), 2)
        cv2.imshow(win, d)

    def on_mouse(ev_, x, y, flags, param):
        if ev_ == cv2.EVENT_LBUTTONDOWN and len(clicks) < 4:
            clicks.append([float(x), float(y)]); redraw()

    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 1280, 720)
    cv2.setMouseCallback(win, on_mouse)
    redraw()
    while True:
        k = cv2.waitKey(20) & 0xFF
        if k == ord('q'):
            print("[quit] not saved"); break
        if k == ord('r'):
            clicks.clear(); redraw()
        if k == ord('s'):
            if len(clicks) != 4:
                print("[!] need exactly 4 corner clicks"); continue
            H = cv2.getPerspectiveTransform(np.array(clicks, np.float32), NORM_CORNERS)
            out = {"markers": {}, "image_size": [frame.shape[1], frame.shape[0]],
                   "table_corners_image": clicks, "exposure": ev, "marker_color": "yellow"}
            for mid, quad in markers.items():
                normq = cv2.perspectiveTransform(quad.reshape(1,-1,2).astype(np.float32), H).reshape(-1,2)
                out["markers"][str(mid)] = normq.tolist()
            with open("board_calib.json", "w") as fjs:
                json.dump(out, fjs, indent=2)
            cv2.imwrite("board_calib_ref.png", frame)
            print(f"[saved] board_calib.json  ({len(markers)} markers), board_calib_ref.png")
            break
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
