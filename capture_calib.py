"""Capture ChArUco views for lens calibration.  RUN FROM THE RDP DESKTOP:

    cd ~/seb-markers && .venv/bin/python capture_calib.py

Live preview shows detected board corners + a 3x3 COVERAGE GRID. Goal: get the
board (well-detected, green corners) into every one of the 9 cells, at varied tilt
and distance. Press SPACE to save the current view; the grid cell lights up when a
view's board-center falls in it. ~2-3 captures per cell (≈20 total) is plenty.
Keys: SPACE = save view, u = undo last, q = done.  Frames saved to calib_shots/.
"""
from __future__ import annotations
import os
os.environ.setdefault("QT_QPA_PLATFORM", "xcb")
import cv2, numpy as np, subprocess, glob
from calib_board import make_board, make_detector

CAM = 0
OUT = "calib_shots"
MIN_CORNERS = 8     # need a decent number of charuco corners to keep a view

def main():
    os.makedirs(OUT, exist_ok=True)
    existing = len(glob.glob(f"{OUT}/*.png"))
    board = make_board(); detector = make_detector(board)

    cap = cv2.VideoCapture(CAM, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    subprocess.run(["v4l2-ctl","-d",f"/dev/video{CAM}","-c","auto_exposure=3"], capture_output=True)
    if not cap.isOpened():
        print("[error] cannot open camera"); return

    win = "lens calibration capture"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL); cv2.resizeWindow(win, 1280, 720)
    coverage = np.zeros((3,3), int)   # how many saved views centered in each cell
    saved = existing
    saved_centers = []

    while True:
        ok, frame = cap.read()
        if not ok: continue
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ch_corners, ch_ids, _, _ = detector.detectBoard(gray)
        n = 0 if ch_ids is None else len(ch_ids)
        center = None
        disp = frame.copy()
        if n > 0:
            cv2.aruco.drawDetectedCornersCharuco(disp, ch_corners, ch_ids, (0,255,0))
            center = ch_corners.reshape(-1,2).mean(axis=0)

        # 3x3 coverage grid overlay
        for i in range(1,3):
            cv2.line(disp,(w*i//3,0),(w*i//3,h),(80,80,80),1)
            cv2.line(disp,(0,h*i//3),(w,h*i//3),(80,80,80),1)
        for gy in range(3):
            for gx in range(3):
                c = coverage[gy,gx]
                col = (0,180,0) if c>=2 else (0,180,180) if c==1 else (60,60,60)
                cx,cy = w*gx//3+30, h*gy//3+40
                cv2.putText(disp,str(c),(cx,cy),cv2.FONT_HERSHEY_SIMPLEX,1.0,col,2)
        msg = f"corners:{n}  saved:{saved}   SPACE=save  u=undo  q=done"
        cv2.putText(disp,msg,(20,h-25),cv2.FONT_HERSHEY_SIMPLEX,0.9,
                    (0,255,0) if n>=MIN_CORNERS else (0,0,255),2)
        cv2.imshow(win, disp)

        k = cv2.waitKey(20) & 0xFF
        if k == ord('q'): break
        if k == ord('u') and saved_centers:
            f = saved_centers.pop()
            os.remove(f["path"]); coverage[f["cell"]] -= 1; saved -= 1
            print(f"[undo] removed {f['path']}")
        if k == ord(' '):
            if n < MIN_CORNERS:
                print(f"[skip] only {n} corners (<{MIN_CORNERS}) — get the board clearer/closer")
                continue
            path = f"{OUT}/shot_{saved:03d}.png"; cv2.imwrite(path, frame)
            gx = min(2, int(center[0]*3//w)); gy = min(2, int(center[1]*3//h))
            coverage[gy,gx]+=1; saved+=1
            saved_centers.append({"path":path,"cell":(gy,gx)})
            print(f"[save] {path}  cell=({gy},{gx})  corners={n}")

    cap.release(); cv2.destroyAllWindows()
    cells_covered = int((coverage>0).sum())
    print(f"\nsaved {saved} views; {cells_covered}/9 grid cells covered.")
    if cells_covered < 9 or saved < 12:
        print("Recommend more: aim for all 9 cells and >=15 views before calibrating.")
    print("Next:  .venv/bin/python compute_calib.py")

if __name__ == "__main__":
    main()
