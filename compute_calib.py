"""Compute lens calibration from captured ChArUco views -> camera_calib.json.

    cd ~/seb-markers && .venv/bin/python compute_calib.py

Reads calib_shots/*.png, detects ChArUco corners, runs cv2.calibrateCamera, and
saves the camera matrix + distortion coefficients. Reports RMS reprojection error
(want < ~1.0 px; < 0.5 is great). Also writes undistort_preview.png so you can see
the arc straightened.
"""
from __future__ import annotations
import cv2, numpy as np, glob, json, sys
from calib_board import make_board, make_detector

def main():
    board = make_board(); detector = make_detector(board)
    chess = board.getChessboardCorners()        # (Ncorners,3) object coords
    files = sorted(glob.glob("calib_shots/*.png"))
    if len(files) < 8:
        print(f"[error] only {len(files)} shots — capture more with capture_calib.py"); sys.exit(1)

    obj_pts, img_pts, size = [], [], None
    used = 0
    for f in files:
        img = cv2.imread(f); gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        size = gray.shape[::-1]
        ch_corners, ch_ids, _, _ = detector.detectBoard(gray)
        if ch_ids is None or len(ch_ids) < 6:
            print(f"  {f}: only {0 if ch_ids is None else len(ch_ids)} corners — skipped"); continue
        ids = ch_ids.flatten()
        obj_pts.append(chess[ids].astype(np.float32))
        img_pts.append(ch_corners.reshape(-1,2).astype(np.float32))
        used += 1
    print(f"using {used}/{len(files)} views")
    if used < 8:
        print("[error] too few usable views"); sys.exit(1)

    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(obj_pts, img_pts, size, None, None)
    print(f"\nRMS reprojection error: {rms:.3f} px  "
          f"({'great' if rms<0.5 else 'good' if rms<1.0 else 'high — recapture with more variety'})")
    print("camera matrix:\n", np.round(K,2))
    print("dist coeffs:", np.round(dist.ravel(),4))

    out = {"camera_matrix": K.tolist(), "dist_coeffs": dist.ravel().tolist(),
           "image_size": list(size), "rms_error": float(rms), "n_views": used}
    with open("camera_calib.json","w") as fjs:
        json.dump(out, fjs, indent=2)
    print("[saved] camera_calib.json")

    # visual proof: undistort one shot
    sample = cv2.imread(files[0])
    newK, _ = cv2.getOptimalNewCameraMatrix(K, dist, size, 1, size)
    und = cv2.undistort(sample, K, dist, None, newK)
    combo = np.hstack([cv2.resize(sample,(640,360)), cv2.resize(und,(640,360))])
    cv2.putText(combo,"original",(20,30),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,0,255),2)
    cv2.putText(combo,"undistorted",(660,30),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,0),2)
    cv2.imwrite("undistort_preview.png", combo)
    print("[saved] undistort_preview.png  (left=original, right=straightened)")

if __name__ == "__main__":
    main()
