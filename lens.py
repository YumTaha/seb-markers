"""Shared lens-undistortion helper. Load once, apply to every frame before any
marker detection / homography so all geometry lives in straightened space.
"""
import cv2, numpy as np, json, os

def load_lens(path="camera_calib.json"):
    if not os.path.exists(path):
        return None
    d = json.load(open(path))
    return (np.array(d["camera_matrix"], np.float64),
            np.array(d["dist_coeffs"], np.float64))

def undistort(frame, lens):
    """Undistort keeping the original camera matrix (same size, comparable coords)."""
    if lens is None:
        return frame
    K, dist = lens
    return cv2.undistort(frame, K, dist, None, K)
