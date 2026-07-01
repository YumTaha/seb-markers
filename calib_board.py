"""Shared ChArUco board definition for camera (lens-distortion) calibration.

Used by generate (this file's __main__), capture_calib.py and compute_calib.py so
all three agree on the exact board. The board is a chessboard with ArUco markers in
the white squares: it gives many precisely-spaced corners, and ChArUco tolerates the
board being partly out of frame or steeply angled.

NOTE: this board uses DICT_5X5_100 — deliberately DIFFERENT from the table markers
(DICT_4X4_50) so there's no confusion. Calibrate the lens with this board only.
"""
import cv2

SQUARES_X = 5
SQUARES_Y = 7
SQUARE_LEN = 0.035     # metres (print at 100%; exact size not critical for undistortion)
MARKER_LEN = 0.026
DICT = cv2.aruco.DICT_5X5_100

def make_board():
    d = cv2.aruco.getPredefinedDictionary(DICT)
    return cv2.aruco.CharucoBoard((SQUARES_X, SQUARES_Y), SQUARE_LEN, MARKER_LEN, d)

def make_detector(board):
    return cv2.aruco.CharucoDetector(board)

if __name__ == "__main__":
    # render printable PNG at ~300 DPI for letter paper
    board = make_board()
    dpi = 300
    px_per_m = dpi / 0.0254
    w = int(SQUARES_X * SQUARE_LEN * px_per_m)   # 5*35mm
    h = int(SQUARES_Y * SQUARE_LEN * px_per_m)   # 7*35mm
    margin = 80
    img = board.generateImage((w + 2*margin, h + 2*margin), marginSize=margin, borderBits=1)
    cv2.imwrite("charuco_5x7.png", img)
    print(f"saved charuco_5x7.png  ({w+2*margin}x{h+2*margin}px, board {SQUARES_X}x{SQUARES_Y}, "
          f"{SQUARE_LEN*1000:.0f}mm squares)")
    print("PRINT AT 100% / ACTUAL SIZE (not 'fit to page'); a square should measure 35 mm.")
