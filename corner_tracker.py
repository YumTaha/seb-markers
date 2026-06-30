from __future__ import annotations

import collections
import json
import math
import os
import re
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import Optional

os.environ.setdefault("QT_QPA_PLATFORM", "xcb")

import cv2
import numpy as np

# ── Constants ──────────────────────────────────────────────────────────────────

ARUCO_DICT_ID = cv2.aruco.DICT_4X4_50
CORNER_IDS: dict[int, str] = {0: "TL", 1: "TR", 2: "BR", 3: "BL"}
ALL_ROLES = ["TL", "TR", "BR", "BL"]

REAL_WORLD: dict[str, np.ndarray] = {
    "TL": np.array([0.0, 0.0]),
    "TR": np.array([1.0, 0.0]),
    "BR": np.array([1.0, 1.0]),
    "BL": np.array([0.0, 1.0]),
}

CAM_INDEX = 0
CAM_WIDTH  = 1920
CAM_HEIGHT = 1080
SNAPSHOT_FRAMES = 20   # frames averaged per saved photo
CORNER_SMOOTH_ALPHA = 0.4   # EMA weight for new detection (lower = smoother)
MIN_AREA = 10000
MIN_MARKER_DIST = 50
BSP_SPLIT_MARGIN = 0.04   # min distance from cell edge to allow a split
DETECT_EVERY_N = 3        # in TRACKING mode, only run ArUco detection every Nth frame
CALIB_FILE = Path(__file__).parent / "calibration.json"

# BGR colors
GREEN  = (0, 200, 0)
YELLOW = (0, 210, 210)
GRAY   = (140, 140, 140)
RED    = (0, 0, 220)
WHITE  = (255, 255, 255)
BLACK  = (0, 0, 0)
CYAN   = (220, 200, 0)
LIGHT_BLUE = (235, 200, 130)   # BGR
LIGHT_RED  = (130, 130, 235)   # BGR

# Layer panel
LAYER_KEYS   = ["raw", "saturation", "rectangles"]
LAYER_LABELS = {
    "raw":        "Raw",
    "saturation": "Saturation",
    "rectangles": "Rectangles",
}
LAYER_ROW_H = 30
PANEL_W     = 145

# Target color detection: name → list of (HSV_lo, HSV_hi) ranges
TARGET_COLORS: dict[str, list[tuple[tuple, tuple]]] = {
    "green":  [((35,  60, 60), (85,  255, 255))],
    "red":    [((0,   80, 80), (10,  255, 255)), ((170, 80, 80), (180, 255, 255))],
    "blue":   [((100, 80, 60), (130, 255, 255))],
    "yellow": [((20,  80, 80), (33,  255, 255))],
}
TARGET_NAMES = list(TARGET_COLORS.keys())
TINT_TEXT: dict[str, tuple] = {   # BGR label color per target
    "green":  (0, 220, 0),
    "red":    (0, 0, 220),
    "blue":   (235, 130, 0),
    "yellow": (0, 210, 210),
}
LOW_PCT_THRESHOLD = 5.0   # pct below this → low tint


# ── Enums / dataclasses ────────────────────────────────────────────────────────

class Mode(Enum):
    CALIBRATION = auto()   # waiting for marker quad
    BSP_BUILD   = auto()   # markers calibrated, drawing section splits
    TRACKING    = auto()   # live tracking + section overlay


class Confidence(str, Enum):
    DETECTED = "DETECTED"
    INFERRED = "INFERRED"
    GHOST    = "GHOST"


@dataclass
class CornerResult:
    pixel: Optional[tuple[float, float]]
    confidence: Confidence


@dataclass
class BSPNode:
    """BSP tree node. Bounds are absolute in normalized [0,1]×[0,1] space."""
    x0: float
    y0: float
    x1: float
    y1: float
    axis: Optional[str] = None      # 'h' (horizontal) or 'v' (vertical); None = leaf
    pos: float = 0.0                # absolute split coord within [0,1]
    left: Optional['BSPNode'] = None
    right: Optional['BSPNode'] = None
    cell_id: Optional[int] = None   # assigned on SPACE, leaves only

    def is_leaf(self) -> bool:
        return self.left is None


@dataclass
class CalibData:
    corners: dict[str, tuple[float, float]]
    bsp: Optional[BSPNode] = None


# ── BSP operations ─────────────────────────────────────────────────────────────

def bsp_find_leaf(node: BSPNode, nx: float, ny: float) -> Optional[BSPNode]:
    if not (node.x0 <= nx <= node.x1 and node.y0 <= ny <= node.y1):
        return None
    if node.is_leaf():
        return node
    if node.axis == 'h':
        child = node.left if ny < node.pos else node.right
    else:
        child = node.left if nx < node.pos else node.right
    return bsp_find_leaf(child, nx, ny)


def bsp_split(leaf: BSPNode, axis: str, pos: float) -> None:
    """Split a leaf in-place. pos is absolute normalized coordinate."""
    leaf.axis = axis
    leaf.pos  = pos
    if axis == 'h':
        leaf.left  = BSPNode(leaf.x0, leaf.y0, leaf.x1, pos)
        leaf.right = BSPNode(leaf.x0, pos,     leaf.x1, leaf.y1)
    else:
        leaf.left  = BSPNode(leaf.x0, leaf.y0, pos,     leaf.y1)
        leaf.right = BSPNode(pos,     leaf.y0, leaf.x1, leaf.y1)


def bsp_get_leaves(root: BSPNode) -> list[BSPNode]:
    if root.is_leaf():
        return [root]
    return bsp_get_leaves(root.left) + bsp_get_leaves(root.right)


def bsp_assign_ids(root: BSPNode) -> None:
    """Assign sequential IDs in scan order (top→bottom, left→right)."""
    leaves = bsp_get_leaves(root)
    leaves.sort(key=lambda n: ((n.y0 + n.y1) / 2, (n.x0 + n.x1) / 2))
    for i, leaf in enumerate(leaves):
        leaf.cell_id = i


def bsp_to_dict(node: BSPNode) -> dict:
    return {
        "x0": node.x0, "y0": node.y0, "x1": node.x1, "y1": node.y1,
        "axis": node.axis, "pos": node.pos, "cell_id": node.cell_id,
        "left":  bsp_to_dict(node.left)  if node.left  else None,
        "right": bsp_to_dict(node.right) if node.right else None,
    }


def bsp_from_dict(d: dict) -> BSPNode:
    node = BSPNode(
        x0=d["x0"], y0=d["y0"], x1=d["x1"], y1=d["y1"],
        axis=d.get("axis"), pos=d.get("pos", 0.0), cell_id=d.get("cell_id"),
    )
    if d.get("left"):
        node.left  = bsp_from_dict(d["left"])
    if d.get("right"):
        node.right = bsp_from_dict(d["right"])
    return node


# ── Calibration I/O ────────────────────────────────────────────────────────────

def save_calibration(
    corners: dict[str, tuple[float, float]],
    bsp: Optional[BSPNode] = None,
) -> None:
    data: dict = {
        "corners":    {k: [float(v[0]), float(v[1])] for k, v in corners.items()},
        "real_world": {k: list(v.tolist()) for k, v in REAL_WORLD.items()},
    }
    if bsp is not None:
        data["bsp"] = bsp_to_dict(bsp)
    CALIB_FILE.write_text(json.dumps(data, indent=2))
    print(f"[calib] saved → {CALIB_FILE}")


def load_calibration() -> Optional[CalibData]:
    if not CALIB_FILE.exists():
        return None
    try:
        data = json.loads(CALIB_FILE.read_text())
        corners = {k: (float(v[0]), float(v[1])) for k, v in data["corners"].items()}
        if not all(r in corners for r in ALL_ROLES):
            raise ValueError("missing roles")
        bsp = bsp_from_dict(data["bsp"]) if data.get("bsp") else None
        return CalibData(corners=corners, bsp=bsp)
    except Exception as exc:
        print(f"[warn] calibration.json unreadable ({exc}), re-entering CALIBRATION mode")
        return None


# ── Sanity checks ──────────────────────────────────────────────────────────────

def quad_is_valid(corners: dict[str, tuple[float, float]]) -> bool:
    pts = np.array([corners[r] for r in ALL_ROLES], dtype=np.float32).reshape(-1, 1, 2)
    if len(cv2.convexHull(pts, returnPoints=True)) != 4:
        return False
    if cv2.contourArea(pts) < MIN_AREA:
        return False
    flat = pts.reshape(4, 2)
    for i in range(4):
        for j in range(i + 1, 4):
            if float(np.linalg.norm(flat[i] - flat[j])) < MIN_MARKER_DIST:
                return False
    return True


# ── Detection ─────────────────────────────────────────────────────────────────

def detect_markers(
    frame: np.ndarray,
    detector: cv2.aruco.ArucoDetector,
) -> dict[str, tuple[float, float]]:
    corners, ids, _ = detector.detectMarkers(frame)
    detected: dict[str, tuple[float, float]] = {}
    if ids is None:
        return detected
    for marker_corners, marker_id in zip(corners, ids.flatten()):
        mid = int(marker_id)
        if mid in CORNER_IDS:
            cx, cy = marker_corners[0].mean(axis=0).tolist()
            detected[CORNER_IDS[mid]] = (float(cx), float(cy))
    return detected


# ── Inference ─────────────────────────────────────────────────────────────────

def _similarity_2pts(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    p1, p2 = src[0], src[1]
    q1, q2 = dst[0], dst[1]
    dx_s, dy_s = p2[0] - p1[0], p2[1] - p1[1]
    dx_d, dy_d = q2[0] - q1[0], q2[1] - q1[1]
    denom = dx_s ** 2 + dy_s ** 2
    if denom < 1e-10:
        return np.array([[1, 0, q1[0] - p1[0]], [0, 1, q1[1] - p1[1]]], dtype=np.float64)
    a = (dx_s * dx_d + dy_s * dy_d) / denom
    b = (dx_s * dy_d - dy_s * dx_d) / denom
    tx = q1[0] - a * p1[0] + b * p1[1]
    ty = q1[1] - b * p1[0] - a * p1[1]
    return np.array([[a, -b, tx], [b, a, ty]], dtype=np.float64)


def _apply_affine(M: np.ndarray, pt: tuple[float, float]) -> tuple[float, float]:
    v = np.array([pt[0], pt[1], 1.0])
    r = M @ v
    return (float(r[0]), float(r[1]))


def infer_missing_corners(
    detected: dict[str, tuple[float, float]],
    calib: CalibData,
) -> dict[str, CornerResult]:
    n = len(detected)
    result: dict[str, CornerResult] = {}

    if n == 4:
        for role in ALL_ROLES:
            result[role] = CornerResult(pixel=detected[role], confidence=Confidence.DETECTED)
        return result

    if n == 3:
        visible = [r for r in ALL_ROLES if r in detected]
        src = np.array([REAL_WORLD[r] for r in visible], dtype=np.float32)
        dst = np.array([detected[r] for r in visible], dtype=np.float32)
        M = cv2.getAffineTransform(src, dst)
        for role in ALL_ROLES:
            if role in detected:
                result[role] = CornerResult(pixel=detected[role], confidence=Confidence.DETECTED)
            else:
                px = _apply_affine(M, tuple(REAL_WORLD[role].tolist()))
                result[role] = CornerResult(pixel=px, confidence=Confidence.INFERRED)
        return result

    if n == 2:
        visible = [r for r in ALL_ROLES if r in detected]
        calib_src = np.array([calib.corners[r] for r in visible], dtype=np.float64)
        curr_dst  = np.array([detected[r] for r in visible], dtype=np.float64)
        M = _similarity_2pts(calib_src, curr_dst)
        for role in ALL_ROLES:
            if role in detected:
                result[role] = CornerResult(pixel=detected[role], confidence=Confidence.DETECTED)
            else:
                px = _apply_affine(M, calib.corners[role])
                result[role] = CornerResult(pixel=px, confidence=Confidence.INFERRED)
        return result

    for role in ALL_ROLES:
        if role in detected:
            result[role] = CornerResult(pixel=detected[role], confidence=Confidence.DETECTED)
        else:
            result[role] = CornerResult(pixel=None, confidence=Confidence.GHOST)
    return result


# ── Homography helpers ─────────────────────────────────────────────────────────

def compute_homographies(
    corners: dict[str, tuple[float, float]],
) -> tuple[np.ndarray, np.ndarray]:
    """Return (H: pixel→norm, H_inv: norm→pixel)."""
    src = np.array([corners[r] for r in ALL_ROLES], dtype=np.float32)
    dst = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32)
    H     = cv2.getPerspectiveTransform(src, dst)
    H_inv = cv2.getPerspectiveTransform(dst, src)
    return H, H_inv


def transform_pt(H: np.ndarray, pt: tuple[float, float]) -> tuple[float, float]:
    v = np.array([[[float(pt[0]), float(pt[1])]]], dtype=np.float32)
    r = cv2.perspectiveTransform(v, H)
    return (float(r[0, 0, 0]), float(r[0, 0, 1]))


def norm_to_px(pts_norm: list[tuple[float, float]], H_inv: np.ndarray) -> np.ndarray:
    arr = np.array(pts_norm, dtype=np.float32).reshape(1, -1, 2)
    return cv2.perspectiveTransform(arr, H_inv).reshape(-1, 2)


# ── Image utils ───────────────────────────────────────────────────────────────

def make_quad_mask(
    corners: dict[str, tuple[float, float]],
    shape: tuple,
) -> np.ndarray:
    pts = np.array([_ipt(corners[r]) for r in ALL_ROLES], dtype=np.int32)
    mask = np.zeros(shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [pts], 255)
    return mask


def change_saturation(
    image: np.ndarray,
    scale: float = 1.5,
    mask: Optional[np.ndarray] = None,
    frame_hsv: Optional[np.ndarray] = None,
) -> np.ndarray:
    hsv = (frame_hsv if frame_hsv is not None
           else cv2.cvtColor(image, cv2.COLOR_BGR2HSV)).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * scale, 0, 255)
    saturated = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    if mask is None:
        return saturated
    return np.where(mask[:, :, np.newaxis] > 0, saturated, image)



# ── Drawing helpers ────────────────────────────────────────────────────────────

def _ipt(p: tuple[float, float]) -> tuple[int, int]:
    return (int(round(p[0])), int(round(p[1])))


def draw_dashed_line(
    img: np.ndarray,
    pt1: tuple[float, float],
    pt2: tuple[float, float],
    color: tuple,
    thickness: int = 2,
    dash_len: int = 12,
    gap_len: int = 8,
) -> None:
    x1, y1 = float(pt1[0]), float(pt1[1])
    x2, y2 = float(pt2[0]), float(pt2[1])
    dx, dy = x2 - x1, y2 - y1
    length = math.hypot(dx, dy)
    if length < 1:
        return
    ux, uy = dx / length, dy / length
    pos = 0.0
    drawing = True
    while pos < length:
        seg = dash_len if drawing else gap_len
        end = min(pos + seg, length)
        if drawing:
            sx, sy = int(x1 + ux * pos), int(y1 + uy * pos)
            ex, ey = int(x1 + ux * end), int(y1 + uy * end)
            cv2.line(img, (sx, sy), (ex, ey), color, thickness)
        pos = end
        drawing = not drawing


def draw_text_outlined(
    img: np.ndarray,
    text: str,
    org: tuple[int, int],
    scale: float = 0.5,
    fg: tuple = WHITE,
    bg: tuple = BLACK,
    thickness: int = 1,
) -> None:
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, bg, thickness + 2)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, fg, thickness)


# ── Layer panel ───────────────────────────────────────────────────────────────

def _draw_eye(img: np.ndarray, cx: int, cy: int, open_: bool) -> None:
    if open_:
        cv2.ellipse(img, (cx, cy), (11, 6), 0, 0, 360, WHITE, 1, cv2.LINE_AA)
        cv2.circle(img, (cx, cy), 3, WHITE, -1, cv2.LINE_AA)
    else:
        cv2.ellipse(img, (cx, cy), (11, 6), 0, 0, 360, GRAY, 1, cv2.LINE_AA)
        cv2.line(img, (cx - 11, cy), (cx + 11, cy), GRAY, 2, cv2.LINE_AA)


def _panel_origin(frame_w: int, frame_h: int) -> tuple[int, int]:
    return (frame_w - PANEL_W - 10, 10)


def draw_layer_panel(img: np.ndarray, layers: dict[str, bool]) -> None:
    h, w = img.shape[:2]
    px, py = _panel_origin(w, h)
    panel_h = len(LAYER_KEYS) * LAYER_ROW_H + 8
    # semi-transparent background
    overlay = img.copy()
    cv2.rectangle(overlay, (px - 4, py - 4), (px + PANEL_W, py + panel_h), BLACK, -1)
    cv2.addWeighted(overlay, 0.45, img, 0.55, 0, img)
    cv2.rectangle(img, (px - 4, py - 4), (px + PANEL_W, py + panel_h), GRAY, 1)
    for i, key in enumerate(LAYER_KEYS):
        ry = py + i * LAYER_ROW_H + LAYER_ROW_H // 2
        _draw_eye(img, px + 12, ry, layers[key])
        color = WHITE if layers[key] else GRAY
        draw_text_outlined(img, LAYER_LABELS[key], (px + 30, ry + 5),
                           scale=0.5, fg=color)


def panel_hit_test(mx: int, my: int, frame_w: int, frame_h: int) -> Optional[str]:
    px, py = _panel_origin(frame_w, frame_h)
    for i, key in enumerate(LAYER_KEYS):
        row_y = py + i * LAYER_ROW_H
        if px - 4 <= mx <= px + PANEL_W and row_y <= my <= row_y + LAYER_ROW_H:
            return key
    return None


# ── BSP drawing ────────────────────────────────────────────────────────────────

def draw_bsp_overlay(
    frame: np.ndarray,
    root: BSPNode,
    H_inv: np.ndarray,
    show_ids: bool,
    hover_leaf: Optional[BSPNode] = None,
    hover_axis: Optional[str] = None,
    hover_nx: float = 0.0,
    hover_ny: float = 0.0,
    source_frame: Optional[np.ndarray] = None,
    target_ranges: Optional[list[tuple[tuple, tuple]]] = None,
    source_hsv: Optional[np.ndarray] = None,
) -> None:
    color_mask: Optional[np.ndarray] = None
    if target_ranges and (source_frame is not None or source_hsv is not None):
        if source_hsv is None:
            source_hsv = cv2.cvtColor(source_frame, cv2.COLOR_BGR2HSV)
        color_mask = np.zeros(source_hsv.shape[:2], dtype=np.uint8)
        for lo, hi in target_ranges:
            color_mask = cv2.bitwise_or(color_mask, cv2.inRange(source_hsv, lo, hi))

    for leaf in bsp_get_leaves(root):
        cell_corners = [
            (leaf.x0, leaf.y0), (leaf.x1, leaf.y0),
            (leaf.x1, leaf.y1), (leaf.x0, leaf.y1),
        ]
        px_corners = norm_to_px(cell_corners, H_inv).astype(np.int32)
        cv2.polylines(frame, [px_corners.reshape(-1, 1, 2)], True, WHITE, 1)

        if show_ids and leaf.cell_id is not None:
            cx_n = (leaf.x0 + leaf.x1) / 2
            cy_n = (leaf.y0 + leaf.y1) / 2
            cp = norm_to_px([(cx_n, cy_n)], H_inv).reshape(2)
            cx_px, cy_px = int(cp[0]), int(cp[1])
            draw_text_outlined(frame, str(leaf.cell_id),
                               (cx_px - 10, cy_px + 10), scale=0.8)

            if color_mask is not None:
                h_img, w_img = color_mask.shape[:2]
                mask = np.zeros((h_img, w_img), dtype=np.uint8)
                cv2.fillPoly(mask, [px_corners.reshape(-1, 1, 2)], 255)
                hit_in_cell = cv2.bitwise_and(color_mask, mask)
                total = cv2.countNonZero(mask)
                hit_count = cv2.countNonZero(hit_in_cell)
                pct = 100.0 * hit_count / total if total else 0.0
                tint = LIGHT_BLUE if pct < LOW_PCT_THRESHOLD else LIGHT_RED
                overlay = frame.copy()
                cv2.fillPoly(overlay, [px_corners.reshape(-1, 1, 2)], tint)
                cv2.addWeighted(overlay, 0.35, frame, 0.65, 0, frame)
                draw_text_outlined(frame, f"{pct:.1f}%",
                                   (cx_px - 20, cy_px + 32), scale=0.5, fg=WHITE)

        if leaf is hover_leaf:
            overlay = frame.copy()
            cv2.fillPoly(overlay, [px_corners.reshape(-1, 1, 2)], CYAN)
            cv2.addWeighted(overlay, 0.12, frame, 0.88, 0, frame)

    # Construction line preview
    if hover_leaf is not None and hover_axis is not None:
        if hover_axis == 'h':
            p1_n = (hover_leaf.x0, hover_ny)
            p2_n = (hover_leaf.x1, hover_ny)
        else:
            p1_n = (hover_nx, hover_leaf.y0)
            p2_n = (hover_nx, hover_leaf.y1)
        pts_px = norm_to_px([p1_n, p2_n], H_inv)
        draw_dashed_line(frame, tuple(pts_px[0]), tuple(pts_px[1]), CYAN, 2, 8, 5)


# ── Quad overlay ───────────────────────────────────────────────────────────────

def draw_quad_overlay(
    frame: np.ndarray,
    result: dict[str, CornerResult],
    last_known: Optional[dict[str, tuple[float, float]]],
) -> None:
    pixels = {r: result[r].pixel for r in ALL_ROLES}
    confs  = {r: result[r].confidence for r in ALL_ROLES}
    n_detected = sum(1 for r in ALL_ROLES if confs[r] == Confidence.DETECTED)
    n_valid    = sum(1 for r in ALL_ROLES if pixels[r] is not None)

    if n_valid == 0 or n_detected <= 1:
        if last_known:
            ghost_pts = np.array([_ipt(last_known[r]) for r in ALL_ROLES], dtype=np.int32)
            overlay = frame.copy()
            cv2.fillPoly(overlay, [ghost_pts], GRAY)
            cv2.polylines(overlay, [ghost_pts], True, GRAY, 2)
            cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
            cx = int(np.mean([last_known[r][0] for r in ALL_ROLES]))
            cy = int(np.mean([last_known[r][1] for r in ALL_ROLES]))
            cv2.putText(frame, "LOST", (cx - 45, cy + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.6, BLACK, 6)
            cv2.putText(frame, "LOST", (cx - 45, cy + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.6, RED, 3)
        return

    if n_detected == 4:
        pts = np.array([_ipt(pixels[r]) for r in ALL_ROLES], dtype=np.int32)
        cv2.polylines(frame, [pts], True, GREEN, 2)
        for r in ALL_ROLES:
            cv2.circle(frame, _ipt(pixels[r]), 8, GREEN, -1)
    else:
        n = len(ALL_ROLES)
        for i in range(n):
            r1 = ALL_ROLES[i]
            r2 = ALL_ROLES[(i + 1) % n]
            p1, p2 = pixels[r1], pixels[r2]
            if p1 is None or p2 is None:
                continue
            if confs[r1] == Confidence.DETECTED and confs[r2] == Confidence.DETECTED:
                cv2.line(frame, _ipt(p1), _ipt(p2), GREEN, 2)
            else:
                draw_dashed_line(frame, p1, p2, YELLOW, 2)
        for r in ALL_ROLES:
            if pixels[r] is None:
                continue
            cv2.circle(frame, _ipt(pixels[r]), 8,
                       GREEN if confs[r] == Confidence.DETECTED else YELLOW, -1)

    for r in ALL_ROLES:
        if pixels[r] is None:
            continue
        ox, oy = int(pixels[r][0]) + 10, int(pixels[r][1]) - 10
        draw_text_outlined(frame, f"{r} [{confs[r].value[:3]}]", (ox, oy), scale=0.5)


# ── HUD ───────────────────────────────────────────────────────────────────────

def draw_hud(
    frame: np.ndarray,
    mode: Mode,
    n_visible: int,
    frame_num: int,
    n_cells: int = 0,
    fps: float = 0.0,
) -> None:
    lines = [f"Mode: {mode.name}", f"Visible: {n_visible}/4", f"Frame: {frame_num}", f"FPS: {fps:.1f}"]
    if mode == Mode.BSP_BUILD:
        lines += [
            f"Cells: {n_cells}",
            "Click=split H | Shift+Click=split V",
            "SPACE=finalize  R=reset all",
        ]
    for i, line in enumerate(lines):
        draw_text_outlined(frame, line, (10, 30 + i * 28), scale=0.6)


def draw_calib_status(
    frame: np.ndarray,
    detected: dict[str, tuple[float, float]],
    quad_ready: bool,
) -> None:
    h, w = frame.shape[:2]
    if detected:
        pts_arr = np.array([detected[r] for r in ALL_ROLES if r in detected], dtype=np.int32)
        if len(pts_arr) == 4:
            color = GREEN if quad_ready else YELLOW
            overlay = frame.copy()
            cv2.fillPoly(overlay, [pts_arr.reshape(-1, 1, 2)], color)
            cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)
            cv2.polylines(frame, [pts_arr.reshape(-1, 1, 2)], True, color, 2)
        for r, px in detected.items():
            color = GREEN if quad_ready else YELLOW
            cv2.circle(frame, _ipt(px), 8, color, -1)
            draw_text_outlined(frame, r, (_ipt(px)[0] + 10, _ipt(px)[1] - 10), scale=0.6)

    if quad_ready:
        msg, fg = "SPACE to begin section editor", GREEN
    elif len(detected) == 4:
        msg, fg = "Quad failed sanity check", RED
    else:
        msg, fg = f"Waiting for all 4 markers... ({len(detected)}/4)", YELLOW

    tx, ty = w // 2 - 240, h - 30
    cv2.putText(frame, msg, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.8, BLACK, 4)
    cv2.putText(frame, msg, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.8, fg, 2)


# ── Result dict ────────────────────────────────────────────────────────────────

def build_result_dict(result: dict[str, CornerResult]) -> dict:
    return {
        role: {"pixel": result[role].pixel, "confidence": result[role].confidence.value}
        for role in ALL_ROLES
    }


# ── Corner smoother ───────────────────────────────────────────────────────────

def smooth_corners(
    detected: dict[str, tuple[float, float]],
    prev: dict[str, tuple[float, float]],
    alpha: float = CORNER_SMOOTH_ALPHA,
) -> dict[str, tuple[float, float]]:
    """EMA blend of new detections onto previous positions."""
    out: dict[str, tuple[float, float]] = {}
    for role, pt in detected.items():
        if role in prev:
            px, py = prev[role]
            out[role] = (alpha * pt[0] + (1 - alpha) * px,
                         alpha * pt[1] + (1 - alpha) * py)
        else:
            out[role] = pt
    return out


# ── Mouse callback ─────────────────────────────────────────────────────────────

def make_mouse_callback(state: dict):
    def on_mouse(event, x, y, flags, param):
        state["x"]     = x
        state["y"]     = y
        state["shift"] = bool(flags & cv2.EVENT_FLAG_SHIFTKEY)
        if event == cv2.EVENT_LBUTTONDOWN:
            state["clicked"] = True
    return on_mouse


def get_screen_size() -> tuple[int, int]:
    try:
        out = subprocess.run(["xrandr"], capture_output=True, text=True).stdout
        m = re.search(r"current (\d+) x (\d+)", out)
        if m:
            return int(m.group(1)), int(m.group(2))
    except Exception:
        pass
    return 1920, 1080


# ── Panel click helper ─────────────────────────────────────────────────────────

def _consume_panel_click(
    mouse_state: dict, layers: dict[str, bool], fw: int, fh: int
) -> bool:
    """Toggle layer if click lands on panel. Returns True if consumed."""
    if not mouse_state["clicked"]:
        return False
    hit = panel_hit_test(mouse_state["x"], mouse_state["y"], fw, fh)
    if hit is not None:
        mouse_state["clicked"] = False
        layers[hit] = not layers[hit]
        print(f"[layer] {hit} → {'on' if layers[hit] else 'off'}")
        return True
    return False


# ── Background frame grabber ──────────────────────────────────────────────────

class FrameGrabber:
    """Decodes MJPEG frames in a background thread so the main loop never blocks on cap.read()."""

    def __init__(self, cap: cv2.VideoCapture) -> None:
        self._cap     = cap
        self._frame:  Optional[np.ndarray] = None
        self._lock    = threading.Lock()
        self._running = True
        self._thread  = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        while self._running:
            ret, frame = self._cap.read()
            if ret:
                with self._lock:
                    self._frame = frame

    def read(self) -> tuple[bool, Optional[np.ndarray]]:
        with self._lock:
            if self._frame is None:
                return False, None
            frame, self._frame = self._frame, None
            return True, frame

    def stop(self) -> None:
        self._running = False



def get_screen_size() -> tuple[int, int]:
    try:
        out = subprocess.run(["xrandr"], capture_output=True, text=True).stdout
        m = re.search(r"current (\d+) x (\d+)", out)
        if m:
            return int(m.group(1)), int(m.group(2))
    except Exception:
        pass
    return 1920, 1080


def main() -> None:
    cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
    if not cap.isOpened():
        print(f"[error] cannot open webcam at index {CAM_INDEX}")
        sys.exit(1)

    grabber = FrameGrabber(cap)

    aruco_dict   = cv2.aruco.getPredefinedDictionary(ARUCO_DICT_ID)
    aruco_params = cv2.aruco.DetectorParameters()
    detector     = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

    calib = load_calibration()
    mode  = Mode.TRACKING if calib is not None else Mode.CALIBRATION
    print(f"[init] {'calibration loaded' if calib else 'no calibration'} → {mode.name}")

    # Homographies (pixel↔norm) — computed once per calibration
    H:     Optional[np.ndarray] = None
    H_inv: Optional[np.ndarray] = None
    if calib:
        H, H_inv = compute_homographies(calib.corners)

    # BSP state
    bsp_root: Optional[BSPNode] = calib.bsp if calib else None

    last_known: Optional[dict[str, tuple[float, float]]] = None
    result: dict[str, CornerResult] = {
        r: CornerResult(pixel=None, confidence=Confidence.GHOST) for r in ALL_ROLES
    }
    current_detected: dict[str, tuple[float, float]] = {}
    quad_ready = False
    frame_num  = 0

    mouse_state: dict = {"x": 0, "y": 0, "shift": False, "clicked": False}
    layers: dict[str, bool] = {k: True for k in LAYER_KEYS}
    target_idx = 0   # index into TARGET_NAMES

    scr_w, scr_h = get_screen_size()
    cv2.namedWindow("ArUco Tracker", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("ArUco Tracker", scr_w, scr_h)
    cv2.moveWindow("ArUco Tracker", 0, 0)
    cv2.setMouseCallback("ArUco Tracker", make_mouse_callback(mouse_state))

    _frame_times: collections.deque = collections.deque(maxlen=30)
    _fps = 0.0
    _detect_tick = 0
    _snap_acc:   Optional[np.ndarray] = None
    _snap_count: int = 0

    while True:
        ret, frame = grabber.read()
        if not ret:
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break
            continue

        frame_num += 1

        # Snapshot averaging — accumulate raw frames, save when complete
        if _snap_count > 0:
            _snap_acc = (frame.astype(np.float32) if _snap_acc is None
                         else _snap_acc + frame.astype(np.float32))
            _snap_count -= 1
            if _snap_count == 0:
                averaged = (_snap_acc / SNAPSHOT_FRAMES).clip(0, 255).astype(np.uint8)
                filename = f"frame_{frame_num:04d}.png"
                cv2.imwrite(filename, averaged)
                print(f"[save] {filename}  ({SNAPSHOT_FRAMES}-frame average, "
                      f"{averaged.shape[1]}x{averaged.shape[0]})")
                print(f"[frame {frame_num}] {build_result_dict(result)}")
                _snap_acc = None

        # Single HSV conversion shared by saturation and color analysis
        frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        _detect_tick += 1
        if mode != Mode.TRACKING or _detect_tick % DETECT_EVERY_N == 0:
            raw_detected = detect_markers(frame, detector)
            current_detected = smooth_corners(raw_detected, current_detected)
        n_visible = len(current_detected)

        # Infer all 4 corners early (needed for saturation mask)
        if calib is not None:
            result = infer_missing_corners(current_detected, calib)

        display = frame.copy() if layers["raw"] else np.zeros_like(frame)

        if layers["saturation"]:
            inferred_px = {r: result[r].pixel for r in ALL_ROLES} if calib else {}
            if all(inferred_px.get(r) is not None for r in ALL_ROLES):
                quad_mask = make_quad_mask(inferred_px, display.shape)
                display = change_saturation(display, scale=2.5, mask=quad_mask,
                                            frame_hsv=frame_hsv)

        try:
            if mode == Mode.CALIBRATION:
                quad_ready = False
                if n_visible == 4 and all(r in current_detected for r in ALL_ROLES):
                    quad_ready = quad_is_valid(current_detected)
                draw_calib_status(display, current_detected, quad_ready)
                draw_hud(display, mode, n_visible, frame_num, fps=_fps)
                _consume_panel_click(mouse_state, layers, display.shape[1], display.shape[0])

            elif mode == Mode.BSP_BUILD:
                # Draw static calibration quad outline
                if calib:
                    cal_pts = np.array(
                        [_ipt(calib.corners[r]) for r in ALL_ROLES], dtype=np.int32
                    )
                    cv2.polylines(display, [cal_pts], True, GREEN, 2)

                hover_leaf: Optional[BSPNode] = None
                hover_axis: Optional[str]     = None
                hover_nx = hover_ny = 0.0

                if H is not None and bsp_root is not None:
                    nx, ny = transform_pt(H, (mouse_state["x"], mouse_state["y"]))
                    if 0.0 <= nx <= 1.0 and 0.0 <= ny <= 1.0:
                        hover_leaf = bsp_find_leaf(bsp_root, nx, ny)
                        hover_axis = 'v' if mouse_state["shift"] else 'h'
                        hover_nx, hover_ny = nx, ny

                    draw_bsp_overlay(
                        display, bsp_root, H_inv,
                        show_ids=False,
                        hover_leaf=hover_leaf,
                        hover_axis=hover_axis,
                        hover_nx=hover_nx,
                        hover_ny=hover_ny,
                    )

                n_cells = len(bsp_get_leaves(bsp_root)) if bsp_root else 1
                draw_hud(display, mode, n_visible, frame_num, n_cells, fps=_fps)

                # Handle left-click split
                if mouse_state["clicked"] and not _consume_panel_click(
                    mouse_state, layers, display.shape[1], display.shape[0]
                ):
                    mouse_state["clicked"] = False
                    if hover_leaf is not None and hover_axis is not None:
                        if hover_axis == 'h':
                            lo, hi, split_val = hover_leaf.y0, hover_leaf.y1, hover_ny
                        else:
                            lo, hi, split_val = hover_leaf.x0, hover_leaf.x1, hover_nx
                        if lo + BSP_SPLIT_MARGIN < split_val < hi - BSP_SPLIT_MARGIN:
                            bsp_split(hover_leaf, hover_axis, split_val)
                            print(f"[bsp] split {hover_axis} at {split_val:.3f} "
                                  f"→ {len(bsp_get_leaves(bsp_root))} cells")
                        else:
                            print("[bsp] split too close to edge, ignored")

            else:  # TRACKING
                if calib is None:
                    mode = Mode.CALIBRATION
                    continue

                all_px = {r: result[r].pixel for r in ALL_ROLES if result[r].pixel is not None}
                if len(all_px) == 4:
                    last_known = all_px  # type: ignore[assignment]

                cur_H_inv = None
                if bsp_root is not None:
                    cur_corners = {
                        r: (result[r].pixel if result[r].pixel is not None
                            else calib.corners[r])
                        for r in ALL_ROLES
                    }
                    _, cur_H_inv = compute_homographies(cur_corners)

                if layers["rectangles"]:
                    draw_quad_overlay(display, result, last_known)
                    if bsp_root is not None and cur_H_inv is not None:
                        draw_bsp_overlay(display, bsp_root, cur_H_inv, show_ids=True,
                                         source_hsv=frame_hsv,
                                         target_ranges=TARGET_COLORS[TARGET_NAMES[target_idx]])

                draw_hud(display, mode, n_visible, frame_num, fps=_fps)
                draw_text_outlined(
                    display, f"Target: {TARGET_NAMES[target_idx]}  (c=cycle)",
                    (10, display.shape[0] - 20), scale=0.6,
                    fg=TINT_TEXT.get(TARGET_NAMES[target_idx], WHITE))
                _consume_panel_click(mouse_state, layers, display.shape[1], display.shape[0])

            draw_layer_panel(display, layers)

        except Exception as exc:
            print(f"[error] frame {frame_num}: {exc}")

        _frame_times.append(time.time())
        if len(_frame_times) >= 2:
            _fps = len(_frame_times) / (_frame_times[-1] - _frame_times[0])
        else:
            _fps = 0.0

        cv2.imshow("ArUco Tracker", display)
        key = cv2.waitKey(1) & 0xFF

        if key in (ord("q"), 27):
            break

        elif key == ord(" "):
            if mode == Mode.CALIBRATION:
                if len(current_detected) == 4 and quad_ready:
                    save_calibration(current_detected)        # corners only, no BSP yet
                    calib  = load_calibration()
                    H, H_inv = compute_homographies(calib.corners)
                    bsp_root = BSPNode(0.0, 0.0, 1.0, 1.0)   # single cell = full rectangle
                    mode   = Mode.BSP_BUILD
                    print("[state] → BSP_BUILD  (divide rectangle, then SPACE to finalize)")
                elif len(current_detected) < 4:
                    print(f"[calib] only {len(current_detected)}/4 markers visible")
                else:
                    print("[calib] quad failed sanity check")

            elif mode == Mode.BSP_BUILD:
                if bsp_root is not None:
                    bsp_assign_ids(bsp_root)
                    n_cells = len(bsp_get_leaves(bsp_root))
                    save_calibration(calib.corners, bsp_root)
                    calib    = load_calibration()
                    bsp_root = calib.bsp
                    mode     = Mode.TRACKING
                    print(f"[state] → TRACKING ({n_cells} cells)")

            else:  # TRACKING
                print(f"[frame {frame_num}] {build_result_dict(result)}")

        elif key == ord("r"):
            if CALIB_FILE.exists():
                CALIB_FILE.unlink()
                print(f"[reset] deleted {CALIB_FILE}")
            calib    = None
            bsp_root = None
            H = H_inv = None
            mode      = Mode.CALIBRATION
            last_known = None
            quad_ready = False
            result = {r: CornerResult(pixel=None, confidence=Confidence.GHOST) for r in ALL_ROLES}
            print("[state] → CALIBRATION")

        elif key == ord("c"):
            target_idx = (target_idx + 1) % len(TARGET_NAMES)
            print(f"[target] → {TARGET_NAMES[target_idx]}")

        elif key == ord("s"):
            if _snap_count == 0:
                _snap_acc   = None
                _snap_count = SNAPSHOT_FRAMES
                print(f"[capture] averaging {SNAPSHOT_FRAMES} frames...")

    grabber.stop()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
