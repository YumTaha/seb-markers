"""Detect present/missing tools by pink backing coverage per slot.

Robust to lighting: captures an EXPOSURE BRACKET and, per slot, takes the MAX pink%
across exposures. Pink can only be under-detected (washout/glare/shadow), never
over-detected (hue gate excludes foam/tools), so the exposure that sees the most pink
is the truest -> max is the correct combiner (averaging/median would bias toward
false "present"). pink >= threshold => MISSING (backing exposed); else PRESENT.

    .venv/bin/python detect_tools.py
"""
from __future__ import annotations
import cv2, numpy as np, json, time, os, sys
from lens import load_lens, undistort
from locate_table import load_board_calib, detect_union, locate, ctl, DICT
from pink_test import CANON, pink_mask

MISSING_THRESH = 35.0                 # max-pink% >= this -> tool missing
COLOR_EXPO = [20, 45, 90, 180]        # bracket; max across these per slot
DISPLAY_EV = 45

def main():
    if not os.path.exists("slots.json"):
        print("[error] no slots.json — run slots_define.py first"); sys.exit(1)
    slots = json.load(open("slots.json"))["slots"]
    lens = load_lens(); board, _ = load_board_calib(); W, H = CANON
    det = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(DICT),
                                  cv2.aruco.DetectorParameters())
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    if not cap.isOpened(): print("[error] no camera"); sys.exit(1)

    # locate table once
    markers = detect_union(cap, det, lens)
    res = locate(markers, board)
    if not res.get("ok"): print(f"[fail] {res.get('reason')}"); sys.exit(1)
    corners = res["corners"].astype(np.float32)
    M = cv2.getPerspectiveTransform(corners, np.array([[0,0],[W,0],[W,H],[0,H]], np.float32))

    # exposure bracket -> per-slot MAX pink% (+ per-exposure detail)
    max_pct = {s["id"]: 0.0 for s in slots}
    per_expo = {s["id"]: [] for s in slots}
    display = None
    for ev in COLOR_EXPO:
        ctl(auto_exposure=1, gain=0, exposure_time_absolute=ev)
        for _ in range(6): cap.read()
        ok, f = cap.read()
        if not ok: continue
        canon = cv2.warpPerspective(undistort(f, lens), M, (W, H))
        if ev == DISPLAY_EV: display = canon.copy()
        m = pink_mask(canon)
        for s in slots:
            x0,y0,x1,y1 = int(s["x0"]*W),int(s["y0"]*H),int(s["x1"]*W),int(s["y1"]*H)
            pct = 100.0*int(np.count_nonzero(m[y0:y1, x0:x1]))/max((x1-x0)*(y1-y0),1)
            per_expo[s["id"]].append(round(pct,1))
            max_pct[s["id"]] = max(max_pct[s["id"]], pct)
    cap.release()
    if display is None: display = canon

    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    results = []
    print(f"located (reproj {res['reproj']:.2f}px), {len(slots)} slots, "
          f"bracket {COLOR_EXPO}, threshold {MISSING_THRESH}%")
    print(f"{'slot':>4} {'maxPink%':>9}  {'per-exposure':>22}  status")
    for s in slots:
        mp = max_pct[s["id"]]; missing = mp >= MISSING_THRESH
        status = "MISSING" if missing else "present"
        results.append({"id": s["id"], "max_pink_pct": round(mp,1),
                        "per_exposure": per_expo[s["id"]], "missing": missing})
        x0,y0,x1,y1 = int(s["x0"]*W),int(s["y0"]*H),int(s["x1"]*W),int(s["y1"]*H)
        col = (0,0,255) if missing else (0,200,0)
        cv2.rectangle(display,(x0,y0),(x1,y1),col,3)
        cv2.putText(display, f"{s['id']}:{status} {mp:.0f}%", (x0, y0-8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
        print(f"{s['id']:>4} {mp:>8.1f}%  {str(per_expo[s['id']]):>22}  {status}")

    n_missing = sum(1 for r in results if r["missing"])
    banner = f"{n_missing} MISSING / {len(slots)}" if n_missing else "ALL PRESENT"
    cv2.putText(display, banner, (15,40), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                (0,0,255) if n_missing else (0,200,0), 3)
    os.makedirs("testing", exist_ok=True)
    stamp = time.strftime("%H%M%S")
    cv2.imwrite(f"testing/audit_{stamp}.png", display)
    json.dump({"timestamp": ts, "reproj_px": round(res["reproj"],2),
               "threshold": MISSING_THRESH, "bracket": COLOR_EXPO,
               "slots": results, "n_missing": n_missing},
              open(f"testing/audit_{stamp}.json","w"), indent=2)
    print(f"\n{banner}\n[saved] testing/audit_{stamp}.png + .json")

if __name__ == "__main__":
    main()
