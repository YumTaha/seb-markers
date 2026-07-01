"""Phase 0 bake-off: which marker COLOR + detector CHANNEL is most robust.

All markers share IDs 0-3, so we classify each detected marker by COLOR instead:
  1. Build a reference from the best-exposed frame: detect all markers, cluster
     their centroids into 4 spatial groups, and within each group assign
     red / yellow / white by relative color score (reddest=red, yellowest=yellow,
     rest=white). -> 12 reference points, each tagged with a color.
  2. Sweep exposure; for every shot & channel, a reference marker is "detected"
     if some detection lands within MATCH_R px of it.
Saves raw shots + per-channel overlays + results matrix to testing/<ts>_<label>/.
"""
from __future__ import annotations
import cv2, numpy as np, subprocess, time, json, os, sys

CAM = 0
EXPOSURES = [8, 15, 25, 40, 60, 90, 130, 180, 250, 350, 500]
GAIN = 0
SETTLE = 8
MATCH_R = 50          # px: a detection matches a reference marker within this radius
DICT = cv2.aruco.DICT_4X4_50
CHANS = ["gray", "R", "G", "B", "maxRG", "V"]
COLORS = ["red", "white", "yellow"]

def ctl(**kw):
    for k, v in kw.items():
        subprocess.run(["v4l2-ctl", "-d", f"/dev/video{CAM}", "-c", f"{k}={v}"],
                       capture_output=True)

def channels(bgr):
    b, g, r = bgr[:, :, 0], bgr[:, :, 1], bgr[:, :, 2]
    return {"gray": cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), "R": r, "G": g, "B": b,
            "maxRG": np.maximum(r, g),
            "V": cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[:, :, 2]}

def color_scores(bgr, quad):
    """Return (red%, yellow%, white%) of the marker's bright (light-module) pixels."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = np.zeros(bgr.shape[:2], np.uint8); cv2.fillPoly(mask, [quad.astype(np.int32)], 255)
    pix = hsv[mask > 0]
    if len(pix) == 0:
        return 0, 0, 0
    bright = pix[pix[:, 2] >= np.percentile(pix[:, 2], 55)]
    H, S = bright[:, 0].astype(int), bright[:, 1].astype(int)
    redp = 100 * np.mean((S > 50) & ((H < 15) | (H > 160)))
    yelp = 100 * np.mean((S > 50) & (H >= 15) & (H <= 45))
    whtp = 100 * np.mean(S <= 45)
    return float(redp), float(yelp), float(whtp)

def detect(detector, img):
    c, ids, _ = detector.detectMarkers(img)
    cents = [m[0].mean(axis=0) for m in c] if ids is not None else []
    return c, cents

def build_reference(bgr, detector):
    """Detect on gray, cluster to 4 groups, assign a color per marker."""
    c, cents = detect(detector, cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY))
    pts = np.array(cents, np.float32)
    k = min(4, len(pts))
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, labels, _ = cv2.kmeans(pts, k, None, crit, 5, cv2.KMEANS_PP_CENTERS)
    labels = labels.flatten()
    ref = []   # (cx, cy, color)
    for g in range(k):
        idx = [i for i in range(len(pts)) if labels[i] == g]
        scored = [(i, color_scores(bgr, c[i][0])) for i in idx]
        # assign within group: reddest->red, then yellowest->yellow, rest->white
        order = sorted(scored, key=lambda t: -t[1][0])
        assigned = {}
        if order:
            assigned[order[0][0]] = "red"
        rest = [t for t in order[1:]]
        rest_y = sorted(rest, key=lambda t: -t[1][1])
        if rest_y:
            assigned[rest_y[0][0]] = "yellow"
        for t in rest_y[1:]:
            assigned[t[0]] = "white"
        for i in idx:
            cx, cy = pts[i]
            ref.append((float(cx), float(cy), assigned.get(i, "white")))
    return ref

def main():
    label = sys.argv[1] if len(sys.argv) > 1 else "run"
    ts = time.strftime("%Y%m%d_%H%M%S")
    outdir = os.path.join("testing", f"{ts}_{label}")
    os.makedirs(outdir, exist_ok=True)
    print(f"[out] {outdir}")

    cap = cv2.VideoCapture(CAM, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    if not cap.isOpened():
        print("[error] cannot open camera"); sys.exit(1)
    det = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(DICT),
                                  cv2.aruco.DetectorParameters())
    ctl(auto_exposure=1, gain=GAIN)

    # ---- pass 1: capture every exposure, save raws ----
    frames = {}
    for ev in EXPOSURES:
        ctl(exposure_time_absolute=ev)
        for _ in range(SETTLE): cap.read()
        ok, f = cap.read()
        if not ok: print(f"[warn] no frame ev={ev}"); continue
        frames[ev] = f
        cv2.imwrite(os.path.join(outdir, f"01_raw_ev{ev:03d}.png"), f)
    cap.release()

    # ---- reference: pick frame with most markers ----
    best_ev = max(frames, key=lambda e: len(detect(det, cv2.cvtColor(frames[e], cv2.COLOR_BGR2GRAY))[1]))
    ref = build_reference(frames[best_ev], det)
    nref = {col: sum(1 for _,_,c in ref if c == col) for col in COLORS}
    print(f"[ref] ev{best_ev}: {len(ref)} markers -> " + ", ".join(f"{c}:{nref[c]}" for c in COLORS))
    # save annotated reference
    vis = frames[best_ev].copy()
    cmap = {"red": (0,0,255), "white": (255,255,255), "yellow": (0,255,255)}
    for cx, cy, col in ref:
        cv2.circle(vis, (int(cx),int(cy)), 30, cmap[col], 3)
        cv2.putText(vis, col[0].upper(), (int(cx)-10,int(cy)+10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, cmap[col], 3)
    cv2.imwrite(os.path.join(outdir, "00_reference.png"), vis)

    # ---- pass 2: per exposure/channel, which reference markers detected ----
    matrix = {ch: {c: [] for c in COLORS} for ch in CHANS}
    for ev in EXPOSURES:
        if ev not in frames: continue
        chs = channels(frames[ev])
        line = f"ev{ev:3d}: "
        for ch in CHANS:
            c, cents = detect(det, chs[ch])
            got = {col: 0 for col in COLORS}
            for cx, cy, col in ref:
                if any((cx-dx)**2 + (cy-dy)**2 <= MATCH_R**2 for dx, dy in cents):
                    got[col] += 1
            for col in COLORS:
                matrix[ch][col].append(got[col])
            vis = cv2.cvtColor(chs[ch], cv2.COLOR_GRAY2BGR)
            if c: cv2.aruco.drawDetectedMarkers(vis, c)
            cv2.putText(vis, f"{ch} ev{ev} r{got['red']} w{got['white']} y{got['yellow']}",
                        (20,50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0), 2)
            cv2.imwrite(os.path.join(outdir, f"02_det_ev{ev:03d}_{ch}.png"), cv2.resize(vis,(960,540)))
            line += f"{ch}[r{got['red']}w{got['white']}y{got['yellow']}] "
        print(line)

    # ---- summary ----
    n = len(EXPOSURES)
    print(f"\n=== robustness: # exposures (of {n}) with ALL {nref} of that color detected ===")
    print(f"{'channel':8s}" + "".join(f"{c:>9s}" for c in COLORS))
    summary = {}
    for ch in CHANS:
        summary[ch] = {}
        line = f"{ch:8s}"
        for c in COLORS:
            full = sum(1 for v in matrix[ch][c] if v >= nref[c] and nref[c] > 0)
            any_ = sum(1 for v in matrix[ch][c] if v > 0)
            summary[ch][c] = {"exposures_all": full, "exposures_any": any_, "counts": matrix[ch][c]}
            line += f"  {full:2d}/{any_:2d} "    # all-detected / any-detected
        print(line + "   (all / any)")
    print("\nLegend: 'all' = exposures where every marker of that color was found;"
          " 'any' = exposures with >=1.")
    print("\n=== best channel per color (by 'all', tiebreak 'any') ===")
    for c in COLORS:
        best = max(CHANS, key=lambda ch: (summary[ch][c]["exposures_all"], summary[ch][c]["exposures_any"]))
        s = summary[best][c]
        print(f"  {c:7s}: {best:6s}  all {s['exposures_all']}/{n}, any {s['exposures_any']}/{n}")

    with open(os.path.join(outdir, "results.json"), "w") as f:
        json.dump({"exposures": EXPOSURES, "gain": GAIN, "best_ev_for_ref": best_ev,
                   "n_per_color": nref, "summary": summary}, f, indent=2)
    print(f"\n[saved] {outdir}/  (00_reference.png, raws, overlays, results.json)")

if __name__ == "__main__":
    main()
