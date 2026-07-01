# Tool-Board Accountability System — Design & Testing Plan

## 1. Goal / operating context
At the end of an operator's shift the tool board (table) is left in a roughly repeatable
spot. A fixed camera takes pictures, a batch pipeline locates the board via ArUco markers,
checks each tool slot for "present vs missing", and saves an annotated audit photo + a
record. Purpose: know exactly when a tool went missing and who had the board.

Hard constraints:
- **Uncontrolled lighting** — operators leave the board anywhere; the system gets a camera
  but no control over ambient light. Must work dark→bright→glary.
- **Not live** — scheduled batch. We can afford slow, heavy processing (multiple exposures,
  upscaling, accurate-but-slow detector params).
- **Corners are occupied by tools** — markers go along the **edges**, spread out, not at the
  physical corners. Their offsets to the table corners are known/measured once.
- **Markers are RED + black** (3D-printed, no white filament). This matters — see §2/§3.1.

## 2. Why the markers fail today (measured)
- The code never sets exposure → camera auto-exposure overexposes bright regions; the
  washed-out marker (id0) was detected **0%** of frames.
- The board is lit **unevenly** (bright top, dark bottom). Any single global exposure is a
  **seesaw**: tuning for the bright top markers starves the dark bottom markers (→ 0%) and
  vice-versa.
- Post-processing alone (global contrast, histogram-eq, aggressive error-correction,
  2× upscale) either didn't recover id0 or did so only by *guessing* bits (false IDs) and
  ran at 0.7–2.1 s/frame.
- **Markers are red/black, but the code detects on standard luminance grayscale, where red
  maps to only ~76/255 vs black ~0 → only ~30% of white/black contrast.** This low baseline
  contrast compounds the lighting washout. **Measured fix: feed the detector the RED CHANNEL
  instead of std gray** — at fixed exposure, per-marker detection went
  std-gray 50/53/83/70% → red-channel 100/63/100/100%.
- CLAHE helps white/black markers but **hurts red/black ones** (amplifies foam texture/noise
  that breaks the clean red-vs-black separation: red+CLAHE = 76/56/93/73%). Don't use it here.

Conclusion: the two big levers are **(1) red-channel input** (fixes the color-contrast deficit)
and **(2) capture-time exposure bracketing** (fixes uneven/unknown lighting). Filters like
CLAHE are not helpful for these markers.

## 3. Architecture

### 3.1 Capture & preprocess — grayscale + exposure/gain bracket + union
**Marker color = YELLOW + black** and **detector input = standard grayscale** (Phase-0
bake-off result, see §3.5). Yellow is bright in every channel, so it survives the widest
exposure range; the channel choice barely matters and plain gray is as good as any. No CLAHE.
(Red — the original color — was the *worst*; white was middling.)

Per run, sweep a wide exposure (and, for dark rooms, gain) bracket with auto-exposure OFF.
Detect markers in every shot; a marker is "found" if **any** shot finds it; keep the
sharpest instance of each. Union across the bracket = every marker captured regardless of
where it sits in the brightness gradient.
- Start with a fixed wide sweep (robust, simplest). Example grid (to be tuned in testing):
  exposure_time_absolute ∈ {10,25,50,100,200,400,800}, gain ∈ {0, low, high-if-dark}.
- Optional later: a metering shot to center the bracket adaptively.

### 3.x Note — red markers vs the color tool-presence stage
The markers are red and the tool-presence stage keys on color (TARGET_COLORS includes red).
Markers sit on the edges and we know their image positions, so **mask out the marker regions**
before color analysis to avoid counting markers as tools.

### 3.2 Locate the table — one homography from known marker layout
("the proper version of: I know each marker's offset, hardcode it")
- Known once per board: for each marker ID, the table-plane coordinates of its 4 corners
  (normalized table frame TL=(0,0), TR=(1,0), BR=(1,1), BL=(0,1)).
- From the unioned detections, gather correspondences: detected image corner ↔ known
  table-plane corner. Each visible marker contributes 4 points.
- `cv2.findHomography(table_pts, image_pts, cv2.RANSAC)` → table→image transform H_inv,
  and its inverse → image→table H. (Generalizes the current `getPerspectiveTransform`.)
- Compute the 4 table corners' image positions from H_inv. Works from any subset
  (even 1 marker = 4 points), more markers → more accurate.

### 3.3 Validate (the audit gate — avoid false "all good")
Reject the run (fail out) unless ALL hold:
1. Enough markers detected (target ≥2 well-spread; ≥1 allowed with low confidence).
2. **Reprojection error** of detected markers under H is small → detections agree with the
   known layout (catches misreads / wrong-ID false positives).
3. All 4 computed table corners lie **inside the image** with a margin → whole board
   captured (kills half-in/half-out false positives).
4. Quad sanity: convex, plausible area, aspect ratio ≈ known table aspect.
   (Optional cheap pre-check: markers fall within an expected ROI — the user's idea.)

### 3.4 Cells + tool presence (existing BSP/color logic, later phase)
Map the known tool-slot cells through H and decide present/missing per slot. Save annotated
audit image + JSON record.

## 3.5 Marker color/channel — DECIDED by bake-off (Phase 0): YELLOW + grayscale
Candidates were red, white, yellow (all + black), 4 markers each, clustered as
red+white+yellow trios at 4 spots so each trio shares identical lighting. All markers share
IDs 0-3, so `testing/bakeoff.py` classifies each by COLOR via spatial grouping (cluster the
12 detections into 4 groups; within each group reddest=red, yellowest=yellow, rest=white),
not by ID.
- **RESULT (normal lighting, `testing/20260630_191156_normal/`):** across an 11-step exposure
  sweep, exposures with all-4-of-color detected were **yellow 10/11, white 7/11, red 2-3/11**.
  Yellow wins decisively, at both dark & bright extremes; channel barely matters.
- **Decision: YELLOW + black markers, standard grayscale detection.** (Red — the original
  color — was the worst.)
- TODO to fully confirm: re-run under real glare and dim conditions
  (`python testing/bakeoff.py glare`, `... dim`). Exposure sweep already covers the
  brightness axis; expect yellow to hold.

## Progress log
- ✅ Phase 0: color/channel bake-off → **yellow + grayscale**.
- ✅ Lens calibration: `camera_calib.json` (RMS 0.755 px, 62 views, k1=-0.41 barrel).
  Pipeline now undistorts every frame first (`lens.py`).
- ✅ Phase 2: board click-calibration (`board_calib.json`, ids 0-3) + runtime
  locate-the-table (`locate_table.py`): undistort→detect→homography→derive→validate.
  Tested live: PASS, reprojection 0.86 px, corners align, spurious marker ignored.
  Markers currently TEMPORARY → re-run `calibrate_board.py` once permanent ones placed.
- ✅ Warp to canonical top-down (`warp_table.py`).
- ⏳ Phase 3: backing-paper color. Bake-off (`paper_bakeoff.py`, testing/paperbake_*):
  pink/green/yellow all segment cleanly at low exposure (counts 9/7/7 = ground truth,
  no false positives — pink NOT confused with red tools). Pink most saturated + most
  robust as brightness rises (green/yellow wash out first). **Leaning PINK** (no tool is
  pink → works for all tools, no per-slot exclusions).
- ✅ Pink under-tool drop test (`pink_test.py`): 3 tools, pink 58-75% exposed → 7-17%
  covered — wide gap, held even in darker light. **PINK LOCKED.** Threshold ~35%.
- ✅ Define tool slots (`slots_define.py` → `slots.json`, normalized table coords).
- ✅ Detect present/missing (`detect_tools.py`): per slot, MAX pink% across an EXPOSURE
  BRACKET [20,45,90,180] (max is the right combiner — pink can only be under-detected, so
  averaging/median would bias to false "present"; e.g. slot read [47,48,33,2], avg 33 would
  fail but max 48 is correct). pink>=35% => MISSING. Saves audit PNG + JSON.
  VALIDATED end-to-end: tools-off = all MISSING (48-59%), tools-on = all PRESENT (11-17%),
  35% threshold with margin both sides. Red handles are a non-issue (we measure pink, not
  tool color — a red-handled tool over pink still reads present).
- ✅ **Auto_Run** (`~/Auto_Run/`, separate uv project): sews the whole pipeline into one app.
  Wizard (calibrate edges → draw+name slots → schedule) + headless `auto_run.run` fired by
  cron at set times: locate (retry 1min×10 if no table → save "NO TABLE"), else detect
  present/missing → dated audit PNG (green/red) + JSON + `records/log.csv`. Headless run
  validated (all-present audit + log row). User runs the GUI wizard from RDP to arm it.
- ⏳ Next: finalize board (permanent yellow markers + pink under every cutout), recalibrate,
  redraw+name all slots; collect a week of data via Auto_Run for the presentation.

## 4. Testing plan (phased)

### Phase 0 — Marker color + channel bake-off  ← DO THIS FIRST
Goal: pick the light-module color and the detector channel together.
- Place the 12 markers as trios (above), whole set in frame.
- Script sweeps exposure (auto OFF) and, per shot, runs detection on each candidate channel:
  std-gray, R, G, B (and a couple combos, e.g. max(R,G), saturation).
- Output: a matrix **color × channel × exposure → detected?**, plus per-color detection rate
  and the exposure *range* each color survives. Save every shot + per-channel detection
  overlays to `testing/run_<ts>/`.
- Repeat under ≥3 lighting conditions (normal, one-sided glare, dim).
- Exit: choose winning color + channel (most robust across exposure & lighting).


All steps dump intermediate images to `testing/run_<timestamp>/` so defects are visible:
```
testing/run_<ts>/
  01_raw_exp<ev>_gain<g>.png     # every bracket shot (color)
  02_red_exp<ev>_gain<g>.png     # red-channel (detector input) per shot
  03_detect_exp<ev>_gain<g>.png  # markers drawn per shot
  04_union.png                   # all markers found across the bracket
  05_homography_corners.png      # derived table corners
  06_validation.png              # in-frame + reprojection check
  07_cells.png                   # tool-slot cells (later)
  08_result.png                  # final annotated audit image
  result.json                    # ids found, corners, validation, timings
```

### Phase 1 — Detection robustness (no geometry needed yet)  ← START HERE
Goal: "given a scene, reliably detect every marker."
- Place markers in the intended edge-spread layout.
- Run the bracket-capture script; save all shots + per-shot detections to `testing/`.
- **Repeat under ≥3 lighting conditions** (e.g., normal, one-sided lamp/glare, dim).
- Detector input = **red channel** (confirmed best). Also dump a std-gray detection for
  comparison so we keep seeing the delta.
- Measure: per-marker detection rate per exposure; does the **union** get all markers in
  every lighting condition? Tune the exposure/gain grid and detector params
  (keep error-correction moderate to avoid false IDs). CLAHE off by default (hurts red markers).
- Exit criterion: union finds all markers in every test lighting condition.

### Phase 2 — Geometry / corner derivation
- Record each marker's table-plane offsets (see §5).
- Implement homography fit + corner derivation + validation (§3.2–3.3).
- Verify derived corners land on the true table corners (overlay in `05_*`,`06_*`).
- Test half-in/half-out and a removed/garbled marker → must fail out correctly.

### Phase 3 — Cells + tool presence + record
- Wire the existing BSP/color cell logic to the derived homography.
- Tune present/missing decision; produce the audit image + JSON.

## 5. What I need from you (physical)
**For Phase 1 (now):**
- Place the markers in the edge-spread layout (two near top edge L/R, two near bottom edge
  L/R; not collinear; flat; **matte**, not glossy; keep a white quiet-zone border around each).
- Tell me the marker IDs you're using (default DICT_4X4_50, ids 0–3).
- Be able to create ≥3 lighting conditions for the test captures.

**For Phase 2 (later):** the marker→table geometry, via EITHER
- (preferred) a one-time **click calibration**: one good-lighting photo with the whole board
  in frame; you click the 4 true table corners; we auto-record each marker's offset. OR
- hand measurement: for each marker, distance from the table's top-left corner along the top
  edge (x) and down the left edge (y) to the marker's top-left corner, the marker side
  length, and the table's overall width & height. Any consistent unit.

## 6. Decisions
- **Accuracy tolerance — COARSE.** Tools are large and easy to spot; small rectangle error
  is tolerable. Initial target ~**10 px** corner error; tune looser/stricter after we see
  real results. Edge-spread 4 markers is sufficient.
- **Geometry capture — CLICK-CALIBRATION.** One good-lighting reference photo (whole board
  in frame), user clicks the 4 true table corners once over RDP; we auto-record each
  marker's offset.
- Still to settle empirically in Phase 1: exact exposure/gain grid, whether to add adaptive
  metering, final marker count / IDs.
