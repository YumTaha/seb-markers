"""Define tool slots by drawing a box around each cutout on the flattened board.

RUN FROM THE RDP DESKTOP:
    cd ~/seb-markers && .venv/bin/python slots_define.py

Grabs the canonical top-down view; drag a rectangle around each tool's cutout.
Boxes are stored in NORMALIZED table coords (0-1) in slots.json, so they track the
board regardless of camera/table position.
Keys:  drag = add box,  u = undo last,  c = clear all,  s = save,  q = quit.
"""
from __future__ import annotations
import os
os.environ.setdefault("QT_QPA_PLATFORM", "xcb")
import cv2, numpy as np, json, sys
from pink_test import grab, CANON   # grab() -> (canonical BGR, reproj); CANON=(W,H)

def main():
    canon, reproj = grab(ev=120)
    W, H = CANON
    print(f"[grab] canonical {W}x{H}, reproj {reproj:.2f}px")
    boxes = []                     # (x0,y0,x1,y1) in canonical px
    drawing = {"on": False, "x0": 0, "y0": 0, "x1": 0, "y1": 0}
    win = "define slots: drag a box per tool | u=undo c=clear s=save q=quit"

    def redraw():
        d = canon.copy()
        for i, (x0, y0, x1, y1) in enumerate(boxes):
            cv2.rectangle(d, (x0, y0), (x1, y1), (0, 255, 0), 2)
            cv2.putText(d, str(i), (x0 + 4, y0 + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        if drawing["on"]:
            cv2.rectangle(d, (drawing["x0"], drawing["y0"]), (drawing["x1"], drawing["y1"]),
                          (0, 200, 255), 2)
        cv2.putText(d, f"{len(boxes)} slots", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow(win, d)

    def on_mouse(ev, x, y, flags, param):
        if ev == cv2.EVENT_LBUTTONDOWN:
            drawing.update(on=True, x0=x, y0=y, x1=x, y1=y)
        elif ev == cv2.EVENT_MOUSEMOVE and drawing["on"]:
            drawing.update(x1=x, y1=y); redraw()
        elif ev == cv2.EVENT_LBUTTONUP and drawing["on"]:
            x0, y0 = drawing["x0"], drawing["y0"]
            x1, y1 = min(max(x,0),W), min(max(y,0),H)
            drawing["on"] = False
            if abs(x1-x0) > 8 and abs(y1-y0) > 8:
                boxes.append((min(x0,x1), min(y0,y1), max(x0,x1), max(y0,y1)))
            redraw()

    cv2.namedWindow(win, cv2.WINDOW_NORMAL); cv2.resizeWindow(win, 1280, 760)
    cv2.setMouseCallback(win, on_mouse); redraw()
    while True:
        k = cv2.waitKey(20) & 0xFF
        if k == ord('q'): print("[quit] not saved"); break
        if k == ord('u') and boxes: boxes.pop(); redraw()
        if k == ord('c'): boxes.clear(); redraw()
        if k == ord('s'):
            slots = [{"id": i, "x0": x0/W, "y0": y0/H, "x1": x1/W, "y1": y1/H}
                     for i, (x0, y0, x1, y1) in enumerate(boxes)]
            json.dump({"canon": [W, H], "slots": slots}, open("slots.json", "w"), indent=2)
            cv2.imwrite("slots_ref.png", canon)
            print(f"[saved] slots.json ({len(slots)} slots) + slots_ref.png"); break
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
