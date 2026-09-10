"""Render the IMU-gated segmentation over the ego video.

Three strips stacked under the motion curve - the model's mask, the ground
truth, and the disagreement between them - so the failure modes are readable
without cross-referencing the JSON. The curve itself is the fused motion score
(wrist gyro minus head jerk), with the percentile cut drawn across it.
"""
import json

import cv2
import numpy as np

SEG = "/home/user/VLM_WORKSPACE/imu_segmentation.json"
OUT = "/home/user/VLM_WORKSPACE/overlay_imu.mp4"
FPS_OUT = 10.0
WIDTH = 960
HZ = 10

GREEN = (80, 200, 80)
GREY = (140, 140, 140)
WHITE = (245, 245, 245)
DARK = (26, 26, 26)
CYAN = (220, 200, 60)
AMBER = (60, 180, 240)
RED = (70, 70, 220)


def main():
    d = json.load(open(SEG))
    dur = d["timing"]["video_dur_s"]
    n = int(dur * HZ)
    score = np.array(d["score"], np.float32)
    if len(score) != n:
        score = np.interp(np.linspace(0, 1, n),
                          np.linspace(0, 1, len(score)), score)

    # normalise the fused z-score to 0..1 for drawing only
    lo, hi = np.percentile(score, 1), np.percentile(score, 99)
    curve = np.clip((score - lo) / max(hi - lo, 1e-6), 0, 1)
    thr_draw = float(np.clip((d["config"]["threshold"] - lo) / max(hi - lo, 1e-6),
                             0, 1))

    pred = np.zeros(n, bool)
    for s in d["segments"]:
        pred[int(s["start"] * HZ):min(int(s["end"] * HZ), n)] = True
    gt = np.zeros(n, bool)
    for s in d["gt"]:
        if s["work"]:
            gt[int(s["start"] * HZ):min(int(s["end"] * HZ), n)] = True

    def pred_at(t):
        for s in d["segments"]:
            if s["start"] <= t < s["end"]:
                return s
        return None

    def gt_at(t):
        for s in d["gt"]:
            if s["start"] <= t < s["end"]:
                return s
        return None

    cap = cv2.VideoCapture(d["source_mp4"])
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, int(round(src_fps / FPS_OUT)))
    ok, first = cap.read()
    if not ok:
        print("cannot read video")
        return
    h0, w0 = first.shape[:2]
    vh = int(h0 * WIDTH / w0)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    panel = 168
    out = cv2.VideoWriter(OUT, cv2.VideoWriter_fourcc(*"mp4v"), FPS_OUT,
                          (WIDTH, vh + panel))

    r = d["result"]
    sub1 = ("Gate: wrist IMU gyro (max L/R, 8s) - 0.75 x head jerk  |  "
            "AUC %.3f  (VLM-only gate scored 0.611)" % d["auc"]["fused_imu"])
    sub2 = ("Labels: Cosmos-Reason2-2B-W4A16, one query per span  |  "
            "P %.2f  R %.2f  F1 %.2f  acc %.2f  |  %.2fx realtime"
            % (r["precision"], r["recall"], r["f1"], r["accuracy"],
               d["timing"]["realtime_factor"]))

    idx = written = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % step:
            idx += 1
            continue
        t = idx / src_fps
        ti = min(int(t * HZ), n - 1)
        canvas = np.full((vh + panel, WIDTH, 3), DARK, np.uint8)
        canvas[:vh] = cv2.resize(frame, (WIDTH, vh), interpolation=cv2.INTER_AREA)

        is_work = bool(pred[ti])
        col = GREEN if is_work else GREY
        cv2.rectangle(canvas, (12, 12), (200, 54), (0, 0, 0), -1)
        cv2.rectangle(canvas, (12, 12), (200, 54), col, 2)
        cv2.putText(canvas, "WORK" if is_work else "IDLE", (24, 43),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.95, col, 2, cv2.LINE_AA)
        cv2.putText(canvas, "motion %.2f" % curve[ti], (212, 42),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, CYAN, 1, cv2.LINE_AA)
        agree = bool(pred[ti]) == bool(gt[ti])
        cv2.putText(canvas, "MATCH" if agree else "MISS", (360, 42),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    GREEN if agree else RED, 2, cv2.LINE_AA)
        cv2.putText(canvas, "%02d:%05.2f" % (int(t) // 60, t % 60),
                    (WIDTH - 140, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                    WHITE, 1, cv2.LINE_AA)

        ps = pred_at(t)
        if ps and ps.get("label"):
            cv2.putText(canvas, "MODEL: %s (%.2f)" % (ps["label"], ps["conf"]),
                        (14, vh - 44), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        GREEN, 2, cv2.LINE_AA)
        gs = gt_at(t)
        if gs:
            cv2.putText(canvas, ("GT: " + gs["title"])[:66], (14, vh - 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        AMBER if gs["work"] else GREY, 2, cv2.LINE_AA)

        cv2.putText(canvas, sub1, (12, vh + 16), cv2.FONT_HERSHEY_SIMPLEX,
                    0.38, (185, 185, 185), 1, cv2.LINE_AA)
        cv2.putText(canvas, sub2, (12, vh + 32), cv2.FONT_HERSHEY_SIMPLEX,
                    0.38, (185, 185, 185), 1, cv2.LINE_AA)

        cx0, cy0, cw, ch = 52, vh + 40, WIDTH - 72, 46
        cv2.rectangle(canvas, (cx0, cy0), (cx0 + cw, cy0 + ch), (46, 46, 46), -1)
        ty = int(cy0 + ch - thr_draw * ch)
        cv2.line(canvas, (cx0, ty), (cx0 + cw, ty), (95, 95, 165), 1)
        cv2.putText(canvas, "cut", (cx0 - 46, ty + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, (150, 150, 205), 1,
                    cv2.LINE_AA)
        pts = [(cx0 + x, int(cy0 + ch - curve[min(int(x / cw * n), n - 1)] * ch))
               for x in range(cw)]
        for a, b in zip(pts, pts[1:]):
            cv2.line(canvas, a, b, CYAN, 1)
        curx = int(cx0 + cw * t / dur)
        cv2.line(canvas, (curx, cy0), (curx, cy0 + ch), WHITE, 1)
        cv2.putText(canvas, "motion", (cx0 - 50, cy0 + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.30, CYAN, 1, cv2.LINE_AA)

        diff = pred != gt
        for row, (mask, name, colr) in enumerate(
                [(pred, "IMU", GREEN), (gt, "GT", AMBER), (diff, "DIFF", RED)]):
            ty0 = cy0 + ch + 6 + row * 14
            hh = 10
            for x in range(cw):
                k = min(int(x / cw * n), n - 1)
                cv2.line(canvas, (cx0 + x, ty0), (cx0 + x, ty0 + hh),
                         colr if mask[k] else (66, 66, 66), 1)
            cv2.line(canvas, (curx, ty0 - 2), (curx, ty0 + hh + 2), WHITE, 1)
            cv2.putText(canvas, name, (cx0 - 50, ty0 + hh - 1),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.32, colr, 1, cv2.LINE_AA)

        out.write(canvas)
        written += 1
        idx += 1
        if written % 400 == 0:
            print("  %d frames (%.0f s)" % (written, t), flush=True)

    cap.release()
    out.release()
    print("done: %s (%d frames)" % (OUT, written))


if __name__ == "__main__":
    main()
