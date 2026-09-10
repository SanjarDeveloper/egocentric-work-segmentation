"""Render the actseg segmentation over the ego camera.

There is no MP4 for this recording and the cached frames are 256 px working
copies, so the video is rebuilt from the MCAP at display resolution in one
sequential pass - the file's index is corrupt, so seeking is not available
anyway.

No ground truth exists for this clip, so the panel carries the model's mask and
the motion curve it came from, and nothing pretends to be an accuracy figure.
"""
import json

import cv2
import numpy as np
from mcap.stream_reader import StreamReader
from mcap.records import Channel, Message

SEG = "/home/user/VLM_WORKSPACE/actseg_segmentation.json"
OUT = "/home/user/VLM_WORKSPACE/overlay_actseg.mp4"
FPS_OUT = 8.0
WIDTH = 960
HZ = 10

GREEN = (80, 200, 80)
GREY = (140, 140, 140)
WHITE = (245, 245, 245)
DARK = (26, 26, 26)
CYAN = (220, 200, 60)


def decode_jpeg(data):
    i = data.find(b"\xff\xd8\xff")
    if i < 0:
        return None
    return cv2.imdecode(np.frombuffer(data[i:], np.uint8), cv2.IMREAD_COLOR)


def main():
    d = json.load(open(SEG))
    dur = d["duration_s"]
    n = int(dur * HZ)
    score = np.array(d["score"], np.float32)
    if len(score) != n:
        score = np.interp(np.linspace(0, 1, n),
                          np.linspace(0, 1, len(score)), score)
    lo, hi = np.percentile(score, 1), np.percentile(score, 99)
    curve = np.clip((score - lo) / max(hi - lo, 1e-6), 0, 1)
    thr_draw = float(np.clip((d["config"]["threshold"] - lo) / max(hi - lo, 1e-6),
                             0, 1))

    pred = np.zeros(n, bool)
    for s in d["segments"]:
        pred[int(s["start"] * HZ):min(int(s["end"] * HZ), n)] = True

    def seg_at(t):
        for s in d["segments"]:
            if s["start"] <= t < s["end"]:
                return s
        return None

    # ego_camera already carries the "camera_" prefix in the JSON
    cam = d["ego_camera"]
    topic = "/%s/color/image_raw/compressed" % cam.lstrip("/")
    print("decoding %s ..." % topic, flush=True)

    chans = {}
    frames = []
    t0 = None
    next_t = 0.0
    want_dt = 1.0 / FPS_OUT
    with open(d["source_mcap"], "rb") as fh:
        for rec in StreamReader(fh, skip_magic=False).records:
            if isinstance(rec, Channel):
                chans[rec.id] = rec.topic
            elif isinstance(rec, Message):
                if t0 is None:
                    t0 = rec.log_time
                if chans.get(rec.channel_id) != topic:
                    continue
                t = (rec.log_time - t0) / 1e9
                if t + 1e-6 < next_t:
                    continue
                img = decode_jpeg(bytes(rec.data))
                if img is None:
                    continue
                h, w = img.shape[:2]
                frames.append((t, cv2.resize(img, (WIDTH, int(h * WIDTH / w)),
                                             interpolation=cv2.INTER_AREA)))
                next_t = t + want_dt
                if len(frames) % 300 == 0:
                    print("  %d frames (%.0f s)" % (len(frames), t), flush=True)

    if not frames:
        # a wrong topic name decodes nothing and would otherwise exit 0,
        # leaving an empty run that looks like success
        print("no frames decoded from %s" % topic)
        print("topics seen: %s" % sorted(set(chans.values()))[:20])
        raise SystemExit(1)
    vh = frames[0][1].shape[0]
    panel = 140
    out = cv2.VideoWriter(OUT, cv2.VideoWriter_fourcc(*"mp4v"), FPS_OUT,
                          (WIDTH, vh + panel))

    sub1 = ("Gate: max wrist-camera gyro (8s) - 0.75 x head jerk  |  "
            "%d spans, %.0f%% work  |  no ground truth for this clip"
            % (len(d["segments"]), d["work_pct"]))
    sub2 = ("Labels: Cosmos-Reason2-2B-W4A16, one query per span  |  "
            "%.1f s for %.0f s of video (%.2fx realtime)"
            % (d["timing"]["total_s"], dur, d["timing"]["realtime_factor"]))

    written = 0
    for t, frame in frames:
        ti = min(int(t * HZ), n - 1)
        canvas = np.full((vh + panel, WIDTH, 3), DARK, np.uint8)
        canvas[:vh] = frame

        is_work = bool(pred[ti])
        col = GREEN if is_work else GREY
        cv2.rectangle(canvas, (12, 12), (200, 54), (0, 0, 0), -1)
        cv2.rectangle(canvas, (12, 12), (200, 54), col, 2)
        cv2.putText(canvas, "WORK" if is_work else "IDLE", (24, 43),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.95, col, 2, cv2.LINE_AA)
        cv2.putText(canvas, "motion %.2f" % curve[ti], (212, 42),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, CYAN, 1, cv2.LINE_AA)
        cv2.putText(canvas, "%02d:%05.2f" % (int(t) // 60, t % 60),
                    (WIDTH - 140, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                    WHITE, 1, cv2.LINE_AA)

        s = seg_at(t)
        if s and s.get("label"):
            cv2.putText(canvas, "%s (%.2f)" % (s["label"], s["conf"]),
                        (14, vh - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                        GREEN, 2, cv2.LINE_AA)

        cv2.putText(canvas, sub1, (12, vh + 16), cv2.FONT_HERSHEY_SIMPLEX,
                    0.38, (185, 185, 185), 1, cv2.LINE_AA)
        cv2.putText(canvas, sub2, (12, vh + 32), cv2.FONT_HERSHEY_SIMPLEX,
                    0.38, (185, 185, 185), 1, cv2.LINE_AA)

        cx0, cy0, cw, ch = 52, vh + 42, WIDTH - 72, 52
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

        ty0 = cy0 + ch + 8
        hh = 12
        for x in range(cw):
            k = min(int(x / cw * n), n - 1)
            cv2.line(canvas, (cx0 + x, ty0), (cx0 + x, ty0 + hh),
                     GREEN if pred[k] else (66, 66, 66), 1)
        cv2.line(canvas, (curx, ty0 - 2), (curx, ty0 + hh + 2), WHITE, 1)
        cv2.putText(canvas, "WORK", (cx0 - 50, ty0 + hh - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, GREEN, 1, cv2.LINE_AA)

        out.write(canvas)
        written += 1
        if written % 400 == 0:
            print("  wrote %d/%d" % (written, len(frames)), flush=True)

    out.release()
    print("done: %s (%d frames)" % (OUT, written))


if __name__ == "__main__":
    main()
