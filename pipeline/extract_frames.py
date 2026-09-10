"""Decode the colour frames of all three cameras from the actseg sample.

The IMU already told us the head camera moves least and the other two swing to
25 m/s^2, which is the signature of wrist mounting - but that is an inference,
and the mounting decides whether the wrist-gyro gate from the previous clip
applies at all. So pull contact-sheet frames from each camera and look.

Frames are also cached at the pipeline's working resolution for the labelling
pass, so the 4.5 GB file is walked once rather than once per stage.
"""
import os
import sys
import time

import cv2
import numpy as np
from mcap.stream_reader import StreamReader
from mcap.records import Channel, Message

MCAP = "/home/user/actseg/sample.mcap"
OUTDIR = "/home/user/VLM_WORKSPACE/actseg_frames"
CAMS = ["CPA9B520080", "CPA9B52005Z", "CPAW752006B"]
FPS_KEEP = 2.0
PX = 256
SHEET_AT = [10.0, 40.0, 80.0, 120.0, 160.0, 200.0, 240.0]


def decode_jpeg(data):
    i = data.find(b"\xff\xd8\xff")
    if i < 0:
        return None
    return cv2.imdecode(np.frombuffer(data[i:], np.uint8), cv2.IMREAD_COLOR)


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    topics = {"/camera_%s/color/image_raw/compressed" % c: c for c in CAMS}
    chans = {}
    keep = {c: {"t": [], "img": []} for c in CAMS}
    sheet = {c: {} for c in CAMS}
    next_t = {c: 0.0 for c in CAMS}
    t0 = None
    n = 0
    t_start = time.time()
    last = time.time()

    with open(MCAP, "rb") as fh:
        for rec in StreamReader(fh, skip_magic=False).records:
            if isinstance(rec, Channel):
                chans[rec.id] = rec.topic
            elif isinstance(rec, Message):
                n += 1
                if t0 is None:
                    t0 = rec.log_time
                topic = chans.get(rec.channel_id)
                cam = topics.get(topic)
                if cam is None:
                    continue
                t = (rec.log_time - t0) / 1e9

                want_sheet = [s for s in SHEET_AT
                              if s not in sheet[cam] and abs(t - s) < 0.6]
                need_keep = t + 1e-6 >= next_t[cam]
                if not want_sheet and not need_keep:
                    continue

                img = decode_jpeg(bytes(rec.data))
                if img is None:
                    continue
                for s in want_sheet:
                    sheet[cam][s] = img.copy()
                if need_keep:
                    h, w = img.shape[:2]
                    sc = PX / max(h, w)
                    small = cv2.resize(img, (int(w * sc), int(h * sc)),
                                       interpolation=cv2.INTER_AREA)
                    keep[cam]["t"].append(t)
                    keep[cam]["img"].append(
                        cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
                    next_t[cam] = t + 1.0 / FPS_KEEP

                if time.time() - last > 30:
                    print("  %.0fs: t=%.1fs, kept %s"
                          % (time.time() - t_start, t,
                             {c: len(keep[c]["t"]) for c in CAMS}), flush=True)
                    last = time.time()

    print("scan done in %.0f s (%d messages)" % (time.time() - t_start, n))

    for cam in CAMS:
        ts = keep[cam]["t"]
        print("  %s: %d frames kept, span %.1f s"
              % (cam, len(ts), (ts[-1] - ts[0]) if len(ts) > 1 else 0))
        if keep[cam]["img"]:
            np.savez_compressed(
                "%s/%s.npz" % (OUTDIR, cam),
                t=np.array(ts, np.float64),
                img=np.stack(keep[cam]["img"]).astype(np.uint8))

        cols = []
        for s in SHEET_AT:
            im = sheet[cam].get(s)
            if im is None:
                continue
            h, w = im.shape[:2]
            im = cv2.resize(im, (320, int(h * 320 / w)))
            cv2.putText(im, "%s t=%.0fs" % (cam, s), (8, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2,
                        cv2.LINE_AA)
            cols.append(im)
        if cols:
            hmin = min(c.shape[0] for c in cols)
            cols = [c[:hmin] for c in cols]
            row = np.hstack(cols)
            cv2.imwrite("%s/sheet_%s.jpg" % (OUTDIR, cam), row)
            print("     sheet: %s/sheet_%s.jpg" % (OUTDIR, cam))


if __name__ == "__main__":
    main()
