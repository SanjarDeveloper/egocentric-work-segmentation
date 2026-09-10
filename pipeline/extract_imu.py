"""Extract IMU streams from an MCAP, discovering the field offsets per topic.

The CDR offsets are not fixed across recordings: they depend on the length of
frame_id, so the ZED files put linear_acceleration at byte 244 while the Orbbec
files put it at 260. Rather than hard-code either, this probes the first few
messages of each topic by sliding a 3-double window until it finds the offset
where the magnitude is gravity in every sample, then uses that offset for the
whole topic.

Orbbec splits the IMU across topics - .../imu is the gyro, .../imu/accel is the
accelerometer - so a topic that yields no gravity offset is treated as a gyro
stream and its vector is read from the same place the search would have looked.
"""
import struct
import sys
import time

import numpy as np
from mcap.stream_reader import StreamReader
from mcap.records import Channel, Message


def find_vec_offset(msgs):
    """Offset of linear_acceleration, identified by gravity.

    A hand-held or head-worn camera swings hard enough that individual samples
    reach 25 m/s^2, so requiring every sample to sit near 9.81 rejects the
    correct offset on the most active camera in the set. What stays true under
    motion is the *median*: gravity is always in there, and no other field in
    the message has a median near 9.8. Candidates are ranked by how close their
    median is to g, so an offset that merely overlaps the range cannot win.
    """
    if len(msgs) < 4:
        return None
    best = None
    for off in range(40, len(msgs[0]) - 24):
        try:
            vs = np.array([struct.unpack_from("<3d", m, off) for m in msgs
                           if len(m) >= off + 24])
        except struct.error:
            continue
        if len(vs) < len(msgs) // 2 or not np.isfinite(vs).all():
            continue
        mag = np.linalg.norm(vs, axis=1)
        med = float(np.median(mag))
        if not (8.5 < med < 11.0) or mag.max() > 200.0:
            continue
        err = abs(med - 9.81)
        if best is None or err < best[0]:
            best = (err, off)
    return best[1] if best else None


def find_small_offset(msgs):
    """For gyro streams: the offset whose vector actually varies between
    messages.

    Angular velocity has no constant magnitude to key on the way gravity does,
    and picking "the last plausible triple" latches onto the orientation
    quaternion instead - which reads as a rock-steady |v| = 1.000 and silently
    produces a dead channel. Variance is the property that separates a live
    sensor field from padding, covariance and an unfused quaternion.
    """
    if len(msgs) < 4:
        return None
    best = None
    for off in range(40, len(msgs[0]) - 24):
        try:
            vs = np.array([struct.unpack_from("<3d", m, off) for m in msgs
                           if len(m) >= off + 24])
        except struct.error:
            continue
        if len(vs) < len(msgs) // 2 or not np.isfinite(vs).all():
            continue
        if np.abs(vs).max() > 50.0:
            continue
        mag = np.linalg.norm(vs, axis=1)
        if mag.std() < 1e-6:
            continue
        if best is None or mag.std() > best[0]:
            best = (mag.std(), off)
    return best[1] if best else None


def main():
    mcap = sys.argv[1]
    out_path = sys.argv[2]
    want = sys.argv[3:]

    print("scanning %s" % mcap, flush=True)
    t_start = time.time()

    # pass 1: collect a few messages per wanted topic to locate the fields
    chans = {}
    probe = {t: [] for t in want}
    with open(mcap, "rb") as fh:
      try:
        for rec in StreamReader(fh, skip_magic=False).records:
            if isinstance(rec, Channel):
                chans[rec.id] = rec.topic
            elif isinstance(rec, Message):
                t = chans.get(rec.channel_id)
                if t in probe and len(probe[t]) < 60:
                    probe[t].append(bytes(rec.data))
                    if all(len(v) >= 60 for v in probe.values()):
                        break
      except Exception as e:
        print("  probe stopped early (%s)" % type(e).__name__, flush=True)

    offsets = {}
    for t in want:
        if not probe[t]:
            print("  %-42s NOT FOUND" % t)
            continue
        acc_off = find_vec_offset(probe[t])
        if acc_off is not None:
            offsets[t] = ("accel", acc_off, acc_off - 96)
            print("  %-42s accel @%d  gyro @%d" % (t, acc_off, acc_off - 96))
        else:
            g = find_small_offset(probe[t])
            offsets[t] = ("gyro", None, g)
            print("  %-42s gyro-only @%s" % (t, g))

    # pass 2: the real read
    data = {t: {"t": [], "acc": [], "gyro": []} for t in offsets}
    want_ids = set()
    chans = {}
    t0 = None
    n = total = dropped = 0
    last = time.time()
    truncated = False
    with open(mcap, "rb") as fh:
      try:
        for rec in StreamReader(fh, skip_magic=False).records:
            if isinstance(rec, Channel):
                chans[rec.id] = rec.topic
                if rec.topic in offsets:
                    want_ids.add(rec.id)
            elif isinstance(rec, Message):
                n += 1
                if t0 is None:
                    t0 = rec.log_time
                if rec.channel_id not in want_ids:
                    continue
                topic = chans[rec.channel_id]
                kind, ao, go = offsets[topic]
                buf = rec.data
                acc = (0.0, 0.0, 0.0)
                gyro = (0.0, 0.0, 0.0)
                try:
                    if ao is not None and len(buf) >= ao + 24:
                        acc = struct.unpack_from("<3d", buf, ao)
                        m = acc[0] ** 2 + acc[1] ** 2 + acc[2] ** 2
                        if not (0.25 < m < 40000.0):   # |a| in 0.5 .. 200
                            dropped += 1
                            continue
                    if go is not None and len(buf) >= go + 24:
                        g = struct.unpack_from("<3d", buf, go)
                        if all(np.isfinite(x) and abs(x) < 100.0 for x in g):
                            gyro = g
                except struct.error:
                    dropped += 1
                    continue
                data[topic]["t"].append((rec.log_time - t0) / 1e9)
                data[topic]["acc"].append(acc)
                data[topic]["gyro"].append(gyro)
                total += 1
                if time.time() - last > 30:
                    print("  %.0fs: %d samples, %d msgs, t=%.1fs"
                          % (time.time() - t_start, total, n,
                             (rec.log_time - t0) / 1e9), flush=True)
                    last = time.time()
      except Exception as e:
        # several files in this set have a corrupt tail: the footer reports an
        # impossible record length and the last chunk is cut short. Everything
        # decoded before that point is still valid, so keep it and say so
        # rather than discarding a whole scan over the final partial record.
        truncated = True
        print("\n  stream ended early (%s: %s) - keeping what was read"
              % (type(e).__name__, e), flush=True)

    print("\nscan finished in %.0f s (%d messages, %d dropped)%s"
          % (time.time() - t_start, n, dropped,
             "  [TRUNCATED FILE]" if truncated else ""))

    out = {}
    for topic, v in data.items():
        t = np.array(v["t"], np.float64)
        a = np.array(v["acc"], np.float32)
        g = np.array(v["gyro"], np.float32)
        key = topic.strip("/").replace("/", "_")
        print("  %-42s %7d samples" % (topic, len(t)), end="")
        if len(t) > 1:
            am = np.linalg.norm(a, axis=1)
            gm = np.linalg.norm(g, axis=1)
            print("  %.1fs %.0fHz  |a| %.2f-%.2f  |g| %.3f-%.3f"
                  % (t[-1] - t[0], len(t) / max(t[-1] - t[0], 1e-9),
                     am.min(), am.max(), gm.min(), gm.max()))
        else:
            print()
        out[key + "_t"] = t
        out[key + "_acc"] = a
        out[key + "_gyro"] = g

        # a channel whose magnitude never changes is a misread field, not a
        # still sensor - flag it rather than let it become a dead feature
        for name, arr in (("acc", a), ("gyro", g)):
            if len(arr) > 10:
                m = np.linalg.norm(arr, axis=1)
                if m.std() < 1e-6 and m.mean() > 1e-9:
                    print("      WARNING: %s is constant at %.4f - offset is "
                          "probably wrong" % (name, m.mean()))

    np.savez_compressed(out_path, **out)
    print("\nsaved:", out_path)


if __name__ == "__main__":
    main()
