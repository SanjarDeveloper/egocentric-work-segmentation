"""Find the real byte offsets of the IMU fields by brute force.

The hand-computed CDR offsets produced 1e150 nonsense, so rather than guess
again at the alignment padding, slide an 8-byte float window across the whole
message and print every offset where three consecutive doubles look like a
gravity vector. Whatever offset holds |a| ~ 9.81 in every sample is the
linear_acceleration field, and the gyro sits a fixed distance before it.
"""
import struct
import sys

import numpy as np
from mcap.stream_reader import StreamReader
from mcap.records import Channel, Message

MCAP = sys.argv[1] if len(sys.argv) > 1 else (
    "/home/user/VLM_WORKSPACE/SAMPLE_MCAP_FROM_CRM/5e19608c34b6a880_merged.mcap")
TOPIC = "/camera_CPA9B520080/imu/accel"
N = 6


def plausible(v):
    return all(abs(x) < 1e4 and (x == 0.0 or abs(x) > 1e-12) for v_ in [v]
               for x in v_)


chans = {}
msgs = []
with open(MCAP, "rb") as fh:
    for rec in StreamReader(fh, skip_magic=False).records:
        if isinstance(rec, Channel):
            chans[rec.id] = rec.topic
        elif isinstance(rec, Message):
            if chans.get(rec.channel_id) == TOPIC:
                msgs.append(bytes(rec.data))
                if len(msgs) >= N:
                    break

print("topic %s, %d messages, len=%d" % (TOPIC, len(msgs), len(msgs[0])))
print("header hex: %s" % msgs[0][:40].hex())

# where does the frame_id string end?
o = 4 + 8
(slen,) = struct.unpack_from("<I", msgs[0], o)
frame = msgs[0][o + 4:o + 4 + slen].rstrip(b"\x00")
end = o + 4 + slen
print("frame_id '%s' ends at byte %d (next 8-aligned: %d)"
      % (frame.decode(), end, (end + 7) & ~7))

print("\noffsets where 3 consecutive doubles have |v| in [8.5, 11.5]:")
hits = []
for off in range(end, len(msgs[0]) - 24):
    mags = []
    ok = True
    for m in msgs:
        try:
            v = struct.unpack_from("<3d", m, off)
        except struct.error:
            ok = False
            break
        n = float(np.linalg.norm(v))
        if not np.isfinite(n) or not (8.5 < n < 11.5):
            ok = False
            break
        mags.append(n)
    if ok:
        hits.append(off)
        print("  offset %3d  |a| = %s" % (off, " ".join("%.3f" % x for x in mags)))
        print("              first sample %s"
              % " ".join("%+.4f" % x for x in struct.unpack_from("<3d", msgs[0], off)))

if not hits:
    print("  none - dumping all plausible small-double triples instead:")
    for off in range(end, len(msgs[0]) - 24):
        v = struct.unpack_from("<3d", msgs[0], off)
        if all(np.isfinite(x) and abs(x) < 100 for x in v) and any(v):
            print("  offset %3d  %s  |v|=%.4f"
                  % (off, " ".join("%+.5f" % x for x in v),
                     float(np.linalg.norm(v))))

print("\nfloat32 check (in case the fields are single precision):")
for off in range(end, len(msgs[0]) - 12):
    v = struct.unpack_from("<3f", msgs[0], off)
    n = float(np.linalg.norm(v))
    if np.isfinite(n) and 8.5 < n < 11.5:
        print("  f32 offset %3d  %s  |a|=%.3f"
              % (off, " ".join("%+.4f" % x for x in v), n))
