"""Locate the angular_velocity field in the Orbbec gyro topic.

The accel topic was easy - gravity is a fingerprint. Angular velocity has no
such constant, and the previous heuristic latched onto the orientation
quaternion instead, which is why every sample came back |g| = 1.000 exactly.

The distinguishing property of the real gyro field is that it *varies* across
messages while the padding, covariance and (for a device that does not fuse
orientation) the quaternion do not. So this samples many messages and reports,
for each candidate offset, the standard deviation across them: the gyro is the
offset with meaningful variance and physically sane magnitudes.
"""
import struct
import sys

import numpy as np
from mcap.stream_reader import StreamReader
from mcap.records import Channel, Message

MCAP = sys.argv[1] if len(sys.argv) > 1 else "/home/user/actseg/sample.mcap"
TOPIC = sys.argv[2] if len(sys.argv) > 2 else "/camera_CPA9B520080/imu/gyro"
N = 200

chans, msgs = {}, []
with open(MCAP, "rb") as fh:
    for rec in StreamReader(fh, skip_magic=False).records:
        if isinstance(rec, Channel):
            chans[rec.id] = rec.topic
        elif isinstance(rec, Message):
            if chans.get(rec.channel_id) == TOPIC:
                msgs.append(bytes(rec.data))
                if len(msgs) >= N:
                    break

print("topic %s  %d messages  len=%d" % (TOPIC, len(msgs), len(msgs[0])))
o = 4 + 8
(slen,) = struct.unpack_from("<I", msgs[0], o)
frame = msgs[0][o + 4:o + 4 + slen].rstrip(b"\x00").decode()
end = o + 4 + slen
print("frame_id '%s' ends %d (aligned %d)" % (frame, end, (end + 7) & ~7))

print("\noffset  mean|v|    std|v|   max|v|   sample")
cands = []
for off in range(end, len(msgs[0]) - 24):
    try:
        vs = np.array([struct.unpack_from("<3d", m, off) for m in msgs
                       if len(m) >= off + 24])
    except struct.error:
        continue
    if len(vs) < N // 2 or not np.isfinite(vs).all():
        continue
    if np.abs(vs).max() > 50.0:
        continue
    mag = np.linalg.norm(vs, axis=1)
    if mag.std() < 1e-9:
        continue
    cands.append((mag.std(), off, mag.mean(), mag.max(), vs[0]))

cands.sort(key=lambda c: -c[0])
for std, off, mean, mx, first in cands[:12]:
    print("%6d  %8.4f  %8.4f  %7.3f   %s"
          % (off, mean, std, mx, " ".join("%+.4f" % x for x in first)))

if cands:
    print("\nmost variable offset: %d" % cands[0][1])
    print("(angular velocity in rad/s should be roughly 0-5 for a head-worn "
          "camera, and must vary between samples)")
else:
    print("\nno varying triple found - the gyro topic may carry its data in "
          "the same field the accel topic uses")
