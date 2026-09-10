"""What is in an MCAP, when its index cannot be trusted.

Several files in this project have a corrupt footer, so get_summary() and every
seek-based path are unavailable; this walks records in order instead and stops
after a bounded number of messages, which is enough to enumerate the channels
and estimate their rates.
"""
import sys
import time

from mcap.stream_reader import StreamReader
from mcap.records import Channel, Message, Schema

MCAP = sys.argv[1]
LIMIT = int(sys.argv[2]) if len(sys.argv) > 2 else 20000

chans, schemas = {}, {}
cnt, first_t, last_t, sizes = {}, {}, {}, {}
t0 = None
n = 0
t_start = time.time()

with open(MCAP, "rb") as fh:
    for rec in StreamReader(fh, skip_magic=False).records:
        if isinstance(rec, Schema):
            schemas[rec.id] = rec.name
        elif isinstance(rec, Channel):
            chans[rec.id] = (rec.topic, rec.schema_id)
        elif isinstance(rec, Message):
            n += 1
            if t0 is None:
                t0 = rec.log_time
            topic, sid = chans.get(rec.channel_id, ("?", 0))
            t = (rec.log_time - t0) / 1e9
            cnt[topic] = cnt.get(topic, 0) + 1
            sizes[topic] = sizes.get(topic, 0) + len(rec.data)
            first_t.setdefault(topic, t)
            last_t[topic] = t
            if n >= LIMIT:
                break

print("scanned %d messages in %.0f s" % (n, time.time() - t_start))
print("%-42s %-38s %7s %8s %10s" % ("topic", "schema", "count", "Hz", "avg bytes"))
for topic in sorted(cnt):
    sid = chans_sid = None
    for cid, (tp, s) in chans.items():
        if tp == topic:
            sid = s
            break
    span = max(last_t[topic] - first_t[topic], 1e-9)
    print("%-42s %-38s %7d %8.1f %10d"
          % (topic[:42], str(schemas.get(sid))[:38], cnt[topic],
             cnt[topic] / span, sizes[topic] // max(cnt[topic], 1)))
print("\ntime span covered by this sample: %.1f s" % max(last_t.values()))
