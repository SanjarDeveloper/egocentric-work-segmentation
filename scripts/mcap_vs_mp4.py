"""Measure: read the 16 frames of a window from MCAP directly vs from the MP4.

The pipeline currently seeks into an 810 MB MP4 with cv2.CAP_PROP_POS_MSEC.
The MCAP is 11.1 GB and holds the same colour stream as JPEG-compressed
sensor_msgs/CompressedImage, so reading it needs no H.264 decode at all -
but it does need a scan to find the messages.

Both paths are asked for the same thing: 16 frames 0.5 s apart from a few
window starts, resized to 256 px, ready to hand to the model.
"""
import sys, time
import cv2
import numpy as np

MP4 = "/home/user/VLM_WORKSPACE/SAMPLE_MCAP_FROM_CRM/5e19608c34b6a880_ego.mp4"
MCAP = "/home/user/VLM_WORKSPACE/SAMPLE_MCAP_FROM_CRM/5e19608c34b6a880_merged.mcap"

NFRAMES = 16
WINDOW = 8.0
PX = 256
STARTS = [0.0, 88.0, 176.0, 264.0]


def resize(bgr):
    h, w = bgr.shape[:2]
    sc = PX / max(h, w)
    return cv2.resize(bgr, (int(w * sc), int(h * sc)), interpolation=cv2.INTER_AREA)


def read_mp4(start):
    cap = cv2.VideoCapture(MP4)
    gap = WINDOW / NFRAMES
    out = []
    for k in range(NFRAMES):
        cap.set(cv2.CAP_PROP_POS_MSEC, (start + k * gap) * 1000)
        ok, f = cap.read()
        if ok:
            out.append(resize(f))
    cap.release()
    return out


def read_mp4_shared(cap, start):
    """Same, but reusing one already-open handle - what the pipeline does."""
    gap = WINDOW / NFRAMES
    out = []
    for k in range(NFRAMES):
        cap.set(cv2.CAP_PROP_POS_MSEC, (start + k * gap) * 1000)
        ok, f = cap.read()
        if ok:
            out.append(resize(f))
    return out


def find_topic():
    from mcap.reader import make_reader
    with open(MCAP, "rb") as fh:
        r = make_reader(fh)
        summary = r.get_summary()
        if summary is None:
            return None, {}
        chans = {}
        for ch in summary.channels.values():
            sch = summary.schemas.get(ch.schema_id)
            chans[ch.topic] = sch.name if sch else "?"
        # colour ego stream only: depth and wrist cameras are different data
        img = [t for t, sc in chans.items()
               if "ego" in t and "depth" not in t
               and ("image_compressed" in t or "image_raw/compressed" in t)]
        if not img:
            img = [t for t, sc in chans.items()
                   if "image_raw/compressed" in t or "CompressedImage" in sc]
        return (img[0] if img else None), chans


def decode_ros_compressed(data):
    """Try JPEG first; report H.264 rather than silently returning nothing."""
    i = data.find(b"\xff\xd8\xff")
    if i >= 0:
        arr = np.frombuffer(data[i:], np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return None


def payload_kind(data):
    if data.find(b"\xff\xd8\xff") >= 0:
        return "jpeg"
    if data.find(b"\x00\x00\x00\x01") >= 0 or data.find(b"\x00\x00\x01") >= 0:
        return "h264/h265 (annex-b)"
    return "unknown"


def read_mcap(topic, start, t0_ns):
    """Frames in [start, start+WINDOW) from one sequential pass."""
    from mcap.reader import make_reader
    gap = WINDOW / NFRAMES
    wanted = [start + k * gap for k in range(NFRAMES)]
    out, wi = [], 0
    with open(MCAP, "rb") as fh:
        r = make_reader(fh)
        s_ns = int(t0_ns + start * 1e9)
        e_ns = int(t0_ns + (start + WINDOW) * 1e9)
        for _sch, _ch, msg in r.iter_messages(topics=[topic],
                                              start_time=s_ns, end_time=e_ns):
            if wi >= len(wanted):
                break
            t = (msg.log_time - t0_ns) / 1e9
            if t + 1e-9 >= wanted[wi]:
                img = decode_ros_compressed(msg.data)
                if img is not None:
                    out.append(resize(img))
                wi += 1
    return out


def main():
    print("=== MP4 (fresh handle per window) ===", flush=True)
    lat = []
    for st in STARTS:
        t0 = time.time()
        fr = read_mp4(st)
        ms = (time.time() - t0) * 1000
        lat.append(ms)
        print("  t=%3.0f s -> %2d frames | %7.0f ms" % (st, len(fr), ms), flush=True)
    print("  median %.0f ms" % np.median(lat))

    print("\n=== MP4 (one open handle - what the pipeline does) ===", flush=True)
    cap = cv2.VideoCapture(MP4)
    lat2 = []
    for st in STARTS:
        t0 = time.time()
        fr = read_mp4_shared(cap, st)
        ms = (time.time() - t0) * 1000
        lat2.append(ms)
        print("  t=%3.0f s -> %2d frames | %7.0f ms" % (st, len(fr), ms), flush=True)
    cap.release()
    print("  median %.0f ms" % np.median(lat2))

    print("\n=== MCAP ===", flush=True)
    try:
        topic, chans = find_topic()
    except Exception as e:
        print("  mcap library missing:", str(e)[:80])
        return
    print("  channels:", list(chans.items())[:6])
    if not topic:
        print("  no compressed image topic found")
        return
    print("  topic:", topic, flush=True)
    from mcap.reader import make_reader as _mr
    with open(MCAP, "rb") as fh:
        m0 = next(_mr(fh).iter_messages(topics=[topic]))[2]
    print("  payload: %s (%d bayt)" % (payload_kind(m0.data), len(m0.data)), flush=True)

    from mcap.reader import make_reader
    t0 = time.time()
    with open(MCAP, "rb") as fh:
        r = make_reader(fh)
        first = next(r.iter_messages(topics=[topic]))
        t0_ns = first[2].log_time
    print("  first message found: %.0f ms" % ((time.time() - t0) * 1000), flush=True)

    lat3 = []
    for st in STARTS:
        t0 = time.time()
        try:
            fr = read_mcap(topic, st, t0_ns)
        except Exception as e:
            print("  t=%3.0f s -> ERROR %s" % (st, str(e)[:60]))
            continue
        ms = (time.time() - t0) * 1000
        lat3.append(ms)
        print("  t=%3.0f s -> %2d frames | %7.0f ms" % (st, len(fr), ms), flush=True)
    if lat3:
        print("  median %.0f ms" % np.median(lat3))
    got = sum(1 for x in lat3) and None

    print("\n=== SUMMARY ===")
    print("  NOTE: if MCAP returned 0 frames its timing is meaningless")
    print("  MP4  (shared handle): %.0f ms/window" % np.median(lat2))
    if lat3:
        print("  MCAP                : %.0f ms/window" % np.median(lat3))
        print("  ratio: MCAP %.1fx %s" % (
            np.median(lat3) / np.median(lat2),
            "sekinroq" if np.median(lat3) > np.median(lat2) else "tezroq"))
    print("  note: model inference is ~5500 ms/window - both reads are small next to it")


if __name__ == "__main__":
    main()
