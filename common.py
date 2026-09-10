"""Shared pieces: ground truth, motion features, metrics, resource sampling.

These started life inside the first analysis script and were imported from
there by everything else, which meant running a pipeline pulled in an
experiment. They are the same functions, moved somewhere honest.
"""
import re
import subprocess
import threading

import numpy as np

HZ = 10          # tick resolution for every mask and metric in the project


# --------------------------------------------------------------- ground truth

def load_gt(path):
    """Read the recorder's metadata.json.

    The file is JSON-shaped but is not JSON: keys and string values are
    unquoted, and the descriptions contain commas and apostrophes, so no
    regex-to-JSON repair survives them. The layout is one `key: value` per
    line, which is unambiguous, so it is read line by line.

    Returns a list of {start, end, title, description, work} in seconds from
    the start of the clip.
    """
    t0 = None
    segs, cur = [], {}
    for line in open(path):
        line = line.strip().rstrip(',')
        if line == '}' and cur.get("start_ts") is not None:
            segs.append(cur)
            cur = {}
            continue
        if ':' not in line:
            continue
        k, _, v = line.partition(':')
        k, v = k.strip(), v.strip().rstrip(',').strip()
        if k == "start_ts" and t0 is None and not cur:
            t0 = int(v)
        elif k in ("start_ts", "end_ts"):
            cur[k] = int(v)
        elif k in ("title", "description"):
            cur[k] = None if v == "null" else v
    if cur.get("start_ts") is not None:
        segs.append(cur)

    out = []
    for s in segs:
        if "start_ts" not in s or "end_ts" not in s:
            continue
        title = str(s.get("title") or "idle")
        out.append({"start": (s["start_ts"] - t0) / 1000.0,
                    "end": (s["end_ts"] - t0) / 1000.0,
                    "title": title,
                    "description": s.get("description"),
                    "work": title.strip().lower() != "idle"})
    out.sort(key=lambda s: s["start"])
    return out


def gt_mask(segs, dur):
    n = int(dur * HZ)
    m = np.zeros(n, bool)
    for s in segs:
        if s["work"]:
            m[int(s["start"] * HZ):min(int(s["end"] * HZ), n)] = True
    return m


# ------------------------------------------------------------------- metrics

def auc(score, gt):
    """Probability a random work tick outscores a random idle tick.

    Threshold-free, so a feature that carries signal cannot be hidden by a
    badly chosen cut point - which matters here because the VLM's scores are
    compressed into a narrow band and every absolute threshold behaves the
    same. 0.5 means the ordering carries nothing.
    """
    w, i = score[gt], score[~gt]
    if len(w) == 0 or len(i) == 0:
        return float("nan")
    allv = np.concatenate([w, i])
    order = np.argsort(allv, kind="mergesort")
    ranks = np.empty(len(order), np.float64)
    ranks[order] = np.arange(1, len(order) + 1)
    sorted_v = allv[order]
    k = 0
    while k < len(sorted_v):                      # average ranks over ties
        j = k
        while j + 1 < len(sorted_v) and sorted_v[j + 1] == sorted_v[k]:
            j += 1
        if j > k:
            ranks[order[k:j + 1]] = ranks[order[k:j + 1]].mean()
        k = j + 1
    rw = ranks[:len(w)].sum()
    return float((rw - len(w) * (len(w) + 1) / 2) / (len(w) * len(i)))


def prf(pred, gt):
    """Precision, recall, F1 and accuracy at tick level."""
    tp = int((pred & gt).sum())
    fp = int((pred & ~gt).sum())
    fn = int((~pred & gt).sum())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f, float((pred == gt).mean())


def always_yes(gt):
    """The bar any gate must clear: predict work everywhere.

    On a clip that is 57% work this scores F1 0.729 while knowing nothing, so
    quoting F1 without it is misleading.
    """
    return prf(np.ones(len(gt), bool), gt)


# ------------------------------------------------------------- motion features

def resample(t, v, n):
    """Mean of v within each tick, gaps filled by interpolation."""
    idx = np.clip((t * HZ).astype(int), 0, n - 1)
    out, cnt = np.zeros(n), np.zeros(n)
    np.add.at(out, idx, v)
    np.add.at(cnt, idx, 1)
    good = cnt > 0
    out[good] /= cnt[good]
    if (~good).any() and good.any():
        out[~good] = np.interp(np.flatnonzero(~good), np.flatnonzero(good),
                               out[good])
    return out


def smooth(x, win_s):
    w = max(1, int(win_s * HZ))
    return np.convolve(x, np.ones(w) / w, mode="same")


def features(t, acc, gyro, n, dur=None):
    """Per-tick motion features from one IMU stream.

    acc_std   spread of |acceleration| within the tick - motion in any
              direction, which a mean would cancel out
    gyro_mag  rotation rate; turning a knob or a screwdriver produces it and
              walking does not
    jerk      |d|a|/dt|, which penalises smooth carrying relative to
              manipulation
    """
    mag = np.linalg.norm(acc, axis=1)
    gmag = np.linalg.norm(gyro, axis=1)
    dt = np.diff(t, prepend=t[0])
    dt[dt <= 0] = 1e-3
    jerk = np.abs(np.diff(mag, prepend=mag[0])) / dt

    m1 = resample(t, mag, n)
    m2 = resample(t, mag * mag, n)
    return {
        "acc_std": np.sqrt(np.maximum(m2 - m1 * m1, 0)),
        "gyro_mag": resample(t, gmag, n),
        "jerk": resample(t, np.minimum(jerk, 500.0), n),
    }


def z(x):
    s = x.std()
    return (x - x.mean()) / s if s > 1e-9 else x * 0


# ------------------------------------------------------------------- spans

def spans_from_mask(mask, dur):
    out, n, i = [], len(mask), 0
    while i < n:
        if mask[i]:
            j = i
            while j < n and mask[j]:
                j += 1
            out.append({"start": round(i / HZ, 1),
                        "end": round(min(j / HZ, dur), 1)})
            i = j
        else:
            i += 1
    return out


def postprocess(mask, dur, min_span=3.0, merge_gap=2.0, max_span=None):
    """Mask to spans: bridge short gaps, drop short spans, split long ones."""
    merged = []
    for s in spans_from_mask(mask, dur):
        if merged and s["start"] - merged[-1]["end"] <= merge_gap:
            merged[-1]["end"] = s["end"]
        else:
            merged.append(dict(s))
    merged = [s for s in merged if s["end"] - s["start"] >= min_span]

    if not max_span:
        return merged
    out = []                       # one label per span only works if the span
    for s in merged:               # holds one action
        d = s["end"] - s["start"]
        if d <= max_span * 1.4:
            out.append(s)
            continue
        k = max(1, int(round(d / max_span)))
        edges = np.linspace(s["start"], s["end"], k + 1)
        for a, b in zip(edges, edges[1:]):
            out.append({"start": round(float(a), 1), "end": round(float(b), 1)})
    return out


def mask_from_spans(spans, dur):
    n = int(dur * HZ)
    m = np.zeros(n, bool)
    for s in spans:
        m[int(s["start"] * HZ):min(int(s["end"] * HZ), n)] = True
    return m


# ---------------------------------------------------------------- resources

class Tegra(threading.Thread):
    """Sample tegrastats in the background so a run reports its own load."""

    def __init__(self, interval_ms=1000):
        super().__init__(daemon=True)
        self.rows = []
        self.stop = threading.Event()
        self.proc = None
        self.interval = interval_ms

    def run(self):
        try:
            self.proc = subprocess.Popen(
                ["tegrastats", "--interval", str(self.interval)],
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
        except Exception:
            return                        # not a Jetson, or tegrastats absent
        for line in self.proc.stdout:
            if self.stop.is_set():
                break
            r = {}
            m = re.search(r'RAM (\d+)/(\d+)MB', line)
            if m:
                r["ram_mb"] = int(m.group(1))
                r["ram_total_mb"] = int(m.group(2))
            m = re.search(r'GR3D_FREQ (\d+)%', line)
            if m:
                r["gpu_pct"] = int(m.group(1))
            m = re.search(r'CPU \[([^\]]+)\]', line)
            if m:
                vals = [int(x.split('%')[0]) for x in m.group(1).split(',')
                        if '%' in x]
                if vals:
                    r["cpu_pct_mean"] = round(sum(vals) / len(vals), 1)
            m = re.search(r'tj@([\d.]+)C', line)
            if m:
                r["tj_c"] = float(m.group(1))
            m = re.search(r'VDD_IN (\d+)mW', line)
            if m:
                r["vdd_in_mw"] = int(m.group(1))
            if r:
                self.rows.append(r)

    def finish(self):
        self.stop.set()
        if self.proc:
            try:
                self.proc.terminate()
            except Exception:
                pass

    def summary(self):
        if not self.rows:
            return {}
        out = {"samples": len(self.rows)}
        for k in ("ram_mb", "gpu_pct", "cpu_pct_mean", "tj_c", "vdd_in_mw"):
            v = [r[k] for r in self.rows if k in r]
            if v:
                out[k + "_mean"] = round(sum(v) / len(v), 1)
                out[k + "_max"] = max(v)
        if "ram_total_mb" in self.rows[0]:
            out["ram_total_mb"] = self.rows[0]["ram_total_mb"]
        return out
