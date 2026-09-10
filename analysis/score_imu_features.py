"""Does wrist IMU separate work from idle where the VLM could not?

The VLM scored AUC 0.611 on the GAS clip - barely above chance, and no
threshold on it beat an always-yes baseline. The claim being tested here is
that the failure was one of input, not of prompting: a camera sees the worker
standing at the stove whether or not their hands are moving, while a wrist
accelerometer measures the movement directly.

Several features are scored, all cheap:

  acc_std     rolling std of |acceleration| - motion regardless of direction
  gyro_mag    rolling mean of |angular velocity| - rotation, which turning a
              knob or a screwdriver produces and walking does not
  jerk        rolling mean of |d(acc)/dt| - penalises smooth carrying motion
              relative to manipulation

Each is scored by AUC against ground truth, which is threshold-free, so a
feature that works cannot be hidden by a badly chosen cut point. Both wrists
are scored separately and combined, since the dominant hand does most of the
work and averaging the two can dilute it.
"""
import json
import re
import sys

import numpy as np

NPZ = sys.argv[1] if len(sys.argv) > 1 else "/home/user/VLM_WORKSPACE/crm_imu.npz"
META = sys.argv[2] if len(sys.argv) > 2 else (
    "/home/user/VLM_WORKSPACE/SAMPLE_MCAP_FROM_CRM/5e19608c34b6a880_metadata.json")
HZ = 10


def load_gt(path):
    """The metadata is JSON-shaped but unquoted; read it line by line."""
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
                    "work": title.strip().lower() != "idle"})
    out.sort(key=lambda s: s["start"])
    return out


def auc(score, gt):
    """Rank-based AUC via the Mann-Whitney statistic - no thresholds involved."""
    w, i = score[gt], score[~gt]
    if len(w) == 0 or len(i) == 0:
        return float("nan")
    order = np.argsort(np.concatenate([w, i]), kind="mergesort")
    ranks = np.empty(len(order), np.float64)
    ranks[order] = np.arange(1, len(order) + 1)
    # average ranks for ties
    allv = np.concatenate([w, i])[order]
    k = 0
    while k < len(allv):
        j = k
        while j + 1 < len(allv) and allv[j + 1] == allv[k]:
            j += 1
        if j > k:
            ranks[order[k:j + 1]] = ranks[order[k:j + 1]].mean()
        k = j + 1
    rw = ranks[:len(w)].sum()
    return float((rw - len(w) * (len(w) + 1) / 2) / (len(w) * len(i)))


def prf(pred, gt):
    tp = int((pred & gt).sum())
    fp = int((pred & ~gt).sum())
    fn = int((~pred & gt).sum())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f, float((pred == gt).mean())


def resample(t, v, n, dur):
    """Mean of v within each tick."""
    idx = np.clip((t * HZ).astype(int), 0, n - 1)
    out = np.zeros(n)
    cnt = np.zeros(n)
    np.add.at(out, idx, v)
    np.add.at(cnt, idx, 1)
    good = cnt > 0
    out[good] /= cnt[good]
    if (~good).any():                      # fill gaps by interpolation
        out[~good] = np.interp(np.flatnonzero(~good), np.flatnonzero(good),
                               out[good])
    return out


def smooth(x, win_s):
    w = max(1, int(win_s * HZ))
    k = np.ones(w) / w
    return np.convolve(x, k, mode="same")


def features(t, acc, gyro, n, dur):
    """Per-tick motion features from one IMU stream."""
    mag = np.linalg.norm(acc, axis=1)
    gmag = np.linalg.norm(gyro, axis=1)
    dt = np.diff(t, prepend=t[0])
    dt[dt <= 0] = 1e-3
    jerk = np.abs(np.diff(mag, prepend=mag[0])) / dt

    # per-tick std of |a| needs E[x^2] - E[x]^2 within the tick
    m1 = resample(t, mag, n, dur)
    m2 = resample(t, mag * mag, n, dur)
    astd = np.sqrt(np.maximum(m2 - m1 * m1, 0))

    return {
        "acc_std": astd,
        "gyro_mag": resample(t, gmag, n, dur),
        "jerk": resample(t, np.minimum(jerk, 500.0), n, dur),
    }


def main():
    d = np.load(NPZ)
    gt_segs = load_gt(META)
    dur = max(s["end"] for s in gt_segs)
    n = int(dur * HZ)
    gt = np.zeros(n, bool)
    for s in gt_segs:
        if s["work"]:
            gt[int(s["start"] * HZ):min(int(s["end"] * HZ), n)] = True

    print("clip %.0f s, GT %d segments, work %.1f%%"
          % (dur, len(gt_segs), 100 * gt.mean()))
    ws = [s for s in gt_segs if s["work"]]
    if ws:
        dw = [s["end"] - s["start"] for s in ws]
        print("work spans %d, dur min/med/max %.1f/%.1f/%.1f s"
              % (len(ws), min(dw), float(np.median(dw)), max(dw)))

    streams = {}
    for key in ("imu_wrist_left", "imu_wrist_right", "imu_head"):
        if key + "_t" not in d:
            continue
        t, a, g = d[key + "_t"], d[key + "_acc"], d[key + "_gyro"]
        if len(t) < 10:
            continue
        streams[key] = features(t, a, g, n, dur)
        print("  %-18s %d samples" % (key, len(t)))

    print("\n=== AUC BY FEATURE AND SMOOTHING WINDOW ===")
    print("  (0.500 = no signal; the VLM scored 0.611 on the GAS clip)")
    rows = []
    for key, feats in streams.items():
        for fname, raw in feats.items():
            for win in (1.0, 2.0, 4.0, 8.0):
                s = smooth(raw, win)
                a = auc(s, gt)
                rows.append((a, key, fname, win, s))
                print("  %-18s %-9s win %4.1fs   AUC %.4f"
                      % (key, fname, win, a))

    # both wrists together - the dominant hand does most of the work
    if "imu_wrist_left" in streams and "imu_wrist_right" in streams:
        print("\n=== BOTH WRISTS COMBINED ===")
        for fname in ("acc_std", "gyro_mag", "jerk"):
            for win in (2.0, 4.0, 8.0):
                l = smooth(streams["imu_wrist_left"][fname], win)
                r = smooth(streams["imu_wrist_right"][fname], win)
                for how, s in (("max", np.maximum(l, r)),
                               ("mean", (l + r) / 2)):
                    a = auc(s, gt)
                    rows.append((a, "wrists_" + how, fname, win, s))
                    print("  wrists %-5s %-9s win %4.1fs   AUC %.4f"
                          % (how, fname, win, a))

    rows.sort(key=lambda r: -r[0])
    best_auc, bkey, bfeat, bwin, bscore = rows[0]
    print("\n=== BEST FEATURE ===")
    print("  %s / %s / %.1f s window   AUC %.4f" % (bkey, bfeat, bwin, best_auc))

    print("\n=== THRESHOLD SWEEP ON THE BEST FEATURE ===")
    always = prf(np.ones(n, bool), gt)
    print("  always-yes baseline: P %.3f R %.3f F1 %.3f acc %.3f" % always)
    best_f1 = None
    for q in range(5, 100, 5):
        thr = np.percentile(bscore, q)
        p, r, f, ac = prf(bscore >= thr, gt)
        if best_f1 is None or f > best_f1[0]:
            best_f1 = (f, q, thr, p, r, ac)
        print("  q%2d thr %8.4f  P %.3f R %.3f F1 %.3f acc %.3f"
              % (q, thr, p, r, f, ac))
    print("  best F1 %.3f at q%d (thr %.4f), acc %.3f"
          % (best_f1[0], best_f1[1], best_f1[2], best_f1[5]))

    out = {
        "clip_dur_s": dur, "gt_work_pct": round(100 * float(gt.mean()), 1),
        "vlm_auc_gas_clip": 0.6108,
        "always_yes": {"precision": round(always[0], 3),
                       "recall": round(always[1], 3),
                       "f1": round(always[2], 3),
                       "accuracy": round(always[3], 3)},
        "features": [{"stream": k, "feature": f, "window_s": w,
                      "auc": round(a, 4)} for a, k, f, w, _ in rows],
        "best": {"stream": bkey, "feature": bfeat, "window_s": bwin,
                 "auc": round(best_auc, 4),
                 "f1": round(best_f1[0], 3), "percentile": best_f1[1],
                 "threshold": round(float(best_f1[2]), 4),
                 "precision": round(best_f1[3], 3),
                 "recall": round(best_f1[4], 3),
                 "accuracy": round(best_f1[5], 3)},
    }
    json.dump(out, open("/home/user/VLM_WORKSPACE/imu_vs_gt.json", "w"), indent=2)
    np.save("/home/user/VLM_WORKSPACE/imu_best_score.npy", bscore)
    print("\nsaved: imu_vs_gt.json, imu_best_score.npy")

    print("\n=== VERDICT ===")
    if best_auc > 0.75:
        print("  AUC %.3f - the wrist IMU separates work from idle where the"
              % best_auc)
        print("  VLM (0.611) did not. Use IMU for the work/idle gate and keep")
        print("  the VLM for labelling only.")
    elif best_auc > 0.65:
        print("  AUC %.3f - better than the VLM but not decisive on its own."
              % best_auc)
    else:
        print("  AUC %.3f - the IMU does not solve it either on this clip."
              % best_auc)


if __name__ == "__main__":
    main()
