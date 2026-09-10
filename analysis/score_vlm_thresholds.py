"""Is there any signal in p_yes at all, once the absolute level is discarded?

The sweep showed every threshold from 0.30 to 0.85 producing the identical
mask, because p_yes never dropped below 0.833. That kills absolute
thresholding, but it does not by itself prove the score is uninformative: if
the ordering were right, a per-clip relative threshold would still recover the
work spans.

So this asks the only question that matters - AUC, the probability that a
randomly chosen work tick scores above a randomly chosen idle tick. 0.5 means
the score carries nothing. Everything else here (percentile thresholds,
z-scores, smoothing) is downstream of that number and cannot beat it.

The always-yes baseline is printed alongside, because on a clip that is 57%
work by construction a useless model still scores F1 0.729.
"""
import json
import numpy as np

SEG = "/home/user/VLM_WORKSPACE/gas_segmentation.json"
HZ = 10


def prf(pred, gt):
    tp = int((pred & gt).sum())
    fp = int((pred & ~gt).sum())
    fn = int((~pred & gt).sum())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f, float((pred == gt).mean())


def main():
    d = json.load(open(SEG))
    dur = d["timing"]["video_dur_s"]
    n = int(dur * HZ)

    acc, cnt = np.zeros(n), np.zeros(n)
    for w in d["windows"]:
        a, b = int(w["start"] * HZ), min(int(w["end"] * HZ), n)
        acc[a:b] += w["p_yes"]
        cnt[a:b] += 1
    s = np.where(cnt > 0, acc / np.maximum(cnt, 1), 0.0)

    gt = np.zeros(n, bool)
    for g in d["gt"]:
        if g["work"]:
            gt[int(g["start"] * HZ):min(int(g["end"] * HZ), n)] = True

    p = np.array([w["p_yes"] for w in d["windows"]])
    print("p_yes distribution over %d windows:" % len(p))
    print("  min %.3f  p10 %.3f  median %.3f  p90 %.3f  max %.3f"
          % (p.min(), np.percentile(p, 10), np.median(p),
             np.percentile(p, 90), p.max()))
    print("  range %.3f" % (p.max() - p.min()))

    pr, rc, f1, ac = prf(np.ones(n, bool), gt)
    print("\nALWAYS-YES baseline : P %.3f  R %.3f  F1 %.3f  acc %.3f"
          % (pr, rc, f1, ac))
    print("(any model must beat this to be worth running)")

    # the decisive number
    w_s, i_s = s[gt], s[~gt]
    gtr = (w_s[:, None] > i_s[None, :]).mean()
    eq = (w_s[:, None] == i_s[None, :]).mean()
    auc = gtr + 0.5 * eq
    print("\n=== RANK SEPARATION ===")
    print("  AUC          %.4f   (0.500 = no signal, 1.0 = perfect)" % auc)
    print("  correlation  %.4f" % np.corrcoef(s, gt.astype(float))[0, 1])
    print("  mean on work %.4f" % w_s.mean())
    print("  mean on idle %.4f" % i_s.mean())
    print("  gap          %.4f" % (w_s.mean() - i_s.mean()))
    print("  std overall  %.4f" % s.std())

    print("\n=== PERCENTILE THRESHOLD (relative, per clip) ===")
    best = None
    for q in range(5, 100, 5):
        thr = np.percentile(s, q)
        pred = s >= thr
        pr, rc, f1, ac = prf(pred, gt)
        if best is None or f1 > best[0]:
            best = (f1, q, thr, pr, rc, ac)
        print("  q%2d  thr %.4f  P %.3f  R %.3f  F1 %.3f  acc %.3f"
              % (q, thr, pr, rc, f1, ac))
    print("  best F1 %.3f at q%d (thr %.4f)" % (best[0], best[1], best[2]))

    # inverted, in case the model is anti-correlated
    print("\n=== SAME, INVERTED (low p_yes = work) ===")
    bi = None
    for q in range(5, 100, 5):
        thr = np.percentile(s, q)
        pred = s <= thr
        pr, rc, f1, ac = prf(pred, gt)
        if bi is None or f1 > bi[0]:
            bi = (f1, q, thr)
    print("  best F1 %.3f at q%d" % (bi[0], bi[1]))

    print("\n=== VERDICT ===")
    if auc < 0.55:
        print("  AUC %.3f - p_yes does NOT separate work from idle on this" % auc)
        print("  clip. No threshold, smoothing or calibration can recover a")
        print("  signal that is not in the ordering. A different input")
        print("  (wrist IMU, optical flow) is required, not a different prompt.")
    elif auc < 0.65:
        print("  AUC %.3f - weak but non-zero ordering. Relative thresholding" % auc)
        print("  is worth keeping; absolute thresholds are not.")
    else:
        print("  AUC %.3f - usable ordering, only the calibration was wrong." % auc)


if __name__ == "__main__":
    main()
