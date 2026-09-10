"""Do the VLM and the IMU fail on different ticks of the GAS clip?

Both are weak here on their own - VLM 0.611, wrist jerk 0.561 - but weakness is
not the same as redundancy. The VLM reads the scene and cannot see motion; the
accelerometer reads motion and cannot see the scene. If their errors are
uncorrelated, a combination beats both; if they are making the same mistakes,
it does not, and that is worth knowing before building anything on top.

Correlation between the two scores is reported first, because it decides
whether the rest of the numbers can mean anything. Then the fusion is scored
the same threshold-free way as everything else - AUC - across weights, with the
always-yes baseline (F1 0.729 on a clip that is 57% work) kept in view, since
that is the bar any gate must clear to be worth its runtime.

A logistic-regression fusion is fitted last. It is fitted and scored on the
same clip, so its AUC is optimistic by construction - it is reported as an
upper bound on what any linear combination could do, not as a result.
"""
import json

import numpy as np

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import common as base

VLM = "/home/user/VLM_WORKSPACE/gas_segmentation.json"
NPZ = "/home/user/VLM_WORKSPACE/gas_imu.npz"
META = "/home/user/gas/47cb14298f7e371d_metadata.json"
OUT = "/home/user/VLM_WORKSPACE/gas_fusion.json"
HZ = 10


def z(x):
    s = x.std()
    return (x - x.mean()) / s if s > 1e-9 else x * 0


def main():
    d = json.load(open(VLM))
    gt_segs = base.load_gt(META)
    dur = d["timing"]["video_dur_s"]
    n = int(dur * HZ)

    gt = np.zeros(n, bool)
    for s in gt_segs:
        if s["work"]:
            gt[int(s["start"] * HZ):min(int(s["end"] * HZ), n)] = True

    # --- VLM score per tick, from the overlapping windows
    acc, cnt = np.zeros(n), np.zeros(n)
    for w in d["windows"]:
        a, b = int(w["start"] * HZ), min(int(w["end"] * HZ), n)
        acc[a:b] += w["p_yes"]
        cnt[a:b] += 1
    vlm = np.where(cnt > 0, acc / np.maximum(cnt, 1), 0.0)

    # --- IMU features. The file is truncated at 294.5 s of 300 s, so the tail
    # has no IMU; it is filled with the clip median rather than zeros, which
    # would otherwise read as "no motion" and bias the last 5 s toward idle.
    dimu = np.load(NPZ)
    F = {k: base.features(dimu[k + "_t"], dimu[k + "_acc"], dimu[k + "_gyro"],
                          n, dur)
         for k in ("imu_wrist_left", "imu_wrist_right", "imu_head")}
    sm = lambda v, w: base.smooth(v, w)
    wrist_jerk = np.maximum(sm(F["imu_wrist_left"]["jerk"], 4.0),
                            sm(F["imu_wrist_right"]["jerk"], 4.0))
    wrist_gyro = np.maximum(sm(F["imu_wrist_left"]["gyro_mag"], 4.0),
                            sm(F["imu_wrist_right"]["gyro_mag"], 4.0))
    head_jerk = sm(F["imu_head"]["jerk"], 8.0)

    imu_t = dimu["imu_wrist_right_t"]
    covered = int(min(float(imu_t[-1]), dur) * HZ)
    if covered < n:
        print("NOTE: IMU covers %.1f s of %.1f s - filling the tail with the "
              "clip median" % (covered / HZ, dur))
        for arr in (wrist_jerk, wrist_gyro, head_jerk):
            arr[covered:] = np.median(arr[:covered])

    print("\n=== INDIVIDUAL (tick level, GT work %.1f%%) ==="
          % (100 * gt.mean()))
    singles = {"VLM p_yes": vlm, "wrist jerk": wrist_jerk,
               "wrist gyro": wrist_gyro, "head jerk": head_jerk}
    for k, v in singles.items():
        print("  %-14s AUC %.4f" % (k, base.auc(v, gt)))

    print("\n=== ARE THEY MAKING THE SAME MISTAKES? ===")
    for k, v in (("wrist jerk", wrist_jerk), ("wrist gyro", wrist_gyro),
                 ("head jerk", head_jerk)):
        c = float(np.corrcoef(vlm, v)[0, 1])
        print("  corr(VLM, %-11s) = %+.4f" % (k, c))
    print("  (near zero means the two see different things, which is the")
    print("   precondition for fusion helping at all)")

    print("\n=== FUSION: z(VLM) + w * z(IMU) ===")
    rows = []
    for iname, iv in (("wrist_jerk", wrist_jerk), ("wrist_gyro", wrist_gyro)):
        for hname, hv in (("none", np.zeros(n)), ("head_jerk", head_jerk)):
            for wi in (0.25, 0.5, 0.75, 1.0, 1.5, 2.0):
                for wh in ((0.0,) if hname == "none" else (0.5, 0.75, 1.0)):
                    s = z(vlm) + wi * z(iv) - wh * z(hv)
                    rows.append((base.auc(s, gt), iname, wi, hname, wh, s))
    rows.sort(key=lambda r: -r[0])
    for a, iname, wi, hname, wh, _ in rows[:8]:
        print("  VLM + %.2f*%-10s - %.2f*%-9s   AUC %.4f"
              % (wi, iname, wh, hname, a))

    best_auc, bi, bwi, bh, bwh, bscore = rows[0]
    vlm_auc = base.auc(vlm, gt)
    imu_auc = base.auc(wrist_jerk, gt)
    print("\n=== BEST FUSION ===")
    print("  VLM + %.2f*%s - %.2f*%s" % (bwi, bi, bwh, bh))
    print("  fusion AUC %.4f" % best_auc)
    print("  VLM alone  %.4f   (%+.4f)" % (vlm_auc, best_auc - vlm_auc))
    print("  IMU alone  %.4f   (%+.4f)" % (imu_auc, best_auc - imu_auc))

    print("\n=== THRESHOLD SWEEP ON THE FUSION ===")
    always = base.prf(np.ones(n, bool), gt)
    print("  always-yes: P %.3f R %.3f F1 %.3f acc %.3f" % always)
    best_f1 = None
    for q in range(5, 100, 5):
        thr = np.percentile(bscore, q)
        p, r, f, ac = base.prf(bscore >= thr, gt)
        if best_f1 is None or f > best_f1[0]:
            best_f1 = (f, q, thr, p, r, ac)
        print("  q%2d  P %.3f R %.3f F1 %.3f acc %.3f" % (q, p, r, f, ac))
    print("  best F1 %.3f at q%d (acc %.3f)"
          % (best_f1[0], best_f1[1], best_f1[5]))

    # upper bound: a linear model fitted on this very clip
    print("\n=== UPPER BOUND (logistic fit on this clip - optimistic) ===")
    X = np.column_stack([z(vlm), z(wrist_jerk), z(wrist_gyro), z(head_jerk)])
    X = np.column_stack([X, np.ones(n)])
    y = gt.astype(np.float64)
    w = np.zeros(X.shape[1])
    for _ in range(300):
        p = 1.0 / (1.0 + np.exp(-X @ w))
        g = X.T @ (p - y) / n
        h = (X * (p * (1 - p))[:, None]).T @ X / n + 1e-4 * np.eye(X.shape[1])
        try:
            w -= np.linalg.solve(h, g)
        except np.linalg.LinAlgError:
            break
    fit = X @ w
    print("  fitted AUC %.4f  (weights VLM %+.2f jerk %+.2f gyro %+.2f "
          "head %+.2f)" % (base.auc(fit, gt), w[0], w[1], w[2], w[3]))
    print("  this is what a linear combination could reach if the weights")
    print("  were tuned on the answers - the honest number is the one above")

    json.dump({
        "clip": "GAS_TESTING 47cb14298f7e371d",
        "gt_work_pct": round(100 * float(gt.mean()), 1),
        "individual": {k: round(base.auc(v, gt), 4)
                       for k, v in singles.items()},
        "correlation_vlm_imu": {
            "wrist_jerk": round(float(np.corrcoef(vlm, wrist_jerk)[0, 1]), 4),
            "wrist_gyro": round(float(np.corrcoef(vlm, wrist_gyro)[0, 1]), 4),
            "head_jerk": round(float(np.corrcoef(vlm, head_jerk)[0, 1]), 4)},
        "always_yes_f1": round(always[2], 3),
        "best_fusion": {"imu_feature": bi, "imu_weight": bwi,
                        "head_feature": bh, "head_weight": bwh,
                        "auc": round(best_auc, 4),
                        "f1": round(best_f1[0], 3),
                        "percentile": best_f1[1],
                        "accuracy": round(best_f1[5], 3)},
        "upper_bound_fitted_auc": round(base.auc(fit, gt), 4),
    }, open(OUT, "w"), indent=2)
    np.save("/home/user/VLM_WORKSPACE/gas_fusion_score.npy", bscore)
    print("\nsaved:", OUT)

    print("\n=== VERDICT ===")
    gain = best_auc - max(vlm_auc, imu_auc)
    if gain > 0.05 and best_f1[0] > always[2] + 0.02:
        print("  Fusion gains %+.3f AUC over the better single channel and" % gain)
        print("  beats the always-yes baseline. Worth building on.")
    elif gain > 0.05:
        print("  Fusion gains %+.3f AUC, but F1 %.3f still does not beat the"
              % (gain, best_f1[0]))
        print("  always-yes baseline %.3f on a clip this dense with work."
              % always[2])
    else:
        print("  Fusion gains only %+.3f AUC - the two channels are not" % gain)
        print("  complementary enough on this clip to rescue each other.")


if __name__ == "__main__":
    main()
