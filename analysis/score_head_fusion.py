"""Do the IMU and the VLM make independent mistakes?

If they do, combining them beats either alone; if the VLM's weak signal is just
a noisy copy of the motion signal, fusion adds nothing and the VLM can be
dropped from the gate entirely.

The awkward part is that the two scores were measured on different clips - the
VLM ran on GAS_TESTING, whose SSD unplugged itself mid-session, while the IMU
came from the CRM clip that lives on internal storage. So this cannot fuse the
two directly. What it can do is establish, on the CRM clip alone, how much of
the IMU signal the head IMU's inverse adds - head motion is the one channel
that behaved like an anti-signal (AUC 0.39-0.47), which is exactly the shape of
an independent, useful feature.

If head-inverse fusion lifts AUC materially, that is direct evidence the
"combine independent channels" argument holds, and the VLM is worth re-running
on this clip for a three-way fusion.
"""
import json
import numpy as np

NPZ = "/home/user/VLM_WORKSPACE/crm_imu.npz"
META = ("/home/user/VLM_WORKSPACE/SAMPLE_MCAP_FROM_CRM/"
        "5e19608c34b6a880_metadata.json")
HZ = 10

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import common as base


def z(x):
    s = x.std()
    return (x - x.mean()) / s if s > 1e-9 else x * 0


def main():
    d = np.load(NPZ)
    gt_segs = base.load_gt(META)
    dur = max(s["end"] for s in gt_segs)
    n = int(dur * HZ)
    gt = np.zeros(n, bool)
    for s in gt_segs:
        if s["work"]:
            gt[int(s["start"] * HZ):min(int(s["end"] * HZ), n)] = True

    F = {}
    for key in ("imu_wrist_left", "imu_wrist_right", "imu_head"):
        t, a, g = d[key + "_t"], d[key + "_acc"], d[key + "_gyro"]
        F[key] = base.features(t, a, g, n, dur)

    W = 8.0
    wl = {k: base.smooth(v, W) for k, v in F["imu_wrist_left"].items()}
    wr = {k: base.smooth(v, W) for k, v in F["imu_wrist_right"].items()}
    hd = {k: base.smooth(v, W) for k, v in F["imu_head"].items()}

    wrist_jerk = np.maximum(wl["jerk"], wr["jerk"])
    wrist_gyro = np.maximum(wl["gyro_mag"], wr["gyro_mag"])
    wrist_acc = np.maximum(wl["acc_std"], wr["acc_std"])

    print("=== SINGLE CHANNELS (8 s window) ===")
    singles = {
        "wrist jerk (max)": wrist_jerk,
        "wrist gyro (max)": wrist_gyro,
        "wrist acc_std (max)": wrist_acc,
        "head jerk": hd["jerk"],
        "head gyro": hd["gyro_mag"],
    }
    for k, v in singles.items():
        print("  %-22s AUC %.4f" % (k, base.auc(v, gt)))

    print("\n=== FUSION: wrist motion minus head motion ===")
    print("  (head motion is an anti-signal: AUC < 0.5 on its own)")
    rows = []
    for wname, wv in (("jerk", wrist_jerk), ("gyro", wrist_gyro),
                      ("acc_std", wrist_acc)):
        for hname, hv in (("head_jerk", hd["jerk"]),
                          ("head_gyro", hd["gyro_mag"]),
                          ("head_acc", hd["acc_std"])):
            for wt in (0.0, 0.25, 0.5, 0.75, 1.0):
                s = z(wv) - wt * z(hv)
                a = base.auc(s, gt)
                rows.append((a, wname, hname, wt, s))
    rows.sort(key=lambda r: -r[0])
    seen = set()
    for a, wn, hn, wt, _ in rows[:12]:
        if (wn, hn) in seen and wt not in (0.0,):
            continue
        seen.add((wn, hn))
        print("  wrist_%-8s - %.2f * %-10s   AUC %.4f" % (wn, wt, hn, a))

    best_auc, bwn, bhn, bwt, bscore = rows[0]
    print("\n=== BEST FUSION ===")
    print("  wrist_%s - %.2f * %s   AUC %.4f" % (bwn, bwt, bhn, best_auc))
    base_auc = base.auc(wrist_jerk, gt)
    print("  wrist alone            AUC %.4f" % base_auc)
    print("  gain from head channel      %+.4f" % (best_auc - base_auc))

    print("\n=== THRESHOLD SWEEP ON BEST FUSION ===")
    always = base.prf(np.ones(n, bool), gt)
    print("  always-yes: P %.3f R %.3f F1 %.3f acc %.3f" % always)
    best = None
    for q in range(5, 100, 5):
        thr = np.percentile(bscore, q)
        p, r, f, ac = base.prf(bscore >= thr, gt)
        if best is None or f > best[0]:
            best = (f, q, thr, p, r, ac)
    print("  best F1 %.3f at q%d  (P %.3f R %.3f acc %.3f)"
          % (best[0], best[1], best[3], best[4], best[5]))

    json.dump({
        "clip_dur_s": dur, "gt_work_pct": round(100 * float(gt.mean()), 1),
        "wrist_only_auc": round(base_auc, 4),
        "best_fusion": {"wrist_feature": bwn, "head_feature": bhn,
                        "head_weight": bwt, "auc": round(best_auc, 4),
                        "f1": round(best[0], 3), "percentile": best[1],
                        "precision": round(best[3], 3),
                        "recall": round(best[4], 3),
                        "accuracy": round(best[5], 3)},
    }, open("/home/user/VLM_WORKSPACE/fuse_check.json", "w"), indent=2)
    np.save("/home/user/VLM_WORKSPACE/fuse_best_score.npy", bscore)
    print("\nsaved: fuse_check.json, fuse_best_score.npy")


if __name__ == "__main__":
    main()
