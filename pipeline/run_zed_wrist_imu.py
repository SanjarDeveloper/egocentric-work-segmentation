"""The IMU-gated pipeline: motion decides when, the VLM decides what.

The measurements that led here, all on the same tick-level footing:

  VLM work/idle           AUC 0.611   (no threshold beat always-yes)
  wrist jerk, 8 s         AUC 0.783
  wrist gyro - head jerk  AUC 0.819

So the gate is motion, not vision. The head term is subtracted because head
motion is an anti-signal on its own (AUC 0.39): when the worker is looking
around or walking, they are not working, and that is precisely the case a
camera cannot distinguish from bending over a task.

The VLM is kept for what it is good at - naming the object and the action -
and is asked once per span rather than once per window, which is also why this
runs in seconds rather than the 584 s the window sweep cost.
"""
import json
import sys
import time

import cv2
import numpy as np
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import common as base

NPZ = "/home/user/VLM_WORKSPACE/crm_imu.npz"
MP4 = ("/home/user/VLM_WORKSPACE/SAMPLE_MCAP_FROM_CRM/"
       "5e19608c34b6a880_ego.mp4")
META = ("/home/user/VLM_WORKSPACE/SAMPLE_MCAP_FROM_CRM/"
        "5e19608c34b6a880_metadata.json")
OUT = "/home/user/VLM_WORKSPACE/imu_segmentation.json"
REPO = "embedl/Cosmos-Reason2-2B-W4A16"

HZ = 10
SMOOTH_S = 8.0
HEAD_WEIGHT = 0.75
PERCENTILE = 70        # from the sweep; work is the top 30% of motion
MIN_SPAN = 3.0
MERGE_GAP = 2.0
NFRAMES = 16
PX = 256

VERBS = ["assembling", "disassembling", "tightening", "loosening", "wiping",
         "scrubbing", "cutting", "placing", "carrying", "adjusting",
         "inspecting", "lifting", "testing", "turning", "pressing",
         "connecting"]

PROMPT_OBJ = (
    "These %d frames cover one continuous %.0f-second clip, in order.\n"
    "What single object are the worker's hands touching or working on for most "
    "of the clip?\n"
    "Name only the object, one or two words, lowercase. No verb, no sentence."
)
PROMPT_VERB = (
    "These %d frames cover one continuous %.0f-second clip, in order.\n"
    "The worker's hands are working on: %s.\n"
    "Which one of these best describes what the hands are doing?\n"
    "%s\n"
    "Answer with exactly one word from the list."
)


def z(x):
    s = x.std()
    return (x - x.mean()) / s if s > 1e-9 else x * 0


def motion_score(npz, n, dur):
    d = np.load(npz)
    F = {k: base.features(d[k + "_t"], d[k + "_acc"], d[k + "_gyro"], n, dur)
         for k in ("imu_wrist_left", "imu_wrist_right", "imu_head")}
    sm = lambda v: base.smooth(v, SMOOTH_S)
    wrist = np.maximum(sm(F["imu_wrist_left"]["gyro_mag"]),
                       sm(F["imu_wrist_right"]["gyro_mag"]))
    head = sm(F["imu_head"]["jerk"])
    return z(wrist) - HEAD_WEIGHT * z(head)


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


def postprocess(mask, dur):
    sp = spans_from_mask(mask, dur)
    merged = []
    for s in sp:
        if merged and s["start"] - merged[-1]["end"] <= MERGE_GAP:
            merged[-1]["end"] = s["end"]
        else:
            merged.append(dict(s))
    return [s for s in merged if s["end"] - s["start"] >= MIN_SPAN]


def read_window(cap, a, b, n):
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    out = []
    for t in np.linspace(a, max(a, b - 1e-3), n):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(t * fps)))
        ok, img = cap.read()
        if not ok:
            if out:
                out.append(out[-1])
            continue
        h, w = img.shape[:2]
        sc = PX / max(h, w)
        img = cv2.resize(img, (int(w * sc), int(h * sc)),
                         interpolation=cv2.INTER_AREA)
        out.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    while len(out) < n and out:
        out.append(out[-1])
    return out


def build(proc, frames, prompt, dur):
    vid = np.stack(frames)
    msgs = [{"role": "user", "content": [{"type": "video", "video": vid},
                                         {"type": "text", "text": prompt}]}]
    try:
        return proc.apply_chat_template(
            msgs, add_generation_prompt=True, tokenize=True, return_dict=True,
            return_tensors="pt",
            video_metadata=[{"fps": len(frames) / max(dur, 1e-3),
                             "total_num_frames": len(vid), "duration": dur}])
    except TypeError:
        return proc.apply_chat_template(msgs, add_generation_prompt=True,
                                        tokenize=True, return_dict=True,
                                        return_tensors="pt")


def gen(model, proc, frames, prompt, dur, max_new=10):
    inp = build(proc, frames, prompt, dur).to("cuda")
    ntok = inp["input_ids"].shape[1]
    with torch.no_grad():
        o = model.generate(**inp, max_new_tokens=max_new, do_sample=False)
    txt = proc.batch_decode(o[:, ntok:], skip_special_tokens=True)[0]
    del inp, o
    torch.cuda.empty_cache()
    return txt


def rank_verbs(model, proc, frames, prompt, dur, verb_ids):
    inp = build(proc, frames, prompt, dur).to("cuda")
    with torch.no_grad():
        lg = model(**inp).logits[0, -1].float()
    sc = {v: max(float(lg[i]) for i in ids) for v, ids in verb_ids.items()}
    del inp, lg
    torch.cuda.empty_cache()
    zz = np.array(list(sc.values()))
    zz = np.exp(zz - zz.max())
    zz /= zz.sum()
    return sorted(zip(sc.keys(), zz), key=lambda x: -x[1])


def first_ids(tok, w):
    ids = set()
    for f in (w, " " + w, w.capitalize(), " " + w.capitalize()):
        i = tok.encode(f, add_special_tokens=False)
        if i:
            ids.add(i[0])
    return sorted(ids)


def clean(t, nw=3):
    import re
    t = t.strip().split("\n")[0].strip().lower()
    t = re.sub(r'^(the |a |an )', '', t)
    t = re.sub(r'^(answer|action|object|label)\s*:?\s*', '', t)
    t = re.sub(r'[^a-z0-9 \-]', ' ', t)
    return " ".join(re.sub(r'\s+', ' ', t).strip().split()[:nw])


def main():
    t_all = time.time()
    gt_segs = base.load_gt(META)
    dur = max(s["end"] for s in gt_segs)
    n = int(dur * HZ)
    gt = np.zeros(n, bool)
    for s in gt_segs:
        if s["work"]:
            gt[int(s["start"] * HZ):min(int(s["end"] * HZ), n)] = True

    print("clip %.0f s, GT work %.1f%%" % (dur, 100 * gt.mean()), flush=True)

    t = time.time()
    score = motion_score(NPZ, n, dur)
    imu_s = time.time() - t
    print("motion score computed in %.2f s  (AUC %.4f)"
          % (imu_s, base.auc(score, gt)), flush=True)

    thr = float(np.percentile(score, PERCENTILE))
    segs = postprocess(score >= thr, dur)
    pred = np.zeros(n, bool)
    for s in segs:
        pred[int(s["start"] * HZ):min(int(s["end"] * HZ), n)] = True
    p, r, f1, acc = base.prf(pred, gt)
    print("gate: %d spans, work %.1f%%   P %.3f R %.3f F1 %.3f acc %.3f"
          % (len(segs), 100 * pred.mean(), p, r, f1, acc), flush=True)

    print("\nloading VLM for labelling...", flush=True)
    t = time.time()
    proc = AutoProcessor.from_pretrained(REPO)
    model = AutoModelForImageTextToText.from_pretrained(
        REPO, dtype=torch.float16, low_cpu_mem_usage=True, device_map="cuda",
        attn_implementation="eager").eval()
    load_s = time.time() - t
    tok = proc.tokenizer
    verb_ids = {v: first_ids(tok, v) for v in VERBS}
    verb_list = "\n".join(VERBS)
    print("model ready in %.0f s" % load_s, flush=True)

    cap = cv2.VideoCapture(MP4)
    t = time.time()
    print("\n=== LABELLING %d SPANS ===" % len(segs), flush=True)
    for s in segs:
        d = s["end"] - s["start"]
        fr = read_window(cap, s["start"], s["end"], NFRAMES)
        obj = clean(gen(model, proc, fr, PROMPT_OBJ % (NFRAMES, d), d), 2)
        ranked = rank_verbs(model, proc, fr,
                            PROMPT_VERB % (NFRAMES, d, obj, verb_list),
                            d, verb_ids)
        vb, pv = ranked[0]
        s.update(object=obj, verb=vb, conf=round(float(pv), 3),
                 label=("%s %s" % (vb, obj)).strip(),
                 top3=[(v, round(float(q), 3)) for v, q in ranked[:3]])
        ov = [g["title"] for g in gt_segs if g["work"]
              and min(g["end"], s["end"]) - max(g["start"], s["start"]) > 0]
        s["gt_overlap"] = ov[:3]
        print("  [%6.1f-%6.1f] %-28s (%.2f)  GT: %s"
              % (s["start"], s["end"], s["label"], pv, "; ".join(ov[:2])),
              flush=True)
    label_s = time.time() - t
    cap.release()
    del model
    torch.cuda.empty_cache()

    total = time.time() - t_all
    out = {
        "source_mp4": MP4, "imu_npz": NPZ, "metadata": META,
        "config": {"gate": "wrist gyro (max of both) - %.2f * head jerk"
                           % HEAD_WEIGHT,
                   "smooth_s": SMOOTH_S, "percentile": PERCENTILE,
                   "threshold": round(thr, 4), "min_span_s": MIN_SPAN,
                   "merge_gap_s": MERGE_GAP, "nframes": NFRAMES, "px": PX,
                   "model": REPO, "verbs": VERBS},
        "gt": gt_segs, "gt_work_pct": round(100 * float(gt.mean()), 1),
        "auc": {"fused_imu": round(base.auc(score, gt), 4),
                "vlm_reference_other_clip": 0.6108},
        "result": {"precision": round(p, 3), "recall": round(r, 3),
                   "f1": round(f1, 3), "accuracy": round(acc, 3),
                   "work_pct": round(100 * float(pred.mean()), 1)},
        "segments": segs,
        "score": [round(float(x), 4) for x in score],
        "timing": {"total_s": round(total, 1),
                   "imu_score_s": round(imu_s, 2),
                   "model_load_s": round(load_s, 1),
                   "labelling_s": round(label_s, 1),
                   "video_dur_s": round(dur, 1),
                   "realtime_factor": round(total / dur, 3)},
    }
    json.dump(out, open(OUT, "w"), indent=2)
    print("\n=== RESULT ===")
    print("  IMU gate AUC %.4f (VLM was 0.611 on the other clip)"
          % out["auc"]["fused_imu"])
    print("  P %.3f  R %.3f  F1 %.3f  acc %.3f" % (p, r, f1, acc))
    print("  total %.1f s for %.0f s of video (%.3f x realtime)"
          % (total, dur, total / dur))
    print("  breakdown: imu %.2f s, model load %.0f s, labelling %.0f s"
          % (imu_s, load_s, label_s))
    print("\nsaved:", OUT)


if __name__ == "__main__":
    main()
