"""IMU-gated action segmentation on the actseg sample MCAP.

Same architecture that scored AUC 0.819 on the CRM clip - motion decides when,
the VLM decides what - carried over to a recording from different hardware.

Two things had to be re-established rather than assumed:

  * the camera roles. The IMU said the head unit moves least while the other
    two swing to 25 m/s^2, and the contact sheets confirmed it: CPA9B520080
    looks down at the bench from head height, the other two sit at the wrists
    with fingers filling the bottom of the frame.

  * the CDR field offsets. They shift with the length of frame_id, so this
    recording puts acceleration at byte 260 where the ZED files put it at 244,
    and Orbbec splits gyro and accel across separate topics.

There is no ground truth for this clip, so nothing here reports F1 - the output
is the segmentation itself plus the motion curve it came from.
"""
import json
import sys
import time

import numpy as np
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

NPZ = "/home/user/VLM_WORKSPACE/actseg_imu.npz"
FRAMES = "/home/user/VLM_WORKSPACE/actseg_frames/CPA9B520080.npz"
OUT = "/home/user/VLM_WORKSPACE/actseg_segmentation.json"
REPO = "embedl/Cosmos-Reason2-2B-W4A16"

HEAD = "camera_CPA9B520080"
WRISTS = ["camera_CPA9B52005Z", "camera_CPAW752006B"]

HZ = 10
SMOOTH_S = 8.0
HEAD_WEIGHT = 0.75
PERCENTILE = 70
MIN_SPAN = 3.0
MERGE_GAP = 2.0
NFRAMES = 16

VERBS = ["unpacking", "assembling", "disassembling", "tightening", "loosening",
         "wiping", "cutting", "placing", "carrying", "adjusting", "inspecting",
         "lifting", "opening", "closing", "connecting", "pulling"]

PROMPT_OBJ = (
    "These %d frames cover one continuous %.0f-second clip, in order, filmed "
    "from a camera on the worker's head.\n"
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


def resample(t, v, n):
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


def jerk_feature(t, acc, n):
    mag = np.linalg.norm(acc, axis=1)
    dt = np.diff(t, prepend=t[0])
    dt[dt <= 0] = 1e-3
    j = np.abs(np.diff(mag, prepend=mag[0])) / dt
    return resample(t, np.minimum(j, 500.0), n)


def gyro_feature(t, gyro, n):
    return resample(t, np.linalg.norm(gyro, axis=1), n)


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
    merged = []
    for s in spans_from_mask(mask, dur):
        if merged and s["start"] - merged[-1]["end"] <= MERGE_GAP:
            merged[-1]["end"] = s["end"]
        else:
            merged.append(dict(s))
    return [s for s in merged if s["end"] - s["start"] >= MIN_SPAN]


def frames_between(ts, imgs, a, b, n):
    want = np.linspace(a, max(a, b - 1e-3), n)
    idx = np.clip(np.searchsorted(ts, want), 0, len(ts) - 1)
    return [imgs[i] for i in idx]


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
    d = np.load(NPZ)
    fr = np.load(FRAMES)
    ts, imgs = fr["t"], fr["img"]
    dur = float(ts[-1])
    n = int(dur * HZ)
    print("clip %.1f s, %d cached frames" % (dur, len(ts)), flush=True)

    t = time.time()
    head_jerk = smooth(jerk_feature(d[HEAD + "_imu_accel_t"],
                                    d[HEAD + "_imu_accel_acc"], n), SMOOTH_S)
    wrist_gyro = None
    for w in WRISTS:
        g = smooth(gyro_feature(d[w + "_imu_gyro_t"],
                                d[w + "_imu_gyro_gyro"], n), SMOOTH_S)
        wrist_gyro = g if wrist_gyro is None else np.maximum(wrist_gyro, g)
    score = z(wrist_gyro) - HEAD_WEIGHT * z(head_jerk)
    imu_s = time.time() - t
    print("motion score in %.2f s  (range %.2f .. %.2f)"
          % (imu_s, score.min(), score.max()), flush=True)

    thr = float(np.percentile(score, PERCENTILE))
    segs = postprocess(score >= thr, dur)
    pred = np.zeros(n, bool)
    for s in segs:
        pred[int(s["start"] * HZ):min(int(s["end"] * HZ), n)] = True
    print("gate: %d spans, work %.1f%% of the clip"
          % (len(segs), 100 * pred.mean()), flush=True)

    print("\nloading VLM...", flush=True)
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

    t = time.time()
    print("\n=== LABELLING %d SPANS ===" % len(segs), flush=True)
    for s in segs:
        dd = s["end"] - s["start"]
        f = frames_between(ts, imgs, s["start"], s["end"], NFRAMES)
        obj = clean(gen(model, proc, f, PROMPT_OBJ % (NFRAMES, dd), dd), 2)
        ranked = rank_verbs(model, proc, f,
                            PROMPT_VERB % (NFRAMES, dd, obj, verb_list),
                            dd, verb_ids)
        vb, pv = ranked[0]
        s.update(object=obj, verb=vb, conf=round(float(pv), 3),
                 label=("%s %s" % (vb, obj)).strip(),
                 top3=[(v, round(float(q), 3)) for v, q in ranked[:3]])
        print("  [%6.1f-%6.1f] %5.1fs  %-30s (%.2f)"
              % (s["start"], s["end"], dd, s["label"], pv), flush=True)
    label_s = time.time() - t

    del model
    torch.cuda.empty_cache()
    total = time.time() - t_all

    json.dump({
        "source_mcap": "/home/user/actseg/sample.mcap",
        "ego_camera": HEAD, "wrist_cameras": WRISTS,
        "config": {"gate": "max wrist gyro (8s) - %.2f * head jerk (8s)"
                           % HEAD_WEIGHT,
                   "smooth_s": SMOOTH_S, "percentile": PERCENTILE,
                   "threshold": round(thr, 4), "min_span_s": MIN_SPAN,
                   "merge_gap_s": MERGE_GAP, "nframes": NFRAMES,
                   "model": REPO, "verbs": VERBS},
        "ground_truth": None,
        "duration_s": round(dur, 1),
        "work_pct": round(100 * float(pred.mean()), 1),
        "segments": segs,
        "score": [round(float(x), 4) for x in score],
        "timing": {"total_s": round(total, 1), "imu_score_s": round(imu_s, 2),
                   "model_load_s": round(load_s, 1),
                   "labelling_s": round(label_s, 1),
                   "video_dur_s": round(dur, 1),
                   "realtime_factor": round(total / dur, 3)},
    }, open(OUT, "w"), indent=2)

    print("\n=== RESULT ===")
    print("  %d work spans over %.0f s (%.1f%% work)"
          % (len(segs), dur, 100 * pred.mean()))
    print("  total %.1f s (%.2fx realtime) - imu %.2fs, load %.0fs, labels %.0fs"
          % (total, total / dur, imu_s, load_s, label_s))
    print("  no ground truth for this clip, so no F1 is reported")
    print("\nsaved:", OUT)


if __name__ == "__main__":
    main()
