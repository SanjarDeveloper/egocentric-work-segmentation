"""Experiment 1: overlapping windows + logit-level yes/no, on the CRM chunk.

Two problems the GT audit exposed, attacked together.

PROBLEM A - window boundaries eat the work.
Ground-truth work segments are 3-12 s, nine of eleven shorter than the 8 s
window. A 5 s work segment straddling a window boundary lands as 25% + 37%
and both windows score idle under the >50% rule. That is why the window-level
metric says 13.5% work while the video is 20.7% work. Non-overlapping windows
cannot recover those segments no matter how good the model is.
  -> 8 s windows with 2 s stride: every second is covered by 4 windows, so a
     short segment always falls near the centre of at least one.

PROBLEM B - the model over-fires.
Precision 20-30%: it answers "yes" for carrying and walking. argmax over the
generated token throws away how close the call was.
  -> read the logits of the "yes" and "no" tokens directly at the first
     generated position. That gives a continuous score instead of a coin flip,
     so the operating point can be chosen after the fact rather than baked in
     by the prompt.

Both are measured at the SECOND level against the raw GT intervals, not at
window level - the window metric is itself part of what is being questioned.
"""
import json, re, time, gc
import cv2, numpy as np, torch
from transformers import AutoModelForImageTextToText, AutoProcessor

MP4 = "/home/user/VLM_WORKSPACE/SAMPLE_MCAP_FROM_CRM/5e19608c34b6a880_ego.mp4"
GT = "/home/user/VLM_WORKSPACE/SAMPLE_MCAP_FROM_CRM/5e19608c34b6a880_metadata.json"
REPO = "embedl/Cosmos-Reason2-2B-W4A16"
OUT_JSON = "/home/user/VLM_WORKSPACE/exp1_overlap.json"

WINDOW = 8.0
STRIDE = 2.0          # 4x coverage
NFRAMES = 16
PX = 256
FPS = NFRAMES / WINDOW

PROMPT = (
    "These %d frames are %.2f seconds apart, in order. Is the camera wearer "
    "actively working with their hands on an object during this period "
    "(wiping, scrubbing, placing, fitting, cutting, operating)? Answer no if "
    "they are only walking, carrying, standing, waiting, looking around, or "
    "their hands are empty or out of view. Answer yes or no."
) % (NFRAMES, WINDOW / NFRAMES)


def load_gt():
    txt = open(GT).read()
    segs = re.findall(r'title:\s*([^,\n]+),\s*\n\s*description:\s*(.*?),\s*\n'
                      r'\s*start_ts:\s*(\d+),\s*\n\s*end_ts:\s*(\d+)', txt)
    base = int(re.search(r'start_ts:\s*(\d+)', txt).group(1))
    return [{"s": (int(a) - base) / 1000, "e": (int(b) - base) / 1000,
             "work": t.strip() != "idle"} for t, d, a, b in segs]


GTS = load_gt()


def gt_second_mask(dur, hz=10):
    """Per-tick ground truth over the whole video. Ticks, not windows."""
    n = int(dur * hz)
    m = np.zeros(n, bool)
    for g in GTS:
        if g["work"]:
            m[int(g["s"] * hz):int(g["e"] * hz)] = True
    return m


def window_frames(cap, start):
    gap = WINDOW / NFRAMES
    out = []
    for k in range(NFRAMES):
        cap.set(cv2.CAP_PROP_POS_MSEC, (start + k * gap) * 1000)
        ok, f = cap.read()
        if not ok:
            continue
        h, w = f.shape[:2]
        sc = PX / max(h, w)
        f = cv2.resize(f, (int(w * sc), int(h * sc)), interpolation=cv2.INTER_AREA)
        out.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    return out


def yes_no_token_ids(proc):
    """Token ids for the first token of ' yes'/'no' style answers.

    Several spellings map to different ids depending on leading space and
    case, so collect them all and take the max logit over each group.
    """
    tok = proc.tokenizer
    ys, ns = set(), set()
    for s in ["yes", "Yes", " yes", " Yes", "YES"]:
        ids = tok.encode(s, add_special_tokens=False)
        if ids:
            ys.add(ids[0])
    for s in ["no", "No", " no", " No", "NO"]:
        ids = tok.encode(s, add_special_tokens=False)
        if ids:
            ns.add(ids[0])
    return sorted(ys), sorted(ns)


def ask_logits(model, proc, frames_rgb, yes_ids, no_ids):
    """Return (p_yes, argmax_pred, text) - the score and the old behaviour."""
    vid = np.stack(frames_rgb)
    msgs = [{"role": "user", "content": [{"type": "video", "video": vid},
                                         {"type": "text", "text": PROMPT}]}]
    try:
        inp = proc.apply_chat_template(
            msgs, add_generation_prompt=True, tokenize=True, return_dict=True,
            return_tensors="pt",
            video_metadata=[{"fps": FPS, "total_num_frames": len(vid),
                             "duration": len(vid) / FPS}])
    except TypeError:
        inp = proc.apply_chat_template(msgs, add_generation_prompt=True,
                                       tokenize=True, return_dict=True,
                                       return_tensors="pt")
    inp = inp.to("cuda")
    with torch.no_grad():
        out = model(**inp)
        logits = out.logits[0, -1].float()
    ly = max(float(logits[i]) for i in yes_ids)
    ln = max(float(logits[i]) for i in no_ids)
    # two-way softmax over just these two options
    p_yes = float(np.exp(ly) / (np.exp(ly) + np.exp(ln)))
    del inp, out, logits
    torch.cuda.empty_cache()
    return p_yes


def seg_metrics(pred_mask, gt_mask):
    tp = int(np.sum(pred_mask & gt_mask))
    tn = int(np.sum(~pred_mask & ~gt_mask))
    fp = int(np.sum(pred_mask & ~gt_mask))
    fn = int(np.sum(~pred_mask & gt_mask))
    pr = tp / (tp + fp) if tp + fp else 0.0
    rc = tp / (tp + fn) if tp + fn else 0.0
    iou = tp / (tp + fp + fn) if tp + fp + fn else 0.0
    n = len(gt_mask)
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn,
            "acc": round(100 * (tp + tn) / n, 1),
            "prec": round(100 * pr, 1), "rec": round(100 * rc, 1),
            "f1": round(2 * pr * rc / (pr + rc), 3) if pr + rc else 0.0,
            "iou": round(iou, 3),
            "pred_work_pct": round(100 * float(np.mean(pred_mask)), 1),
            "gt_work_pct": round(100 * float(np.mean(gt_mask)), 1)}


def main():
    proc = AutoProcessor.from_pretrained(REPO)
    model = AutoModelForImageTextToText.from_pretrained(
        REPO, dtype=torch.float16, low_cpu_mem_usage=True, device_map="cuda",
        attn_implementation="eager").eval()
    yes_ids, no_ids = yes_no_token_ids(proc)
    print("yes token ids:", yes_ids, "| no token ids:", no_ids, flush=True)

    cap = cv2.VideoCapture(MP4)
    dur = cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)
    starts = [float(s) for s in np.arange(0, dur - WINDOW + 1e-6, STRIDE)]
    print("video %.0f s | window %.0f s | stride %.0f s | %d window (overlap %.1fx)"
          % (dur, WINDOW, STRIDE, len(starts), WINDOW / STRIDE), flush=True)

    HZ = 10
    gt_mask = gt_second_mask(dur, HZ)
    scores, lat = [], []
    t0all = time.time()
    for i, st in enumerate(starts):
        fr = window_frames(cap, st)
        if not fr:
            scores.append((st, None))
            continue
        t0 = time.time()
        p = ask_logits(model, proc, fr, yes_ids, no_ids)
        lat.append((time.time() - t0) * 1000)
        scores.append((st, p))
        if i % 10 == 0 or p > 0.5:
            covers = np.mean(gt_mask[int(st * HZ):int((st + WINDOW) * HZ)])
            print("    [%3.0f-%3.0f] p_yes=%.3f | GT work fraction %.0f%%"
                  % (st, st + WINDOW, p, 100 * covers), flush=True)
        json.dump({"starts": [s for s, _ in scores],
                   "p_yes": [p for _, p in scores]}, open(OUT_JSON + ".partial", "w"))
    cap.release()
    total = time.time() - t0all
    del model
    gc.collect()
    torch.cuda.empty_cache()

    valid = [(s, p) for s, p in scores if p is not None]
    print("\n%d window scored | %.0f ms/window | total %.0f s"
          % (len(valid), np.median(lat), total), flush=True)

    # per-tick score: average p_yes of every window covering that tick
    n_ticks = len(gt_mask)
    acc = np.zeros(n_ticks)
    cnt = np.zeros(n_ticks)
    for s, p in valid:
        a, b = int(s * HZ), int((s + WINDOW) * HZ)
        acc[a:b] += p
        cnt[a:b] += 1
    tick_score = np.where(cnt > 0, acc / np.maximum(cnt, 1), 0.0)

    results = {"window_s": WINDOW, "stride_s": STRIDE, "frames": NFRAMES,
               "px": PX, "hz": HZ, "n_windows": len(valid),
               "median_ms": round(float(np.median(lat)), 1),
               "total_s": round(total, 1),
               "windows": [{"start": s, "p_yes": p} for s, p in valid],
               "sweeps": {}}

    print("\n=== THRESHOLD SWEEP (tick level, GT work %.1f%%) ==="
          % (100 * np.mean(gt_mask)))
    print("%-6s %7s %6s %6s %6s %6s %7s" %
          ("thr", "F1", "IoU", "prec", "rec", "acc", "work%"))
    best = None
    for thr in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]:
        pm = tick_score >= thr
        m = seg_metrics(pm, gt_mask)
        results["sweeps"]["thr_%.2f" % thr] = m
        print("%-6.2f %7.3f %6.3f %5.1f%% %5.1f%% %5.1f%% %6.1f%%"
              % (thr, m["f1"], m["iou"], m["prec"], m["rec"], m["acc"],
                 m["pred_work_pct"]))
        if best is None or m["f1"] > best[1]["f1"]:
            best = (thr, m)

    # what the old non-overlapping argmax protocol scores on the same ticks
    old = np.zeros(n_ticks, bool)
    for s, p in valid:
        if s % WINDOW == 0 and p >= 0.5:
            old[int(s * HZ):int((s + WINDOW) * HZ)] = True
    m_old = seg_metrics(old, gt_mask)
    results["baseline_nonoverlap_argmax"] = m_old

    print("\n=== COMPARISON (tick level) ===")
    print("  old: non-overlapping + argmax : F1 %.3f | IoU %.3f | prec %.1f%% | rec %.1f%%"
          % (m_old["f1"], m_old["iou"], m_old["prec"], m_old["rec"]))
    print("  new: %.1fx overlap + thr %.2f  : F1 %.3f | IoU %.3f | prec %.1f%% | rec %.1f%%"
          % (WINDOW / STRIDE, best[0], best[1]["f1"], best[1]["iou"],
             best[1]["prec"], best[1]["rec"]))
    results["best_threshold"] = best[0]
    json.dump(results, open(OUT_JSON, "w"), indent=2)
    print("\nsaved:", OUT_JSON)


if __name__ == "__main__":
    main()
