"""Cosmos-Reason2-2B-W4A16: stricter prompts on the video path.

The video path finds every work window (FN 0, recall 100%) but calls 18-20 of
32 idle windows work too. The verdict is not blind - it is trigger-happy. So
the wording has to raise the bar for "yes", not describe work differently.

Three variants, all on the same 37 windows, same 16-frame video input with
fps=2, same greedy decode. A is the current wording as the control.

  A  current      the F1 0.600 image-path wording, unchanged
  B  majority     work must fill MOST of the window, not appear in it
  C  advancing    the task must visibly progress - a state change between the
                  first and last frames, not just hands in motion

B and C attack the two things that plausibly cause the false positives:
brief incidental contact, and motion without a task advancing.
"""
import json, re, time, gc
import cv2, numpy as np, torch
from transformers import AutoModelForImageTextToText, AutoProcessor

MP4 = "/home/user/VLM_WORKSPACE/SAMPLE_MCAP_FROM_CRM/5e19608c34b6a880_ego.mp4"
GT = "/home/user/VLM_WORKSPACE/SAMPLE_MCAP_FROM_CRM/5e19608c34b6a880_metadata.json"
REPO = "embedl/Cosmos-Reason2-2B-W4A16"

WINDOW = 8.0
NFRAMES = 16
PX = 256
FPS = NFRAMES / WINDOW
OUT_JSON = "/home/user/VLM_WORKSPACE/cosmos_v9.json"

PROMPTS = {
    "A_current": (
        "These 16 frames are 0.5 seconds apart, in order. Is the camera wearer "
        "actively working with their hands on an object during this period "
        "(wiping, scrubbing, placing, fitting, cutting, operating)? Answer no if "
        "they are only walking, carrying, standing, waiting, looking around, or "
        "their hands are empty or out of view. Answer yes or no."
    ),
    "B_majority": (
        "This is an 8 second clip, in order. Judge the whole clip, not a moment "
        "in it.\n"
        "Answer yes only if the camera wearer is working with their hands on an "
        "object for MOST of these 8 seconds - wiping, scrubbing, placing, "
        "fitting, cutting, operating.\n"
        "Answer no if the working part is brief, if they are mainly walking, "
        "carrying, standing, waiting, reaching or looking around, or if their "
        "hands are empty or out of view for most of the clip.\n"
        "When unsure, answer no. Answer yes or no."
    ),
    "C_advancing": (
        "This is an 8 second clip, in order. Compare the first frames with the "
        "last frames.\n"
        "Answer yes only if a task visibly ADVANCED during the clip - an object "
        "ends up cleaner, moved into place, assembled, cut, or otherwise changed "
        "by the wearer's hands.\n"
        "Answer no if nothing about the objects changed: hands moving without "
        "changing anything, walking, carrying, holding, reaching, searching, "
        "waiting, or hands empty or out of view.\n"
        "Moving your hands is not work. Changing something is work.\n"
        "When unsure, answer no. Answer yes or no."
    ),
}
ORDER = ["A_current", "B_majority", "C_advancing"]


def load_gt():
    txt = open(GT).read()
    segs = re.findall(r'title:\s*([^,\n]+),\s*\n\s*description:\s*(.*?),\s*\n'
                      r'\s*start_ts:\s*(\d+),\s*\n\s*end_ts:\s*(\d+)', txt)
    base = int(re.search(r'start_ts:\s*(\d+)', txt).group(1))
    return [{"s": (int(a) - base) / 1000, "e": (int(b) - base) / 1000,
             "work": t.strip() != "idle"} for t, d, a, b in segs]


GTS = load_gt()


def gt_work_fraction(start, end):
    w = 0.0
    for g in GTS:
        if g["work"]:
            w += max(0.0, min(end, g["e"]) - max(start, g["s"]))
    return w / (end - start) if end > start else 0.0


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


YES = re.compile(r"^\s*\W*(yes|yeah|yep|true)\b", re.I)


def ask_video(model, proc, frames_rgb, prompt):
    vid = np.stack(frames_rgb)
    msgs = [{"role": "user", "content": [{"type": "video", "video": vid},
                                         {"type": "text", "text": prompt}]}]
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
    ntok = inp["input_ids"].shape[1]
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(**inp, max_new_tokens=8, do_sample=False)
    ms = (time.time() - t0) * 1000
    ans = proc.batch_decode(out[:, ntok:], skip_special_tokens=True)[0].strip()
    del inp, out
    torch.cuda.empty_cache()
    return ans, ntok, ms


def score(rows):
    tp = sum(1 for r in rows if r["gt_work"] and r["pred_work"])
    tn = sum(1 for r in rows if not r["gt_work"] and not r["pred_work"])
    fp = sum(1 for r in rows if not r["gt_work"] and r["pred_work"])
    fn = sum(1 for r in rows if r["gt_work"] and not r["pred_work"])
    n = len(rows) or 1
    pr = tp / (tp + fp) if tp + fp else 0.0
    rc = tp / (tp + fn) if tp + fn else 0.0
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn,
            "acc": round(100 * (tp + tn) / n, 1),
            "prec": round(100 * pr, 1), "rec": round(100 * rc, 1),
            "f1": round(2 * pr * rc / (pr + rc), 3) if pr + rc else 0.0,
            "pred_work_pct": round(100 * sum(r["pred_work"] for r in rows) / n, 1),
            "gt_work_pct": round(100 * sum(r["gt_work"] for r in rows) / n, 1)}


def main():
    proc = AutoProcessor.from_pretrained(REPO)
    model = AutoModelForImageTextToText.from_pretrained(
        REPO, dtype=torch.float16, low_cpu_mem_usage=True, device_map="cuda",
        attn_implementation="eager").eval()

    cap = cv2.VideoCapture(MP4)
    dur = cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)
    starts = [s for s in np.arange(0, dur, WINDOW) if s + WINDOW <= dur + 1e-6]
    print("video %.0f s | %d frames @%.1f fps | window %.0f s | %d window | %d prompt"
          % (dur, NFRAMES, FPS, WINDOW, len(starts), len(ORDER)), flush=True)

    # decode every window once; all three prompts see identical frames
    cache = {}
    for st in starts:
        fr = window_frames(cap, st)
        if fr:
            cache[st] = fr
    cap.release()
    print("frames cached: %d window" % len(cache), flush=True)

    results = []
    for name in ORDER:
        prompt = PROMPTS[name]
        print("\n### %s" % name, flush=True)
        rows, lat = [], []
        for st in starts:
            fr = cache.get(st)
            if not fr:
                continue
            try:
                ans, ntok, ms = ask_video(model, proc, fr, prompt)
            except Exception as e:
                print("    ERROR: %s" % str(e)[:90], flush=True)
                gc.collect(); torch.cuda.empty_cache()
                break
            lat.append(ms)
            pred = bool(YES.match(ans))
            frac = gt_work_fraction(st, st + WINDOW)
            g = bool(frac > 0.5)
            rows.append({"start": float(st), "end": float(st + WINDOW),
                         "gt_work": g, "gt_work_frac": round(float(frac), 3),
                         "pred_work": pred, "label": "", "raw": ans[:40],
                         "tokens": int(ntok)})
            if g or pred:
                print("    [%3.0f-%3.0f] GT %-4s -> %-4s  %s"
                      % (st, st + WINDOW, "WORK" if g else "idle",
                         "WORK" if pred else "IDLE",
                         "OK " if g == pred else "MISS"), flush=True)
        if not rows:
            continue
        m = score(rows)
        results.append({"config": name, "frames": NFRAMES, "px": PX,
                        "window_s": WINDOW, "fps": FPS, "prompt": prompt,
                        "input_mode": "video+metadata",
                        "median_ms": round(float(np.median(lat)), 1),
                        "median_tokens": int(np.median([r["tokens"] for r in rows])),
                        "metrics": m, "rows": rows})
        print("  -> F1 %.3f | acc %.0f%% | TP %d TN %d FP %d FN %d | work %.1f%%"
              % (m["f1"], m["acc"], m["tp"], m["tn"], m["fp"], m["fn"],
                 m["pred_work_pct"]), flush=True)
        json.dump(results, open(OUT_JSON, "w"), indent=2)
        gc.collect(); torch.cuda.empty_cache()

    print("\n=== SUMMARY ===")
    print("%-12s %6s %5s %4s %4s %4s %4s %7s"
          % ("prompt", "F1", "acc", "TP", "TN", "FP", "FN", "work%"))
    for r in results:
        m = r["metrics"]
        print("%-12s %6.3f %4.0f%% %4d %4d %4d %4d %6.1f%%"
              % (r["config"], m["f1"], m["acc"], m["tp"], m["tn"], m["fp"],
                 m["fn"], m["pred_work_pct"]))
    print("\nGT: work 13.5%% (5/37)")
    print("baseline (16 images, not video): F1 0.600 | TP 3 TN 30 FP 2 FN 2")
    print("saved:", OUT_JSON)


if __name__ == "__main__":
    main()
