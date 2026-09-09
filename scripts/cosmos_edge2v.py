"""Edge2 vs base W4A16 on the video path, frozen protocol.

The image path (1347 tokens) no longer fits: today it OOM-ed on window 1 for
Edge2 and on the 4 s sweep for the base model, even though the same code ran
end to end yesterday. The video path costs 470 tokens for the same 16 frames,
and has never OOM-ed in this session.

Frames are decoded once per window and handed to both checkpoints in turn, so
the only thing separating the two runs is the checkpoint itself. Frames are
not cached across all 37 windows up front - that is what starved Edge2 of
memory - each window is decoded, used by both models, then dropped.

FROZEN PROTOCOL:
  16 frames @2 fps, 256 px, video input with video_metadata fps=2,
  apply_chat_template, max_new_tokens=8, greedy, baseline prompt wording,
  8 s non-overlapping windows, >50% work rule.
"""
import json, re, time, gc
import cv2, numpy as np, torch
from transformers import AutoModelForImageTextToText, AutoProcessor

MP4 = "/home/user/VLM_WORKSPACE/SAMPLE_MCAP_FROM_CRM/5e19608c34b6a880_ego.mp4"
GT = "/home/user/VLM_WORKSPACE/SAMPLE_MCAP_FROM_CRM/5e19608c34b6a880_metadata.json"
OUT_JSON = "/home/user/VLM_WORKSPACE/cosmos_edge2v.json"

REPOS = [("edge2", "embedl/Cosmos-Reason2-2B-W4A16-Edge2"),
         ("base", "embedl/Cosmos-Reason2-2B-W4A16")]

WINDOW = 8.0
NFRAMES = 16
PX = 256
FPS = NFRAMES / WINDOW
MAX_NEW = 8

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


def ask(model, proc, frames_rgb):
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
    ntok = inp["input_ids"].shape[1]
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(**inp, max_new_tokens=MAX_NEW, do_sample=False)
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
            "gt_work_pct": round(100 * sum(r["gt_work"] for r in rows) / n, 1),
            "n_windows": len(rows)}


def main():
    cap = cv2.VideoCapture(MP4)
    dur = cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)
    starts = [s for s in np.arange(0, dur, WINDOW) if s + WINDOW <= dur + 1e-6]
    cap.release()
    print("FROZEN PROTOCOL (video path) | %d frames @%.1f fps, %d px, max_new=%d"
          % (NFRAMES, FPS, PX, MAX_NEW), flush=True)
    print("%d window x %.0f s | models: %s\n"
          % (len(starts), WINDOW, [t for t, _ in REPOS]), flush=True)

    results = []
    for tag, repo in REPOS:
        print("### %s  (%s)" % (tag, repo), flush=True)
        try:
            proc = AutoProcessor.from_pretrained(repo)
            t0 = time.time()
            model = AutoModelForImageTextToText.from_pretrained(
                repo, dtype=torch.float16, low_cpu_mem_usage=True,
                device_map="cuda", attn_implementation="eager").eval()
            print("    loaded %.0f s" % (time.time() - t0), flush=True)
        except Exception as e:
            print("    LOAD ERROR: %s" % str(e)[:110], flush=True)
            gc.collect(); torch.cuda.empty_cache()
            continue

        cap = cv2.VideoCapture(MP4)
        rows, lat = [], []
        died = False
        for st in starts:
            fr = window_frames(cap, st)
            if not fr:
                continue
            try:
                ans, ntok, ms = ask(model, proc, fr)
            except Exception as e:
                print("    ERROR [%3.0f]: %s" % (st, str(e)[:80]), flush=True)
                died = True
                break
            lat.append(ms)
            pred = bool(YES.match(ans))
            frac = gt_work_fraction(st, st + WINDOW)
            g = bool(frac > 0.5)
            rows.append({"start": float(st), "end": float(st + WINDOW),
                         "gt_work": g, "gt_work_frac": round(float(frac), 3),
                         "pred_work": pred, "label": "", "raw": ans[:30],
                         "tokens": int(ntok)})
            if g or pred:
                print("    [%3.0f-%3.0f] GT %-4s -> %-4s  %s"
                      % (st, st + WINDOW, "WORK" if g else "idle",
                         "WORK" if pred else "IDLE",
                         "OK " if g == pred else "MISS"), flush=True)
        cap.release()
        del model, proc
        gc.collect()
        torch.cuda.empty_cache()

        if not rows:
            continue
        m = score(rows)
        results.append({"config": tag, "repo": repo, "window_s": WINDOW,
                        "frames": NFRAMES, "px": PX, "fps": FPS,
                        "max_new_tokens": MAX_NEW, "input_mode": "video+metadata",
                        "prompt": PROMPT, "completed": not died,
                        "median_ms": round(float(np.median(lat)), 1),
                        "median_tokens": int(np.median([r["tokens"] for r in rows])),
                        "metrics": m, "rows": rows})
        print("  -> F1 %.3f | acc %.0f%% | TP %d TN %d FP %d FN %d | %d/%d window | %d tok | %.0f ms\n"
              % (m["f1"], m["acc"], m["tp"], m["tn"], m["fp"], m["fn"],
                 m["n_windows"], len(starts), results[-1]["median_tokens"],
                 np.median(lat)), flush=True)
        json.dump(results, open(OUT_JSON, "w"), indent=2)

    print("=== COMPARISON (same protocol, same windows) ===")
    print("%-8s %7s %6s %5s %5s %5s %5s %8s %8s"
          % ("model", "F1", "acc", "TP", "TN", "FP", "FN", "work%", "ms"))
    for r in results:
        m = r["metrics"]
        print("%-8s %7.3f %5.0f%% %5d %5d %5d %5d %7.1f%% %8.0f"
              % (r["config"], m["f1"], m["acc"], m["tp"], m["tn"], m["fp"],
                 m["fn"], m["pred_work_pct"], r["median_ms"]))
    if len(results) == 2 and all(r["completed"] for r in results):
        a = [r["pred_work"] for r in results[0]["rows"]]
        b = [r["pred_work"] for r in results[1]["rows"]]
        n = min(len(a), len(b))
        print("\nsame verdict: %d/%d window" % (sum(1 for i in range(n) if a[i] == b[i]), n))
    print("GT: work 13.5%")
    print("saved:", OUT_JSON)


if __name__ == "__main__":
    main()
