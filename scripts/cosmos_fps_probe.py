"""What happens if the model gets the native 30 fps instead of 2 fps?

The processor has do_sample_frames=True and its own fps=2, so it resamples
whatever it is handed. Feeding all 240 frames of an 8 s window at 30 fps is
therefore not obviously more information - the processor may throw most of it
away - but it changes which frames survive and how the timing is described.

Four rates over the same windows, same prompt, same 8 s span:

    2 fps  ->  16 frames   the current pipeline
    5 fps  ->  40 frames
   10 fps  ->  80 frames
   30 fps  -> 240 frames   native

Reports the visual grid the processor actually produced for each, so a rate
that changes nothing downstream is visible as such rather than assumed.
"""
import json, re, time, gc
import cv2, numpy as np, torch
from transformers import AutoModelForImageTextToText, AutoProcessor

MP4 = "/home/user/VLM_WORKSPACE/SAMPLE_MCAP_FROM_CRM/5e19608c34b6a880_ego.mp4"
GT = "/home/user/VLM_WORKSPACE/SAMPLE_MCAP_FROM_CRM/5e19608c34b6a880_metadata.json"
REPO = "embedl/Cosmos-Reason2-2B-W4A16"

WINDOW = 8.0
PX = 256
OUT_JSON = "/home/user/VLM_WORKSPACE/cosmos_fps.json"
RATES = [2.0, 5.0, 10.0, 30.0]
# a spread that includes every ground-truth work window plus idle ones
STARTS = [0.0, 24.0, 40.0, 56.0, 64.0, 120.0, 200.0, 272.0, 280.0]

# C_advancing: best precision of the three video prompts (FP 20 -> 5).
PROMPT = (
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
)


def load_gt():
    txt = open(GT).read()
    segs = re.findall(r'title:\s*([^,\n]+),\s*\n\s*description:\s*(.*?),\s*\n'
                      r'\s*start_ts:\s*(\d+),\s*\n\s*end_ts:\s*(\d+)', txt)
    base = int(re.search(r'start_ts:\s*(\d+)', txt).group(1))
    return [{"s": (int(a) - base) / 1000, "e": (int(b) - base) / 1000,
             "work": t.strip() != "idle"} for t, d, a, b in segs]


GTS = load_gt()


def gt_work(start, end):
    w = 0.0
    for g in GTS:
        if g["work"]:
            w += max(0.0, min(end, g["e"]) - max(start, g["s"]))
    return (w / (end - start)) > 0.5 if end > start else False


def window_frames(cap, start, fps):
    """Read the window at the given rate. 30 fps reads sequentially - seeking
    240 times would dominate the measurement."""
    n = int(round(WINDOW * fps))
    out = []
    if fps >= 25:
        cap.set(cv2.CAP_PROP_POS_MSEC, start * 1000)
        src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        want = int(round(WINDOW * src_fps))
        for _ in range(want):
            ok, f = cap.read()
            if not ok:
                break
            h, w = f.shape[:2]
            sc = PX / max(h, w)
            out.append(cv2.cvtColor(
                cv2.resize(f, (int(w * sc), int(h * sc)),
                           interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB))
        return out
    gap = WINDOW / n
    for k in range(n):
        cap.set(cv2.CAP_PROP_POS_MSEC, (start + k * gap) * 1000)
        ok, f = cap.read()
        if not ok:
            continue
        h, w = f.shape[:2]
        sc = PX / max(h, w)
        out.append(cv2.cvtColor(
            cv2.resize(f, (int(w * sc), int(h * sc)),
                       interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB))
    return out


YES = re.compile(r"^\s*\W*(yes|yeah|yep|true)\b", re.I)


def ask(model, proc, frames_rgb, fps):
    vid = np.stack(frames_rgb)
    msgs = [{"role": "user", "content": [{"type": "video", "video": vid},
                                         {"type": "text", "text": PROMPT}]}]
    inp = proc.apply_chat_template(
        msgs, add_generation_prompt=True, tokenize=True, return_dict=True,
        return_tensors="pt",
        video_metadata=[{"fps": fps, "total_num_frames": len(vid),
                         "duration": len(vid) / fps}])
    grid = inp["video_grid_thw"][0].tolist() if "video_grid_thw" in inp else None
    inp = inp.to("cuda")
    ntok = inp["input_ids"].shape[1]
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(**inp, max_new_tokens=8, do_sample=False)
    ms = (time.time() - t0) * 1000
    ans = proc.batch_decode(out[:, ntok:], skip_special_tokens=True)[0].strip()
    del inp, out
    torch.cuda.empty_cache()
    return ans, ntok, ms, grid


def main():
    proc = AutoProcessor.from_pretrained(REPO)
    model = AutoModelForImageTextToText.from_pretrained(
        REPO, dtype=torch.float16, low_cpu_mem_usage=True, device_map="cuda",
        attn_implementation="eager").eval()
    cap = cv2.VideoCapture(MP4)
    print("window %.0f s | %d points | prompt: C_advancing" % (WINDOW, len(STARTS)),
          flush=True)

    results = []
    for fps in RATES:
        nf = int(round(WINDOW * fps))
        print("\n### %.0f fps  (~%d frames / %.0f s window)" % (fps, nf, WINDOW),
              flush=True)
        rows, lat, read_ms = [], [], []
        died = False
        for st in STARTS:
            t0 = time.time()
            fr = window_frames(cap, st, fps)
            read_ms.append((time.time() - t0) * 1000)
            if not fr:
                continue
            try:
                ans, ntok, ms, grid = ask(model, proc, fr, fps)
            except Exception as e:
                print("    ERROR: %s" % str(e)[:90], flush=True)
                died = True
                gc.collect(); torch.cuda.empty_cache()
                break
            lat.append(ms)
            pred = bool(YES.match(ans))
            g = gt_work(st, st + WINDOW)
            rows.append({"t": st, "gt": g, "pred": pred, "tokens": int(ntok),
                         "ms": ms, "grid": grid, "n_in": len(fr)})
            print("    t=%3.0f GT %-4s -> %-4s | %d frames given | grid %s | %4d tok | %6.0f ms  %s"
                  % (st, "WORK" if g else "idle", "yes" if pred else "no",
                     len(fr), grid, ntok, ms, "OK " if g == pred else "MISS"),
                  flush=True)
        if died or not rows:
            continue
        ok = sum(1 for r in rows if r["gt"] == r["pred"])
        results.append({"fps": fps, "frames_given": nf, "rows": rows,
                        "gt_match": ok, "n": len(rows),
                        "median_tokens": int(np.median([r["tokens"] for r in rows])),
                        "median_ms": round(float(np.median(lat)), 1),
                        "median_read_ms": round(float(np.median(read_ms)), 1)})
        print("  -> GT match %d/%d | %d tok | %.0f ms infer | %.0f ms oqwork"
              % (ok, len(rows), results[-1]["median_tokens"],
                 np.median(lat), np.median(read_ms)), flush=True)
        json.dump(results, open(OUT_JSON, "w"), indent=2)
        gc.collect(); torch.cuda.empty_cache()
    cap.release()

    print("\n=== SUMMARY ===")
    print("%-6s %7s %8s %7s %9s %9s" %
          ("fps", "frames", "grid t", "token", "infer ms", "oqwork ms"))
    for r in results:
        gt_ = r["rows"][0]["grid"]
        print("%-6.0f %7d %8s %7d %9.0f %9.0f"
              % (r["fps"], r["frames_given"], gt_[0] if gt_ else "?",
                 r["median_tokens"], r["median_ms"], r["median_read_ms"]))
    print()
    for r in results:
        print("  %2.0f fps -> GT match %d/%d" % (r["fps"], r["gt_match"], r["n"]))
    print("\nsaved:", OUT_JSON)


if __name__ == "__main__":
    main()
