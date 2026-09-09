"""Frozen benchmark protocol + window-length curve.

Two things at once, because they depend on each other:

  1. Reproduce the F1 0.600 baseline exactly. Its script (cosmos_bench.py)
     turns out to use the same apply_chat_template path and the same
     max_new_tokens=8 as every later run, so the "different code path" theory
     is dead. Run A below is that configuration, bit for bit, to find out
     whether 0.600 was real or a one-off.

  2. Sweep window length 4/6/8/10/12 s under one frozen protocol, so the
     numbers are comparable to each other and to run A.

FROZEN PROTOCOL - identical for every run here:
  model      embedl/Cosmos-Reason2-2B-W4A16, fp16, device_map=cuda, eager attn
  input      16 images (not video), 256 px long edge, INTER_AREA
  call       apply_chat_template(..., tokenize=True, return_dict=True)
  decode     max_new_tokens=8, do_sample=False
  prompt     the baseline wording, with only the frame spacing substituted
  parse      ^\\W*(yes|yeah|yep|true)\\b
  windows    non-overlapping from t=0, a window counts as work when ground
             truth work covers more than half of it

Window length changes the number of windows (37 at 8 s, 75 at 4 s), so F1 is
computed per run against that run's own window set - the only honest way to
compare across lengths.
"""
import json, re, time, gc
import cv2, numpy as np, torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

MP4 = "/home/user/VLM_WORKSPACE/SAMPLE_MCAP_FROM_CRM/5e19608c34b6a880_ego.mp4"
GT = "/home/user/VLM_WORKSPACE/SAMPLE_MCAP_FROM_CRM/5e19608c34b6a880_metadata.json"
REPO = "embedl/Cosmos-Reason2-2B-W4A16"
OUT_JSON = "/home/user/VLM_WORKSPACE/cosmos_freeze.json"

NFRAMES = 16          # frozen
PX = 256              # frozen
MAX_NEW = 8           # frozen
# 8 s first: it is the reproduction check against F1 0.600.
WINDOWS = [8.0, 4.0, 6.0, 10.0, 12.0]


def build_prompt(window):
    """Baseline wording. Only the spacing number changes with window length."""
    gap = window / NFRAMES
    return (
        "These %d frames are %.2f seconds apart, in order. Is the camera wearer "
        "actively working with their hands on an object during this period "
        "(wiping, scrubbing, placing, fitting, cutting, operating)? Answer no if "
        "they are only walking, carrying, standing, waiting, looking around, or "
        "their hands are empty or out of view. Answer yes or no."
    ) % (NFRAMES, gap)


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


def window_frames(cap, start, window):
    gap = window / NFRAMES
    out = []
    for k in range(NFRAMES):
        cap.set(cv2.CAP_PROP_POS_MSEC, (start + k * gap) * 1000)
        ok, f = cap.read()
        if not ok:
            continue
        h, w = f.shape[:2]
        sc = PX / max(h, w)
        f = cv2.resize(f, (int(w * sc), int(h * sc)), interpolation=cv2.INTER_AREA)
        out.append(Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)))
    return out


YES = re.compile(r"^\s*\W*(yes|yeah|yep|true)\b", re.I)


def ask(model, proc, pil, prompt):
    """The frozen call. Images, chat template, greedy, 8 new tokens."""
    content = [{"type": "image", "image": im} for im in pil]
    msgs = [{"role": "user", "content": content + [{"type": "text", "text": prompt}]}]
    inp = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True,
                                   return_dict=True, return_tensors="pt").to("cuda")
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
    proc = AutoProcessor.from_pretrained(REPO)
    model = AutoModelForImageTextToText.from_pretrained(
        REPO, dtype=torch.float16, low_cpu_mem_usage=True, device_map="cuda",
        attn_implementation="eager").eval()

    cap = cv2.VideoCapture(MP4)
    dur = cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)
    print("FROZEN PROTOCOL: %d rasm, %d px, max_new=%d, chat_template, greedy"
          % (NFRAMES, PX, MAX_NEW), flush=True)
    print("video %.0f s | window uzunliklari: %s\n" % (dur, WINDOWS), flush=True)

    results = []
    for win in WINDOWS:
        starts = [s for s in np.arange(0, dur, win) if s + win <= dur + 1e-6]
        prompt = build_prompt(win)
        tag = "%.0fs" % win
        print("### %s window  (%d window, frames oraligi %.2f s)"
              % (tag, len(starts), win / NFRAMES), flush=True)
        rows, lat = [], []
        died = False
        for st in starts:
            pil = window_frames(cap, st, win)
            if not pil:
                continue
            try:
                ans, ntok, ms = ask(model, proc, pil, prompt)
            except Exception as e:
                print("    ERROR: %s" % str(e)[:90], flush=True)
                died = True
                gc.collect(); torch.cuda.empty_cache()
                break
            lat.append(ms)
            pred = bool(YES.match(ans))
            frac = gt_work_fraction(st, st + win)
            g = bool(frac > 0.5)
            rows.append({"start": float(st), "end": float(st + win), "gt_work": g,
                         "gt_work_frac": round(float(frac), 3), "pred_work": pred,
                         "label": "", "raw": ans[:30], "tokens": int(ntok)})
            if g or pred:
                print("    [%3.0f-%3.0f] GT %-4s -> %-4s  %s"
                      % (st, st + win, "WORK" if g else "idle",
                         "WORK" if pred else "IDLE",
                         "OK " if g == pred else "MISS"), flush=True)
        if died or not rows:
            continue
        m = score(rows)
        results.append({"config": tag, "window_s": win, "frames": NFRAMES,
                        "px": PX, "max_new_tokens": MAX_NEW,
                        "input_mode": "images+chat_template", "prompt": prompt,
                        "median_ms": round(float(np.median(lat)), 1),
                        "median_tokens": int(np.median([r["tokens"] for r in rows])),
                        "metrics": m, "rows": rows})
        print("  -> F1 %.3f | acc %.0f%% | TP %d TN %d FP %d FN %d | %d window | work %.1f%% (GT %.1f%%)\n"
              % (m["f1"], m["acc"], m["tp"], m["tn"], m["fp"], m["fn"],
                 m["n_windows"], m["pred_work_pct"], m["gt_work_pct"]), flush=True)
        json.dump(results, open(OUT_JSON, "w"), indent=2)
        gc.collect(); torch.cuda.empty_cache()
    cap.release()

    print("=== WINDOW LENGTH CURVE ===")
    print("%-6s %8s %7s %6s %5s %5s %5s %5s %8s"
          % ("window", "windows", "F1", "acc", "TP", "TN", "FP", "FN", "work%"))
    for r in sorted(results, key=lambda x: x["window_s"]):
        m = r["metrics"]
        print("%-6s %8d %7.3f %5.0f%% %5d %5d %5d %5d %7.1f%%"
              % (r["config"], m["n_windows"], m["f1"], m["acc"], m["tp"],
                 m["tn"], m["fp"], m["fn"], m["pred_work_pct"]))
    r8 = next((r for r in results if r["window_s"] == 8.0), None)
    if r8:
        print("\n=== IS 0.600 REPRODUCIBLE? ===")
        print("  reported baseline : F1 0.600 | TP 3 TN 30 FP 2 FN 2")
        m = r8["metrics"]
        print("  current 8s           : F1 %.3f | TP %d TN %d FP %d FN %d"
              % (m["f1"], m["tp"], m["tn"], m["fp"], m["fn"]))
        print("  ->", "TAKRORLANDI" if abs(m["f1"] - 0.6) < 0.05 else "TAKRORLANMADI")
    print("\nsaved:", OUT_JSON)


if __name__ == "__main__":
    main()
