"""Action segmentation on the GAS_TESTING clip, scored against real ground truth.

This is the first recording in the project that ships its own labels, so for
once the numbers mean something: metadata.json carries 33 segments over the
300 s clip, and the video is exactly 300.0 s at 30 fps, so subtracting the
clip's start_ts turns label time into video time directly.

Two passes over the clip:

  1. verdict - work or idle, 8 s windows every 2 s (4x overlap), scored by
     reading the yes/no logits rather than taking argmax, so the answer is a
     probability and the threshold stays tunable afterwards. Overlapping
     windows matter here: GT work spans run 3-14 s and non-overlapping windows
     with a majority rule silently drop the short ones.

  2. labels - for every work span the verdict produced, ask for the object,
     then pick the verb from a closed list scored by first-token logit. Open
     verb generation inverted the action on the previous clip ("disconnecting"
     for an assembly); a closed set makes that particular error impossible.

Timing and system load are sampled throughout so the run doubles as the
resource measurement for a 5-minute clip.
"""
import json, os, re, subprocess, sys, threading, time
import cv2, numpy as np, torch
from transformers import AutoModelForImageTextToText, AutoProcessor

BASE = "/media/user/EGOX-040-8F52E/MCAP_ORBBEC_SAMPLE/SAMPLE_GAS_TESTING"
MP4 = BASE + "/47cb14298f7e371d_ego.mp4"
META = BASE + "/47cb14298f7e371d_metadata.json"
OUT = "/home/user/VLM_WORKSPACE/gas_segmentation.json"
REPO = "embedl/Cosmos-Reason2-2B-W4A16"

WINDOW = 8.0
STRIDE = 2.0
NFRAMES = 16
PX = 256
THRESHOLD = 0.55          # swept afterwards; this is only the reported default
MIN_SPAN = 2.0            # drop work spans shorter than this
MERGE_GAP = 1.5           # bridge idle gaps shorter than this
HZ = 10                   # tick resolution for metrics

VERBS = ["testing", "adjusting", "inspecting", "placing", "opening", "closing",
         "turning", "pressing", "picking", "carrying", "connecting", "wiping"]

PROMPT_WORK = (
    "These %d frames cover one continuous %.0f-second clip, in order, filmed "
    "from a camera on the worker's head.\n"
    "Are the worker's hands actively performing a task on an object right now "
    "- touching, holding, turning, pressing, moving or carrying something?\n"
    "Walking, looking around, standing still or empty hands are not a task.\n"
    "Answer yes or no."
)
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


# ---------------------------------------------------------------- ground truth

def load_gt():
    """metadata.json is JSON-shaped but not JSON: keys and string values are
    unquoted, and the descriptions contain commas and apostrophes, so no
    regex-to-JSON trick survives them. Read it line by line instead - the
    layout is one `key: value` per line, which is unambiguous."""
    t0 = None
    chunk = None
    segs = []
    cur = {}
    for line in open(META):
        line = line.strip().rstrip(',')
        if not line or line in '{}[]':
            if line == '}' and cur.get("start_ts") is not None:
                segs.append(cur)
                cur = {}
            continue
        if line in ('segments: [', '],'):
            continue
        if ':' not in line:
            continue
        k, _, v = line.partition(':')
        k = k.strip()
        v = v.strip().rstrip(',').strip()
        if k == "chunk_id" and chunk is None:
            chunk = v
        elif k == "start_ts" and t0 is None and not cur:
            t0 = int(v)
        elif k in ("start_ts", "end_ts"):
            cur[k] = int(v)
        elif k in ("title", "description"):
            cur[k] = None if v == "null" else v
    if cur.get("start_ts") is not None:
        segs.append(cur)

    out = []
    for s in segs:
        if "start_ts" not in s or "end_ts" not in s:
            continue
        title = str(s.get("title") or "idle")
        out.append({"start": (s["start_ts"] - t0) / 1000.0,
                    "end": (s["end_ts"] - t0) / 1000.0,
                    "title": title,
                    "description": s.get("description"),
                    "work": title.strip().lower() != "idle"})
    out.sort(key=lambda s: s["start"])
    return {"chunk_id": chunk, "start_ts": t0}, out


# --------------------------------------------------------------------- frames

def read_window(cap, t_start, t_end, n):
    """Sample n frames between two timestamps from an already-open capture."""
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    want = np.linspace(t_start, max(t_start, t_end - 1e-3), n)
    out = []
    for t in want:
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


# ---------------------------------------------------------------------- model

def build(proc, frames_rgb, prompt, dur):
    vid = np.stack(frames_rgb)
    fps = len(frames_rgb) / max(dur, 1e-3)
    msgs = [{"role": "user", "content": [{"type": "video", "video": vid},
                                         {"type": "text", "text": prompt}]}]
    try:
        return proc.apply_chat_template(
            msgs, add_generation_prompt=True, tokenize=True, return_dict=True,
            return_tensors="pt",
            video_metadata=[{"fps": fps, "total_num_frames": len(vid),
                             "duration": dur}])
    except TypeError:
        return proc.apply_chat_template(msgs, add_generation_prompt=True,
                                        tokenize=True, return_dict=True,
                                        return_tensors="pt")


def p_yes(model, proc, frames, dur, yes_ids, no_ids):
    inp = build(proc, frames, PROMPT_WORK % (len(frames), dur), dur).to("cuda")
    with torch.no_grad():
        lg = model(**inp).logits[0, -1].float()
    ly = max(float(lg[i]) for i in yes_ids)
    ln = max(float(lg[i]) for i in no_ids)
    del inp, lg
    torch.cuda.empty_cache()
    m = max(ly, ln)
    return float(np.exp(ly - m) / (np.exp(ly - m) + np.exp(ln - m)))


def gen(model, proc, frames, prompt, dur, max_new=10):
    inp = build(proc, frames, prompt, dur).to("cuda")
    ntok = inp["input_ids"].shape[1]
    with torch.no_grad():
        out = model.generate(**inp, max_new_tokens=max_new, do_sample=False)
    txt = proc.batch_decode(out[:, ntok:], skip_special_tokens=True)[0]
    del inp, out
    torch.cuda.empty_cache()
    return txt


def rank_verbs(model, proc, frames, prompt, dur, verb_ids):
    inp = build(proc, frames, prompt, dur).to("cuda")
    with torch.no_grad():
        lg = model(**inp).logits[0, -1].float()
    sc = {v: max(float(lg[i]) for i in ids) for v, ids in verb_ids.items()}
    del inp, lg
    torch.cuda.empty_cache()
    z = np.array(list(sc.values()))
    z = np.exp(z - z.max())
    z /= z.sum()
    return sorted(zip(sc.keys(), z), key=lambda x: -x[1])


def first_ids(tok, word):
    ids = set()
    for form in (word, " " + word, word.capitalize(), " " + word.capitalize()):
        i = tok.encode(form, add_special_tokens=False)
        if i:
            ids.add(i[0])
    return sorted(ids)


def clean(t, nwords=3):
    t = t.strip().split("\n")[0].strip().lower()
    t = re.sub(r'^(the |a |an )', '', t)
    t = re.sub(r'^(answer|action|object|label)\s*:?\s*', '', t)
    t = re.sub(r'[^a-z0-9 \-]', ' ', t)
    return " ".join(re.sub(r'\s+', ' ', t).strip().split()[:nwords])


# -------------------------------------------------------------------- metrics

def to_ticks(spans, dur, key=None):
    n = int(dur * HZ)
    m = np.zeros(n, bool)
    for s in spans:
        if key is None or s[key]:
            m[int(s["start"] * HZ):min(int(s["end"] * HZ), n)] = True
    return m


def prf(pred, gt):
    tp = int((pred & gt).sum())
    fp = int((pred & ~gt).sum())
    fn = int((~pred & gt).sum())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    acc = float((pred == gt).mean())
    return {"precision": round(p, 3), "recall": round(r, 3), "f1": round(f, 3),
            "accuracy": round(acc, 3), "tp": tp, "fp": fp, "fn": fn}


def spans_from_mask(mask, dur):
    out = []
    n = len(mask)
    i = 0
    while i < n:
        if mask[i]:
            j = i
            while j < n and mask[j]:
                j += 1
            out.append({"start": round(i / HZ, 1), "end": round(min(j / HZ, dur), 1)})
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


# ------------------------------------------------------------------ resources

class Tegra(threading.Thread):
    """Sample tegrastats in the background for the resource report."""
    def __init__(self):
        super().__init__(daemon=True)
        self.rows = []
        self.stop = threading.Event()
        self.proc = None

    def run(self):
        try:
            self.proc = subprocess.Popen(
                ["tegrastats", "--interval", "1000"],
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
        except Exception:
            return
        for line in self.proc.stdout:
            if self.stop.is_set():
                break
            r = {}
            m = re.search(r'RAM (\d+)/(\d+)MB', line)
            if m:
                r["ram_mb"] = int(m.group(1)); r["ram_total_mb"] = int(m.group(2))
            m = re.search(r'GR3D_FREQ (\d+)%', line)
            if m:
                r["gpu_pct"] = int(m.group(1))
            cpus = re.search(r'CPU \[([^\]]+)\]', line)
            if cpus:
                vals = [int(x.split('%')[0]) for x in cpus.group(1).split(',')
                        if '%' in x]
                if vals:
                    r["cpu_pct_mean"] = round(sum(vals) / len(vals), 1)
            m = re.search(r'tj@([\d.]+)C', line)
            if m:
                r["tj_c"] = float(m.group(1))
            for tag in ("VDD_IN", "VDD_CPU_GPU_CV", "VDD_SOC"):
                m = re.search(tag + r' (\d+)mW', line)
                if m:
                    r[tag.lower() + "_mw"] = int(m.group(1))
            if r:
                self.rows.append(r)

    def finish(self):
        self.stop.set()
        if self.proc:
            try:
                self.proc.terminate()
            except Exception:
                pass

    def summary(self):
        if not self.rows:
            return {}
        out = {"samples": len(self.rows)}
        for k in ("ram_mb", "gpu_pct", "cpu_pct_mean", "tj_c", "vdd_in_mw"):
            v = [r[k] for r in self.rows if k in r]
            if v:
                out[k + "_mean"] = round(sum(v) / len(v), 1)
                out[k + "_max"] = max(v)
        if self.rows and "ram_total_mb" in self.rows[0]:
            out["ram_total_mb"] = self.rows[0]["ram_total_mb"]
        return out


# ------------------------------------------------------------------------ run

def main():
    meta, gt = load_gt()
    cap = cv2.VideoCapture(MP4)
    fps = cap.get(cv2.CAP_PROP_FPS)
    dur = cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps
    print("video %.1f s @ %.1f fps" % (dur, fps), flush=True)
    print("GT: %d segments, %d work" % (len(gt), sum(g["work"] for g in gt)),
          flush=True)
    gt_mask = to_ticks(gt, dur, "work")
    print("GT work fraction %.1f%%" % (100 * gt_mask.mean()), flush=True)

    teg = Tegra()
    teg.start()
    t_start = time.time()

    print("loading model...", flush=True)
    t_load = time.time()
    proc = AutoProcessor.from_pretrained(REPO)
    model = AutoModelForImageTextToText.from_pretrained(
        REPO, dtype=torch.float16, low_cpu_mem_usage=True, device_map="cuda",
        attn_implementation="eager").eval()
    load_s = time.time() - t_load
    print("model ready in %.0f s" % load_s, flush=True)

    tok = proc.tokenizer
    yes_ids = first_ids(tok, "yes") + first_ids(tok, "Yes")
    no_ids = first_ids(tok, "no") + first_ids(tok, "No")
    verb_ids = {v: first_ids(tok, v) for v in VERBS}
    verb_list = "\n".join(VERBS)

    # ---- pass 1: verdict
    starts = np.arange(0.0, max(dur - WINDOW, 0) + 1e-6, STRIDE)
    print("\n=== PASS 1: %d windows of %.0f s every %.0f s ==="
          % (len(starts), WINDOW, STRIDE), flush=True)
    wins, t_read, t_infer = [], 0.0, 0.0
    for k, s in enumerate(starts):
        e = min(s + WINDOW, dur)
        t = time.time(); fr = read_window(cap, s, e, NFRAMES); t_read += time.time() - t
        t = time.time(); p = p_yes(model, proc, fr, e - s, yes_ids, no_ids)
        t_infer += time.time() - t
        wins.append({"start": round(float(s), 1), "end": round(float(e), 1),
                     "p_yes": round(p, 4)})
        if k % 20 == 0 or k == len(starts) - 1:
            print("  %3d/%d  t=%6.1f  p_yes=%.3f" % (k + 1, len(starts), s, p),
                  flush=True)
    pass1_s = time.time() - t_start - load_s

    # ---- tick scores from overlapping windows
    n = int(dur * HZ)
    acc, cnt = np.zeros(n), np.zeros(n)
    for w in wins:
        a, b = int(w["start"] * HZ), min(int(w["end"] * HZ), n)
        acc[a:b] += w["p_yes"]; cnt[a:b] += 1
    score = np.where(cnt > 0, acc / np.maximum(cnt, 1), 0.0)

    # ---- threshold sweep against real GT
    print("\n=== THRESHOLD SWEEP (tick level) ===", flush=True)
    sweep = []
    for thr in np.arange(0.30, 0.86, 0.025):
        m = postprocess(score >= thr, dur)
        r = prf(to_ticks(m, dur), gt_mask)
        r["threshold"] = round(float(thr), 3)
        r["work_pct"] = round(100 * to_ticks(m, dur).mean(), 1)
        r["n_spans"] = len(m)
        sweep.append(r)
        print("  thr %.3f  P %.3f  R %.3f  F1 %.3f  acc %.3f  work %.1f%%  spans %d"
              % (thr, r["precision"], r["recall"], r["f1"], r["accuracy"],
                 r["work_pct"], r["n_spans"]), flush=True)
    best = max(sweep, key=lambda r: r["f1"])
    print("  best F1 %.3f at thr %.3f" % (best["f1"], best["threshold"]),
          flush=True)

    # separation: does p_yes actually distinguish work from idle?
    sep = {"mean_on_work": round(float(score[gt_mask].mean()), 4),
           "mean_on_idle": round(float(score[~gt_mask].mean()), 4)}
    sep["gap"] = round(sep["mean_on_work"] - sep["mean_on_idle"], 4)
    print("  separation: work %.4f vs idle %.4f (gap %.4f)"
          % (sep["mean_on_work"], sep["mean_on_idle"], sep["gap"]), flush=True)

    # ---- final mask at both the default and the best threshold
    spans_def = postprocess(score >= THRESHOLD, dur)
    spans_best = postprocess(score >= best["threshold"], dur)
    res_def = prf(to_ticks(spans_def, dur), gt_mask)

    # ---- pass 2: labels for the default-threshold spans
    print("\n=== PASS 2: labelling %d work spans ===" % len(spans_def), flush=True)
    t2 = time.time()
    for s in spans_def:
        d = s["end"] - s["start"]
        fr = read_window(cap, s["start"], s["end"], NFRAMES)
        obj = clean(gen(model, proc, fr, PROMPT_OBJ % (NFRAMES, d), d), 2)
        ranked = rank_verbs(model, proc, fr,
                            PROMPT_VERB % (NFRAMES, d, obj, verb_list), d, verb_ids)
        vb, p = ranked[0]
        s["object"] = obj
        s["verb"] = vb
        s["conf"] = round(float(p), 3)
        s["label"] = ("%s %s" % (vb, obj)).strip()
        s["top3"] = [(v, round(float(q), 3)) for v, q in ranked[:3]]
        ov = [g["title"] for g in gt if g["work"]
              and min(g["end"], s["end"]) - max(g["start"], s["start"]) > 0]
        s["gt_overlap"] = ov[:3]
        print("  [%6.1f-%6.1f] %-30s (%.2f)  GT: %s"
              % (s["start"], s["end"], s["label"], p, "; ".join(ov[:2])),
              flush=True)
    pass2_s = time.time() - t2
    cap.release()
    total_s = time.time() - t_start
    teg.finish()
    time.sleep(1.2)

    del model
    torch.cuda.empty_cache()

    timing = {
        "total_s": round(total_s, 1),
        "model_load_s": round(load_s, 1),
        "pass1_verdict_s": round(pass1_s, 1),
        "pass2_label_s": round(pass2_s, 1),
        "video_dur_s": round(dur, 1),
        "realtime_factor": round(total_s / dur, 2),
        "windows": len(wins),
        "frame_read_s_total": round(t_read, 1),
        "inference_s_total": round(t_infer, 1),
        "ms_per_window_read": round(1000 * t_read / max(len(wins), 1)),
        "ms_per_window_infer": round(1000 * t_infer / max(len(wins), 1)),
    }
    out = {
        "source_mp4": MP4, "metadata": META,
        "chunk_id": meta.get("chunk_id"),
        "config": {"window_s": WINDOW, "stride_s": STRIDE, "nframes": NFRAMES,
                   "px": PX, "threshold": THRESHOLD, "min_span_s": MIN_SPAN,
                   "merge_gap_s": MERGE_GAP, "model": REPO, "verbs": VERBS},
        "gt": gt, "gt_work_pct": round(100 * float(gt_mask.mean()), 1),
        "windows": wins,
        "sweep": sweep, "best": best, "separation": sep,
        "result_at_default_thr": res_def,
        "segments": spans_def,
        "spans_at_best_thr": spans_best,
        "timing": timing, "resources": teg.summary(),
    }
    json.dump(out, open(OUT, "w"), indent=2)

    print("\n=== RESULT (tick level, GT work %.1f%%) ===" % out["gt_work_pct"])
    print("  at thr %.2f : P %.3f  R %.3f  F1 %.3f  acc %.3f"
          % (THRESHOLD, res_def["precision"], res_def["recall"],
             res_def["f1"], res_def["accuracy"]))
    print("  best       : F1 %.3f at thr %.3f" % (best["f1"], best["threshold"]))
    print("\n=== TIMING ===")
    for k, v in timing.items():
        print("  %-24s %s" % (k, v))
    print("\n=== RESOURCES ===")
    for k, v in sorted(teg.summary().items()):
        print("  %-24s %s" % (k, v))
    print("\nsaved:", OUT)


if __name__ == "__main__":
    main()
