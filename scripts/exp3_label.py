"""Experiment 3: verb + noun for each WORK span, over the whole span.

Two span sources, run in that order on purpose:

  GT spans      the 11 annotated work intervals. Clean input, so whatever the
                labels look like is purely the VLM's labelling ability - span
                detection cannot be blamed.
  p_yes spans   what exp1 actually produces at thr 0.60, merged and filtered.
                This is the real pipeline; comparing it against the GT-span
                labels shows how much span error costs.

Frames come from the WHOLE span, not a fixed window: a 12 s span is sampled
across all 12 s, a 3 s span across 3 s. That is the point of doing this at
span level rather than window level - the model sees the action from start to
finwork and can say what the dominant one was.

Measured earlier and applied here: asking for a label together with the
work/idle verdict wrecks the verdict (F1 0.600 -> 0.222). Here the verdict is
already settled, so the label costs nothing.

Output is verb + noun only, 2-3 words, no sentence.
"""
import json, re, time, gc
import cv2, numpy as np, torch
from transformers import AutoModelForImageTextToText, AutoProcessor

MP4 = "/home/user/VLM_WORKSPACE/SAMPLE_MCAP_FROM_CRM/5e19608c34b6a880_ego.mp4"
GT = "/home/user/VLM_WORKSPACE/SAMPLE_MCAP_FROM_CRM/5e19608c34b6a880_metadata.json"
EXP1 = "/home/user/VLM_WORKSPACE/exp1_overlap.json"
REPO = "embedl/Cosmos-Reason2-2B-W4A16"
OUT_JSON = "/home/user/VLM_WORKSPACE/exp3_label.json"

NFRAMES = 16
PX = 256
HZ = 10
THR = 0.60            # exp1's best operating point
MIN_SPAN = 2.0        # drop anything shorter - it cannot carry an action
MERGE_GAP = 2.0       # bridge sub-2 s dips inside one continuous action

LABEL_PROMPT = (
    "These %d frames cover one continuous %.0f-second clip, in order, and the "
    "camera wearer is working throughout it.\n"
    "Look at the whole clip and name the single dominant action.\n"
    "Answer with a verb and the object only, 2 or 3 words, lowercase, no "
    "sentence and no explanation.\n"
    "Examples of the format: wiping tray, cutting cardboard, rinsing cloth, "
    "placing inserts."
)


def load_gt():
    txt = open(GT).read()
    segs = re.findall(r'title:\s*([^,\n]+),\s*\n\s*description:\s*(.*?),\s*\n'
                      r'\s*start_ts:\s*(\d+),\s*\n\s*end_ts:\s*(\d+)', txt)
    base = int(re.search(r'start_ts:\s*(\d+)', txt).group(1))
    return [{"s": (int(a) - base) / 1000, "e": (int(b) - base) / 1000,
             "work": t.strip() != "idle", "title": t.strip(),
             "desc": d.strip()} for t, d, a, b in segs]


GTS = load_gt()


def gt_spans():
    return [{"start": g["s"], "end": g["e"], "gt_title": g["title"],
             "gt_desc": g["desc"]} for g in GTS if g["work"]]


def pyes_spans():
    """Thresholded p_yes -> merged intervals, the exp1 pipeline output."""
    d = json.load(open(EXP1))
    win = d["window_s"]
    wins = [(w["start"], w["p_yes"]) for w in d["windows"]]
    dur = max(s for s, _ in wins) + win
    n = int(dur * HZ)
    acc = np.zeros(n)
    cnt = np.zeros(n)
    for s, p in wins:
        a, b = int(s * HZ), int(min(s + win, dur) * HZ)
        acc[a:b] += p
        cnt[a:b] += 1
    tick = np.where(cnt > 0, acc / np.maximum(cnt, 1), 0.0)
    mask = tick >= THR

    spans, i = [], 0
    while i < n:
        if mask[i]:
            j = i
            while j < n and mask[j]:
                j += 1
            spans.append([i / HZ, j / HZ])
            i = j
        else:
            i += 1
    # bridge short dips, then drop short spans
    merged = []
    for sp in spans:
        if merged and sp[0] - merged[-1][1] <= MERGE_GAP:
            merged[-1][1] = sp[1]
        else:
            merged.append(sp)
    return [{"start": a, "end": b} for a, b in merged if b - a >= MIN_SPAN]


def span_frames(cap, start, end):
    """NFRAMES spread across the whole span, however long it is."""
    dur = end - start
    ts = np.linspace(start, max(start, end - 1e-3), NFRAMES)
    out = []
    for t in ts:
        cap.set(cv2.CAP_PROP_POS_MSEC, float(t) * 1000)
        ok, f = cap.read()
        if not ok:
            continue
        h, w = f.shape[:2]
        sc = PX / max(h, w)
        f = cv2.resize(f, (int(w * sc), int(h * sc)), interpolation=cv2.INTER_AREA)
        out.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    return out, dur


def clean(text):
    """Keep the first line, strip punctuation and filler, cap at 3 words."""
    t = text.strip().split("\n")[0].strip().lower()
    t = re.sub(r'^(the |a |an )', '', t)
    t = re.sub(r'^(answer|action|label)\s*:?\s*', '', t)
    t = re.sub(r'[^a-z0-9 \-]', ' ', t)
    t = re.sub(r'\s+', ' ', t).strip()
    words = t.split()
    return " ".join(words[:3])


def label_span(model, proc, frames_rgb, span_dur):
    vid = np.stack(frames_rgb)
    prompt = LABEL_PROMPT % (len(frames_rgb), span_dur)
    fps = len(frames_rgb) / max(span_dur, 1e-3)
    msgs = [{"role": "user", "content": [{"type": "video", "video": vid},
                                         {"type": "text", "text": prompt}]}]
    try:
        inp = proc.apply_chat_template(
            msgs, add_generation_prompt=True, tokenize=True, return_dict=True,
            return_tensors="pt",
            video_metadata=[{"fps": fps, "total_num_frames": len(vid),
                             "duration": span_dur}])
    except TypeError:
        inp = proc.apply_chat_template(msgs, add_generation_prompt=True,
                                       tokenize=True, return_dict=True,
                                       return_tensors="pt")
    inp = inp.to("cuda")
    ntok = inp["input_ids"].shape[1]
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(**inp, max_new_tokens=12, do_sample=False)
    ms = (time.time() - t0) * 1000
    raw = proc.batch_decode(out[:, ntok:], skip_special_tokens=True)[0]
    del inp, out
    torch.cuda.empty_cache()
    return clean(raw), raw.strip()[:60], int(ntok), ms


def run(model, proc, cap, tag, spans):
    print("\n### %s  (%d span)" % (tag, len(spans)), flush=True)
    rows, lat = [], []
    for sp in spans:
        fr, dur = span_frames(cap, sp["start"], sp["end"])
        if not fr:
            continue
        lab, raw, ntok, ms = label_span(model, proc, fr, dur)
        lat.append(ms)
        r = dict(sp)
        r.update({"dur": round(dur, 1), "label": lab, "raw": raw,
                  "tokens": ntok, "ms": round(ms)})
        rows.append(r)
        gtxt = sp.get("gt_title", "")
        print("    [%6.1f-%6.1f] %4.1fs -> %-22s %s"
              % (sp["start"], sp["end"], dur, lab,
                 ("| GT: " + gtxt[:30]) if gtxt else ""), flush=True)
    return rows, lat


def main():
    proc = AutoProcessor.from_pretrained(REPO)
    model = AutoModelForImageTextToText.from_pretrained(
        REPO, dtype=torch.float16, low_cpu_mem_usage=True, device_map="cuda",
        attn_implementation="eager").eval()
    cap = cv2.VideoCapture(MP4)

    g = gt_spans()
    p = pyes_spans()
    print("GT span: %d | p_yes span (thr %.2f, min %.0fs): %d"
          % (len(g), THR, MIN_SPAN, len(p)), flush=True)

    out = {"nframes": NFRAMES, "px": PX, "thr": THR, "min_span_s": MIN_SPAN,
           "merge_gap_s": MERGE_GAP, "prompt": LABEL_PROMPT}
    rows_g, lat_g = run(model, proc, cap, "GT spans (clean input)", g)
    out["gt_spans"] = rows_g
    json.dump(out, open(OUT_JSON, "w"), indent=2)

    rows_p, lat_p = run(model, proc, cap, "p_yes spans (full pipeline)", p)
    out["pyes_spans"] = rows_p
    cap.release()
    del model
    gc.collect()
    torch.cuda.empty_cache()

    print("\n=== VOCABULARY ===")
    for tag, rows in (("GT span", rows_g), ("p_yes span", rows_p)):
        vocab = {}
        for r in rows:
            vocab[r["label"]] = vocab.get(r["label"], 0) + 1
        print("  %s: %d span -> %d distinct labels" % (tag, len(rows), len(vocab)))
        for k, v in sorted(vocab.items(), key=lambda x: -x[1]):
            print("      %-26s x%d" % (k, v))

    print("\n=== SIDE BY SIDE WITH GT TITLE ===")
    for r in rows_g:
        print("  %-24s | GT: %s" % (r["label"], r.get("gt_title", "")[:34]))

    if lat_g or lat_p:
        allm = lat_g + lat_p
        print("\nmedian %.0f ms/span | %d spans total %.0f s"
              % (np.median(allm), len(allm), sum(allm) / 1000))
    json.dump(out, open(OUT_JSON, "w"), indent=2)
    print("saved:", OUT_JSON)


if __name__ == "__main__":
    main()
