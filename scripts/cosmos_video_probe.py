"""Image-list input vs video input, same 16 frames, same prompt.

The pipeline currently hands the model 16 separate images. Qwen3-VL also has a
video path (Qwen3VLVideoProcessor) whose config sets temporal_patch_size: 2 -
it pairs adjacent frames into one temporal patch, so 16 frames should cost
roughly half the visual tokens. Fewer tokens on an eager-attention Jetson is
the one lever that actually moves latency here.

This measures both on the same windows: token count, latency, and whether the
verdict changes.
"""
import re, time, json
import cv2, numpy as np, torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

MP4 = "/home/user/VLM_WORKSPACE/SAMPLE_MCAP_FROM_CRM/5e19608c34b6a880_ego.mp4"
GT = "/home/user/VLM_WORKSPACE/SAMPLE_MCAP_FROM_CRM/5e19608c34b6a880_metadata.json"
REPO = "embedl/Cosmos-Reason2-2B-W4A16"

WINDOW = 8.0
NFRAMES = 16
PX = 256
# A spread of windows including the ones ground truth calls work.
STARTS = [0.0, 24.0, 40.0, 64.0, 120.0, 200.0, 272.0, 280.0]

PROMPT = (
    "These %d frames are %.1f seconds apart, in order. Is the camera wearer "
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


def gt_work(start, end):
    w = 0.0
    for g in GTS:
        if g["work"]:
            w += max(0.0, min(end, g["e"]) - max(start, g["s"]))
    return (w / (end - start)) > 0.5 if end > start else False


def frames(cap, start):
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


def run_once(model, proc, msgs):
    inp = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True,
                                   return_dict=True, return_tensors="pt").to("cuda")
    ntok = inp["input_ids"].shape[1]
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(**inp, max_new_tokens=8, do_sample=False)
    ms = (time.time() - t0) * 1000
    ans = proc.batch_decode(out[:, ntok:], skip_special_tokens=True)[0].strip()
    del inp, out
    torch.cuda.empty_cache()
    return ans, ntok, ms


def main():
    proc = AutoProcessor.from_pretrained(REPO)
    model = AutoModelForImageTextToText.from_pretrained(
        REPO, dtype=torch.float16, low_cpu_mem_usage=True, device_map="cuda",
        attn_implementation="eager").eval()
    cap = cv2.VideoCapture(MP4)

    print("%-6s %-5s | %-28s | %-28s" % ("t", "GT", "IMAGE (current)", "VIDEO"))
    print("-" * 78)
    rows = []
    for st in STARTS:
        fr = frames(cap, st)
        if not fr:
            continue
        g = gt_work(st, st + WINDOW)
        pil = [Image.fromarray(f) for f in fr]

        # current path: 16 separate images
        msgs_i = [{"role": "user",
                   "content": [{"type": "image", "image": im} for im in pil]
                              + [{"type": "text", "text": PROMPT}]}]
        ai, ti, mi = run_once(model, proc, msgs_i)

        # video path: one clip, temporal_patch_size pairs adjacent frames
        vid = np.stack(fr)                       # (T, H, W, C) uint8 RGB
        av, tv, mv = "-", 0, 0.0
        try:
            msgs_v = [{"role": "user",
                       "content": [{"type": "video", "video": vid},
                                   {"type": "text", "text": PROMPT}]}]
            av, tv, mv = run_once(model, proc, msgs_v)
        except Exception as e:
            av = "ERROR: " + str(e)[:40]

        pi, pv = bool(YES.match(ai)), bool(YES.match(av))
        print("%-6.0f %-5s | %-4s %5d tok %6.0f ms | %-4s %5d tok %6.0f ms  %s"
              % (st, "WORK" if g else "idle",
                 "yes" if pi else "no", ti, mi,
                 "yes" if pv else "no", tv, mv,
                 "" if pi == pv else "<- FARQ"), flush=True)
        rows.append({"t": st, "gt": g, "img_pred": pi, "img_tok": ti, "img_ms": mi,
                     "vid_pred": pv, "vid_tok": tv, "vid_ms": mv,
                     "img_raw": ai[:30], "vid_raw": str(av)[:30]})
    cap.release()

    ok = [r for r in rows if r["vid_tok"] > 0]
    if ok:
        it = np.median([r["img_tok"] for r in ok])
        vt = np.median([r["vid_tok"] for r in ok])
        im = np.median([r["img_ms"] for r in ok])
        vm = np.median([r["vid_ms"] for r in ok])
        agree = sum(1 for r in ok if r["img_pred"] == r["vid_pred"])
        print("\n=== SUMMARY ===")
        print("  token : image %4.0f -> video %4.0f  (%.2fx)" % (it, vt, vt / it))
        print("  vaqt  : image %4.0f -> video %4.0f ms (%.2fx)" % (im, vm, vm / im))
        print("  same verdict: %d/%d" % (agree, len(ok)))
        gi = sum(1 for r in ok if r["gt"] == r["img_pred"])
        gv = sum(1 for r in ok if r["gt"] == r["vid_pred"])
        print("  GT match: image %d/%d | video %d/%d" % (gi, len(ok), gv, len(ok)))
    json.dump(rows, open("/home/user/VLM_WORKSPACE/cosmos_video_probe.json", "w"),
              indent=2, default=str)
    print("saved: cosmos_video_probe.json")


if __name__ == "__main__":
    main()
