"""Cosmos-Reason2-2B-W4A16, CRM chunk, work/idle at 0.5 fps.

One window = 16 frames sampled every 2 s = 32 s of video. The 5-minute chunk
therefore splits into 9 non-overlapping windows plus a short tail.

Ground truth comes from the CRM annotation; a window counts as work when work
covers more than half of it, which is the same rule the Encord projection uses.
"""
import json, re, sys, time, subprocess, gc
import cv2, numpy as np, torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

MP4 = "/home/user/VLM_WORKSPACE/SAMPLE_MCAP_FROM_CRM/5e19608c34b6a880_ego.mp4"
GT = "/home/user/VLM_WORKSPACE/SAMPLE_MCAP_FROM_CRM/5e19608c34b6a880_metadata.json"
REPO = "embedl/Cosmos-Reason2-2B-W4A16"

FPS = 2.0          # one frame every 0.5 s
NFRAMES = 16       # 16 x 0.5 s = 8 s per window
PX = 256
WINDOW = NFRAMES / FPS
ASK_LABEL = False   # off: the extra pass OOMs a 7.6 GB Orin NX

Q = ("These frames are 0.5 seconds apart, in order. Is the camera wearer actively "
     "working with their hands on an object during this period (wiping, scrubbing, "
     "placing, fitting, cutting, operating)? Answer no if they are only walking, "
     "carrying, standing, waiting, looking around, or their hands are empty or out "
     "of view. Answer yes or no.")

Q_WHAT = ("These frames are 0.5 seconds apart, in order. What is the camera wearer "
          "doing? Answer in 3 words.")


def load_gt():
    txt = open(GT).read()
    segs = re.findall(r'title:\s*([^,\n]+),\s*\n\s*description:\s*(.*?),\s*\n'
                      r'\s*start_ts:\s*(\d+),\s*\n\s*end_ts:\s*(\d+)', txt)
    base = int(re.search(r'start_ts:\s*(\d+)', txt).group(1))
    return [{"s": (int(a) - base) / 1000, "e": (int(b) - base) / 1000,
             "work": t.strip() != "idle"} for t, d, a, b in segs]


GTS = load_gt()


def gt_work_fraction(start, end):
    """Seconds of annotated work inside [start, end), as a fraction."""
    work = 0.0
    for g in GTS:
        if g["work"]:
            work += max(0.0, min(end, g["e"]) - max(start, g["s"]))
    return work / (end - start) if end > start else 0.0


def window_frames(cap, start):
    out = []
    for k in range(NFRAMES):
        cap.set(cv2.CAP_PROP_POS_MSEC, (start + k / FPS) * 1000)
        ok, f = cap.read()
        if not ok:
            continue
        h, w = f.shape[:2]
        sc = PX / max(h, w)
        f = cv2.resize(f, (int(w * sc), int(h * sc)), interpolation=cv2.INTER_AREA)
        out.append(Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)))
    return out


def tegrastats_start():
    return subprocess.Popen(["tegrastats", "--interval", "1000"],
                            stdout=open("/tmp/cosmos_tegra.log", "w"),
                            stderr=subprocess.DEVNULL)


def tegrastats_summary():
    ram, gpu, cpu = [], [], []
    for ln in open("/tmp/cosmos_tegra.log"):
        m = re.search(r"RAM (\d+)/(\d+)MB", ln)
        if m:
            ram.append(int(m.group(1)))
        m = re.search(r"GR3D_FREQ (\d+)%", ln)
        if m:
            gpu.append(int(m.group(1)))
        m = re.search(r"CPU \[([^\]]+)\]", ln)
        if m:
            vals = [int(x.split("%")[0]) for x in m.group(1).split(",") if "%" in x]
            if vals:
                cpu.append(sum(vals) / len(vals))
    f = lambda v: {"mean": round(float(np.mean(v)), 1),
                   "max": round(float(np.max(v)), 1)} if v else None
    return {"ram_mb": f(ram), "gpu_pct": f(gpu), "cpu_pct": f(cpu)}


def yes(s):
    return bool(re.match(r"^\s*\W*(yes|yeah|yep|true)", s or "", re.I))


def main():
    proc = AutoProcessor.from_pretrained(REPO)
    # eager: JetPack torch 2.5.0a0 has no enable_gqa in SDPA.
    model = AutoModelForImageTextToText.from_pretrained(
        REPO, dtype=torch.float16, low_cpu_mem_usage=True, device_map="cuda",
        attn_implementation="eager").eval()

    cap = cv2.VideoCapture(MP4)
    dur = cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)
    starts = [s for s in np.arange(0, dur, WINDOW) if s + WINDOW <= dur + 1e-6]
    print("video %.0f s | window %.0f s (%d frames @ %.1f fps) | %d window"
          % (dur, WINDOW, NFRAMES, FPS, len(starts)), flush=True)

    teg = tegrastats_start()
    rows, lat = [], []
    t_all = time.time()
    for i, st in enumerate(starts):
        pil = window_frames(cap, st)
        if not pil:
            continue
        content = [{"type": "image", "image": im} for im in pil]
        msgs = [{"role": "user", "content": content + [{"type": "text", "text": Q}]}]
        inp = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True,
                                       return_dict=True, return_tensors="pt").to("cuda")
        ntok = inp["input_ids"].shape[1]
        t0 = time.time()
        with torch.no_grad():
            out = model.generate(**inp, max_new_tokens=8, do_sample=False)
        ans = proc.batch_decode(out[:, ntok:], skip_special_tokens=True)[0].strip()
        pred = yes(ans)

        what = ""
        if pred and ASK_LABEL:
            msgs2 = [{"role": "user",
                      "content": content + [{"type": "text", "text": Q_WHAT}]}]
            inp2 = proc.apply_chat_template(msgs2, add_generation_prompt=True,
                                            tokenize=True, return_dict=True,
                                            return_tensors="pt").to("cuda")
            with torch.no_grad():
                o2 = model.generate(**inp2, max_new_tokens=12, do_sample=False)
            what = proc.batch_decode(o2[:, inp2["input_ids"].shape[1]:],
                                     skip_special_tokens=True)[0].strip()
        lat.append((time.time() - t0) * 1000)

        del inp, out
        torch.cuda.empty_cache()

        frac = gt_work_fraction(st, st + WINDOW)
        g = bool(frac > 0.5)
        rows.append({"start": float(st), "end": float(st + WINDOW), "gt_work": g,
                     "gt_work_frac": round(float(frac), 3), "pred_work": bool(pred),
                     "raw": ans[:70], "label": what[:40], "tokens": int(ntok)})
        print("  [%3.0f-%3.0f] GT %-4s (%.0f%% work) -> %-3s | %-22s | %5d tok"
              % (st, st + WINDOW, "WORK" if g else "idle", 100 * frac,
                 "yes" if pred else "no", (what or "-")[:22], ntok), flush=True)
        json.dump(rows, open("/home/user/VLM_WORKSPACE/cosmos_rows_partial.json", "w"))
    cap.release()
    total = time.time() - t_all
    teg.terminate()
    del model
    gc.collect()
    torch.cuda.empty_cache()

    tp = sum(1 for r in rows if r["gt_work"] and r["pred_work"])
    tn = sum(1 for r in rows if not r["gt_work"] and not r["pred_work"])
    fp = sum(1 for r in rows if not r["gt_work"] and r["pred_work"])
    fn = sum(1 for r in rows if r["gt_work"] and not r["pred_work"])
    n = len(rows) or 1
    pr = tp / (tp + fp) if tp + fp else 0.0
    rc = tp / (tp + fn) if tp + fn else 0.0
    metrics = {"tp": tp, "tn": tn, "fp": fp, "fn": fn,
               "acc": round(100 * (tp + tn) / n, 1),
               "prec": round(100 * pr, 1), "rec": round(100 * rc, 1),
               "f1": round(2 * pr * rc / (pr + rc), 3) if pr + rc else 0.0,
               "pred_work_pct": round(100 * sum(r["pred_work"] for r in rows) / n, 1),
               "gt_work_pct": round(100 * sum(r["gt_work"] for r in rows) / n, 1)}

    res = {"model": REPO, "fps": FPS, "frames_per_window": NFRAMES, "px": PX,
           "window_s": WINDOW, "windows": n,
           "median_ms": round(float(np.median(lat)), 1),
           "total_s": round(total, 1),
           "realtime_factor": round(dur / total, 2),
           "resources": tegrastats_summary(), "metrics": metrics, "rows": rows}
    json.dump(res, open("/home/user/VLM_WORKSPACE/cosmos_bench.json", "w"), indent=2)

    print("\n=== RESULT ===")
    print(json.dumps(metrics, indent=2))
    print("median %.0f ms/window | total %.0f s | %.2fx realtime"
          % (np.median(lat), total, dur / total))
    print("resurslar:", json.dumps(res["resources"]))
    print("saved: /home/user/VLM_WORKSPACE/cosmos_bench.json")


if __name__ == "__main__":
    main()
