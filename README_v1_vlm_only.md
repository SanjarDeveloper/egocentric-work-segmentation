# Egocentric work/idle segmentation on Jetson Orin NX

Temporal segmentation of egocentric (head-mounted, first-person) workshop
video into **work** and **idle** intervals, with a short `verb + noun` label
for each work interval — running entirely on an **NVIDIA Jetson Orin NX 8 GB**
with no training and no ground-truth labels required at inference time.

This repository is a record of what was measured, including what did not work.
Every number below came from a run on the device; nothing is estimated.

---

## Hardware and platform

| | |
|---|---|
| Device | ZED Box Orin NX (Jetson Orin NX 8 GB, sm_87) |
| Memory | 7.62 GB unified (CPU+GPU share one pool) |
| JetPack | 6.2.1 / L4T R36.4.4, CUDA 12.6.68, TensorRT 10.3.0 |
| Camera | Orbbec Gemini 2L, head-mounted, first-person view |
| Recording | 5-minute MCAP chunks, 1920×1200 @ 30 fps |

The 8 GB unified memory is the binding constraint throughout. Everything below
that failed, failed because of it.

---

## Final configuration

```
model         embedl/Cosmos-Reason2-2B-W4A16   (Qwen3-VL 2B, INT4 weights, FP16 compute)
input         video path, 16 frames @ 2 fps, 256 px long edge
window        8 s, 2 s stride  ->  4x coverage
decision      logits of the "yes"/"no" tokens, threshold 0.60 on the averaged score
labelling     second VLM pass, whole span, verb + noun only
runtime       ~3.0 s per window inference, ~470 tokens
memory        1.1 GB model, ~7.3 GB peak
```

### Pipeline

```
Orbbec Gemini 2L  ->  MCAP (5-min chunk)
                          |
                    MP4 (H.264)                 [MCAP direct read also works: 2316 ms/window vs 3079 ms]
                          |
        +-----------------+-----------------+
        |  8 s windows, 2 s stride           |
        |  16 frames @ 2 fps, 256 px         |
        +-----------------+-----------------+
                          |
              Cosmos-Reason2-2B-W4A16
              video input + video_metadata(fps=2)
              "is the wearer working? yes/no"
                          |
              read yes/no token logits  ->  p_yes in [0,1]
                          |
              average over the 4 windows covering each tick
                          |
              threshold 0.60  ->  work/idle mask
                          |
        +-----------------+-----------------+
        |                                   |
     IDLE spans                        WORK spans
     (no description,                       |
      by design)                  second VLM pass over the
                                  whole span, 16 frames
                                            |
                                  "name the dominant action,
                                   verb + noun, 2-3 words"
                                            |
                                  "wiping tray", "placing inserts"
```

Two design decisions came directly from measurement:

**The verdict and the label are separate passes.** Asking for both in one
query destroys the verdict — measured four times across four different
wordings (F1 0.600 → 0.000–0.333). Once the verdict is settled, the label
costs nothing.

**Idle spans get no description.** Asking the model to describe idle time
produces confident hallucinations ("pick up a box, place it in a carton" over
a clip of someone walking).

---

## Results

Ground truth: one 5-minute CRM chunk, 11 annotated work segments, 62 s of work
(20.7% of the video). Work segments are short — nine of eleven are under 8
seconds, which matters for the metric (see *Metric pitfall* below).

### Work/idle detection

Scored per 0.1 s tick against the raw GT intervals.

| Configuration | F1 | IoU | Precision | Recall |
|---|---|---|---|---|
| **8 s / 2 s stride, logit threshold 0.60** | **0.434** | **0.277** | 29.2% | 84.0% |
| 8 s non-overlapping, argmax | 0.405 | 0.254 | 26.5% | 85.6% |

Threshold sweep (this is the useful part):

| threshold | F1 | precision | recall | predicted work |
|---|---|---|---|---|
| 0.30 | 0.366 | 22.6% | 96.9% | 88.7% |
| 0.50 | 0.408 | 26.4% | 88.9% | 69.4% |
| **0.60** | **0.434** | 29.2% | 84.0% | 59.4% |
| 0.70 | 0.317 | 32.8% | 30.7% | 19.3% |
| 0.80 | 0.000 | 0.0% | 0.0% | 0.0% |

The model never scores above ~0.75. Everything lives in 0.41–0.75, and recall
falls off a cliff between 0.60 and 0.70. **This is a calibration problem, not
a decision problem** — the two classes are barely separated. Two example
frames make it concrete: a clip of genuine wiping scores 0.703, a clip of the
worker merely holding a tray scores 0.640. A 0.063 gap.

### Action labelling

Given clean spans (the 11 GT intervals), the second pass produces usable
labels:

| VLM output | Ground truth | |
|---|---|---|
| cleaning tray | Wiping the baking tray | correct |
| placing papers in | Placing inserts in tray | correct |
| cleaning metal tray | Wiping the baking tray | correct |
| placing papers in | Placing inserts in tray | correct |
| cleaning tray | Wiping the baking tray | correct |
| placing papers in | Placing inserts in tray | correct |
| unpacking plastic bags | Wiping the baking tray | wrong |
| unpacking boxes | Placing cardboard strips in tray | close |
| holding and arranging | Taking foam pieces from colleague | close |
| filling bucket | Rinsing and wringing cloth | close |
| opening oven door | Cleaning oven front panel | close |

Six of eleven are accurate. The four repeated wiping segments all received the
same label, and the three insert-placing segments all received the same label
— the model is consistent on recurring tasks, which is what a taxonomy needs.

Given the spans the detector actually produces, labelling degrades badly,
because those spans are wrong (see *Open problem*).

---

## What was measured and rejected

### Models that do not fit 7.6 GB

| Model | Weights | Outcome |
|---|---|---|
| VideoChat3-4B | 8.94 GB | larger than total memory |
| InternVideo2.5-Chat-8B | 16.80 GB | no |
| InternVideo3-8B | 18.73 GB | no |
| Gemma-4-E2B-it | 10.25 GB | no (smallest Gemma 4) |
| VideoChat-Flash-2B | 4.14 GB | loads on CPU at 4.69 GB, OOM on `.cuda()` |
| moondream3-9B | 10.5 GB fp8 | fp8 needs sm_89+, Orin is sm_87 |
| Marlin-2B | — | GPTQ build needs `gptqmodel`, which needs `pcre`, not on PyPI |
| Cosmos-Reason2-8B-W4A16-FlashHead | 7.30 GB | fits, but FlashHead is a vLLM plugin and vLLM has no aarch64 build |

### Models that fit but perform worse

| Model | F1 (window level) | Note |
|---|---|---|
| **Cosmos-Reason2-2B-W4A16** | **0.333** | the chosen one |
| Cosmos-Reason2-2B-W4A16-Edge2 | 0.278 | 26 false positives vs 20; 4 layers kept at FP16 did not help |
| FastVLM-0.5B | — | says "work" 70% of the time (GT 20%); echoes long prompts verbatim |
| SmolVLM2-500M-Video | — | says "work" 96% of the time; ~870 tokens per frame, OOM above 2 frames |
| X3D-M + moondream2 | 0.216 | earlier pipeline |

### Settings that changed nothing

| Change | Result |
|---|---|
| 16 → 32 frames per window | identical grid `[8,10,16]`, 471 tokens, no change |
| 2 → 5 → 10 → 30 fps input | identical 506 tokens; the processor resamples to 16 frames regardless |
| 256 → 384 px | identical verdict on 37 of 37 windows |
| 384 px on the image path | OOM |

The video processor has `do_sample_frames=True` and its own `fps=2`, so it
normalises whatever it is given. Benchmark against the grid the processor
actually emits, not the frame count you passed in.

### Prompt wordings

| Prompt style | F1 | Predicted work | Note |
|---|---|---|---|
| `yes` / `no`, explicit exclusion list | 0.333 | 67.6% | baseline |
| `Moving your hands is not work. Changing something is work.` | 0.333 | **18.9%** | precision 46% → 78%, FP 20 → 5 |
| "work for MOST of these 8 seconds" + "when unsure, say no" | 0.000 | 0.0% | never fires |
| `yes - <action>` | 0.000–0.333 | — | asking for the label breaks the verdict |
| `WORK: <label>` / `IDLE: <label>` | 0.238 | 100% | never says idle |

Naming what does *not* count is what makes a small VLM say no at all. Four
variants measured against GT: the strict "look only at the hands … answer no
for anything else, including walking with an object" wording scored 55%
accuracy against 28–31% for looser phrasings.

### Input path

| Path | Tokens | Inference | Note |
|---|---|---|---|
| **video + `video_metadata(fps=2)`** | **470** | **2.9 s** | 5 of 5 GT work windows found |
| video without fps metadata | 182 | 2.5 s | model assumes 24 fps → thinks an 8 s clip is 0.67 s; 3/8 correct |
| 16 separate images | 1347 | 5.5 s | OOM-prone; the original baseline path |

Passing `video_metadata` is not optional. Without it the model is told the
wrong time scale and the temporal reasoning is meaningless.

### Reading the source

| Source | Read time per window |
|---|---|
| MCAP direct (H.264 via ffmpeg from the preceding keyframe) | 2316 ms |
| MP4 (`cv2.CAP_PROP_POS_MSEC` seeks) | 3079 ms |

MCAP is 25% faster and skips the conversion step and the 810 MB duplicate
entirely. The colour topic is `/ego/zed_head/left/image_compressed`, schema
`foxglove.CompressedVideo` — H.264 packets, not JPEG, so they cannot be
decoded one at a time.

---

## Metric pitfall

Ground-truth work segments are 3–12 s, nine of eleven shorter than the 8 s
window. Under a non-overlapping window scheme with a ">50% work" rule, a 5 s
segment straddling a boundary lands as 25% + 37% and **both windows score
idle**. That is why the window-level metric reports 13.5% work while the video
is 20.7% work — a third of the work is lost to window alignment before the
model is even consulted.

All the window-level F1 numbers in this repository are therefore pessimistic.
The tick-level numbers (exp1 onward) are the honest ones. Scored per tick, the
old non-overlapping protocol is F1 0.405, not 0.333.

**A second caveat:** the baseline that scored F1 0.600 could not be reproduced.
Re-running the identical script with the identical prompt and decode settings
gave F1 0.000 on a later day. The cause was not found. Treat 0.600 as
unverified and compare only numbers produced under the frozen protocol
(`scripts/cosmos_freeze.py`).

---

## Open problem

**Precision.** At the chosen operating point the detector predicts 59.4% work
against a ground truth of 20.6%. Adjacent false positives merge into blocks:
the four spans it produces are 118 s, 42 s, 14 s and 4 s. The 118 s block
swallows a 90 s idle stretch.

Splitting long spans with X3D-M motion change points was tried
(`exp4_split.py`, not included as it did not help): 118 s → ten sub-spans,
longest 36 s, but ten of the sixteen sub-spans still land in false-positive
territory and the labels degrade to a repeated `unpacking boxes`. Splitting a
wrong span produces smaller wrong spans.

Everything downstream — span boundaries, labels, the taxonomy — is blocked on
precision. Directions not yet measured: prompt ensembling with logit
averaging, contrastive scoring (`p(work) − p(idle)` to cancel the yes-bias),
hand-object interaction detection, optical-flow ego-motion compensation.

---

## Environment setup

Two virtualenvs, because the VLM needs a newer `transformers` than the rest of
the stack tolerates.

```bash
# ~/photon-env — moondream2, X3D, pytorchvideo, transformers 4.56.2
# ~/cosmos-env — Cosmos-Reason2, transformers 4.57.1

python3 -m venv ~/cosmos-env
source ~/cosmos-env/bin/activate
# reach the JetPack torch without copying it: pip install torch here would
# fetch an x86 wheel and break CUDA (this happened once, via torchcodec)
echo /home/user/photon-env/lib/python3.10/site-packages \
  > ~/cosmos-env/lib/python3.10/site-packages/_jetson_torch.pth

pip install "transformers==4.57.1" "numpy<2" "opencv-python-headless<5" \
            accelerate pillow "compressed-tensors==0.13.0" \
            typing_extensions filelock "sympy==1.13.1" networkx jinja2 fsspec requests
```

Then, every session:

```bash
source ~/cosmos-env/bin/activate
export CUDART128=/home/user/photon-env/lib/python3.10/site-packages/nvidia/cuda_runtime/lib
export LD_LIBRARY_PATH=$CUDART128:$HOME/opt/cusparselt:/usr/local/cuda/lib64:/usr/local/cuda/targets/aarch64-linux/lib:$LD_LIBRARY_PATH
```

### Version pins that are not optional

| Package | Pin | Why |
|---|---|---|
| `transformers` | **4.57.1** | `qwen3_vl` first appears here; 5.x rejects the JetPack torch version string `2.5.0a0` outright |
| `compressed-tensors` | **0.13.0** | 0.9.4 has no `compress_model`; 0.17+ needs `torch.distributed.Work`, added after torch 2.5 |
| `numpy` | **< 2** | the JetPack torch is built against NumPy 1.x; with NumPy 2 `torch.numpy()` raises "Numpy is not available". OpenCV 4.14+ drags NumPy 2 back in, so pin OpenCV too |
| `attn_implementation` | **`"eager"`** | JetPack's torch 2.5.0a0 has no `enable_gqa` in `scaled_dot_product_attention`; SDPA raises `TypeError` on every call |

`libcusparseLt.so.0` ships with neither JetPack nor apt. Unpack it from the
NVIDIA tarball into `~/opt/cusparselt` (no sudo needed) and put it on
`LD_LIBRARY_PATH`.

### torchvision shim

No aarch64 torchvision wheel matches torch 2.5.0a0, and the PyPI one breaks
CUDA. A hand-written shim under `site-packages/torchvision/` provides what is
actually used. Three additions were needed for this work specifically:

- `transforms.v2` namespace — `transformers` probes for it and falls back to
  v1, but a missing module raises `ModuleNotFoundError` before the fallback
- `functional.pil_to_tensor` and `to_pil_image` — the fast image processor
  calls them; `to_tensor` is not a substitute (it rescales to 0–1)
- a tensor branch in `functional.resize` — Cosmos passes tensors, the original
  shim assumed PIL and produced a bogus 4-tuple size

---

## Repository layout

```
scripts/
  cosmos_freeze.py       frozen benchmark protocol + window-length sweep
  cosmos_bench.py        the original 8 s benchmark (the unreproducible 0.600)
  cosmos_v9.py           three prompt wordings on the video path
  cosmos_edge2v.py       Edge2 vs base checkpoint, identical frames
  cosmos_video_probe.py  image input vs video input
  cosmos_fps_probe.py    2 / 5 / 10 / 30 fps input
  mcap_vs_mp4.py         MCAP direct read vs MP4 seeking
  exp1_overlap.py        overlapping windows + logit-level scoring   <- current best
  exp3_label.py          verb + noun per span, GT spans and detected spans

results/                 raw JSON from every run above
frames/                  overlay stills referenced in this README
```

Each script is standalone and writes its own JSON. Paths to the video and
ground truth are constants at the top of each file.

---

## Reproducing

```bash
source ~/cosmos-env/bin/activate    # plus the LD_LIBRARY_PATH exports above
cd scripts

python3 exp1_overlap.py     # work/idle, ~70 min for 146 windows
python3 exp3_label.py       # verb + noun per span, ~2 min
```

`exp1_overlap.py` writes `exp1_overlap.json`, which `exp3_label.py` reads to
derive its spans.
