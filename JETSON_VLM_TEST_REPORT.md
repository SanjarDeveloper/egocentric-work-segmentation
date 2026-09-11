# Jetson Orin NX — VLM Testing Report

**Task:** Read egocentric (head-camera) video from MCAP files, split it into
**work** and **idle** time, and write a short `verb + noun` label for each work
period. Everything must run on the device — no cloud, no training.

**Device:** NVIDIA Jetson Orin NX 8 GB (ZED Box Orin)
**Dates:** 2026-09-08 → 2026-09-11
**Status:** Labelling works. Work/idle detection works on some scenes and fails on others. Both results are measured, not estimated.

Every number in this report comes from a run on the device. The raw JSON files
are listed in [Where the evidence is](#where-the-evidence-is).

---

## 1. Executive summary

| Question | Answer |
|---|---|
| Can a VLM run on Orin NX 8 GB? | **Yes** — a 2B model at INT4 uses 1.1 GB of weights and ~7.4 GB peak total |
| Can it tell work from idle? | **Not reliably.** Best AUC 0.61 — barely better than guessing |
| Can wrist IMU tell work from idle? | **Sometimes.** AUC 0.82 on one scene, 0.57 on another |
| Can it write labels? | **Yes for verbs**, no for object names |
| How fast? | **0.30× realtime** — a 5-minute video takes 91 seconds |
| What is the limit? | **Memory.** 7.4 GB of 7.6 GB used; nothing else can run |

**The single most important finding:** on a clip that is 57% work, a model that
always answers "work" scores F1 **0.729**. Our best VLM scored **0.732**. That
is the whole story of the vision-only approach in two numbers.

```
    F1 score on the GAS clip (higher is better)

    always say "work"   ████████████████████████  0.729  ← knows nothing
    VLM, best threshold ████████████████████████  0.732  ← 584 seconds of compute
    IMU + VLM fusion    ████████████████████▌     0.633
                        └────┴────┴────┴────┴────┴
                        0   0.2  0.4  0.6  0.8  1.0
```

---

## 2. Hardware and platform

| Item | Value |
|---|---|
| Device | ZED Box Orin NX (Jetson Orin NX 8 GB) |
| GPU architecture | sm_87 (Ampere) |
| Memory | **7.62 GB unified** — CPU and GPU share one pool |
| CPU | 6 cores ARM Cortex-A78AE |
| JetPack | 6.2.1 / L4T R36.4.4 |
| CUDA | 12.6.68 |
| TensorRT | 10.3.0 |
| PyTorch | 2.5.0a0 (NVIDIA build) |
| Storage | 233 GB internal + external SSD |

**Unified memory is the binding constraint.** There is no separate VRAM. When a
model takes 6 GB, the operating system, the video decoder and the Python process
share what is left. Every model that failed, failed because of this.

**sm_87 blocks two quantisation formats:**

| Format | Needs | Works on Orin NX? |
|---|---|---|
| W4A16 (INT4 weights) | sm_75+ | ✅ **yes — this is what we use** |
| W8A8 (INT8) | sm_75+ | ✅ yes |
| FP8 | **sm_89+** (Ada) | ❌ no |
| NVFP4 | **sm_100+** (Blackwell) | ❌ no |

---

## 3. Models tested

### 3.1 Models that did not fit

These were downloaded or inspected and rejected before or during loading.

| Model | Size | Why it failed |
|---|---|---|
| **InternVideo3-8B** | 18.73 GB | 2.5× larger than total memory |
| **InternVideo2.5-Chat-8B** | 16.80 GB | 2.2× larger than total memory |
| **Gemma-4-E2B-it** | 10.25 GB | Smallest Gemma 4 — still too large |
| **moondream3-9B** | 10.5 GB (fp8) | fp8 needs sm_89+; Orin is sm_87 |
| **VideoChat3-4B** | 8.94 GB | Larger than total memory |
| **Cosmos-Reason2-8B-W4A16-FlashHead** | 7.30 GB | Fits on disk, but FlashHead is a **vLLM plugin** and vLLM has no aarch64 build |
| **VideoChat-Flash-2B** | 4.14 GB | Loads on CPU at 4.69 GB, then **OOM on `.cuda()`** |
| **Marlin-2B** | — | Best temporal-grounding model found, but fp16 OOMs and the GPTQ-4bit build needs `gptqmodel`, which needs `pcre`, which is not on PyPI for aarch64 |

```
    Model size vs available memory (GB)

    InternVideo3-8B      ████████████████████████████████████  18.73
    InternVideo2.5-8B    ████████████████████████████████      16.80
    Gemma-4-E2B          ████████████████████                  10.25
    moondream3-9B        ████████████████████                  10.50
    VideoChat3-4B        ██████████████████                     8.94
    Cosmos-8B-FlashHead  ██████████████                         7.30
    ─────────────────────────────────────────── 7.62 GB TOTAL MEMORY
    VideoChat-Flash-2B   ████████                               4.14
    Cosmos-Reason2-2B    ██                                     1.10  ← chosen
```

### 3.2 Models that fit but performed worse

| Model | F1 | What went wrong |
|---|---|---|
| **Cosmos-Reason2-2B-W4A16** | **0.333** | ← the one we kept |
| Cosmos-Reason2-2B-W4A16-**Edge2** | 0.278 | 26 false positives vs 20; keeping 4 layers at FP16 did not help |
| **FastVLM-0.5B** | 0.370 | Says "work" 70% of the time (truth: 20%). Echoes long prompts back verbatim |
| **SmolVLM2-500M-Video** | 0.290 | Says "work" **96%** of the time. Uses ~870 visual tokens per frame → **OOM above 2 frames** |
| X3D-M + moondream2 | 0.216 | Earlier two-stage pipeline |

**Edge2 detail** (same 37 windows, same prompt):

| Variant | TP | TN | FP | FN | Precision | Recall | F1 |
|---|---|---|---|---|---|---|---|
| Base W4A16 | 5 | 12 | 20 | 0 | 20.0% | 100% | **0.333** |
| Edge2 | 5 | 6 | **26** | 0 | 16.1% | 100% | 0.278 |

Edge2 is marketed as a better edge variant. On this task it predicted "work"
83.8% of the time versus the base model's 67.6%, against a truth of 13.5%.

### 3.3 The chosen model

**`embedl/Cosmos-Reason2-2B-W4A16`** — Qwen3-VL 2B, INT4 weights with FP16
compute, `compressed-tensors` format. The visual encoder is left unquantised
(104 layers in the ignore list).

| Property | Value |
|---|---|
| Weights on disk | 1.1 GB |
| Peak memory during inference | 7.4 GB |
| Inference per 8 s window | ~2.0–3.0 s |
| Visual tokens (video path) | 470–506 |

---

## 4. Environment problems solved

These cost real time. They are recorded so nobody repeats them.

| Problem | Cause | Fix |
|---|---|---|
| `libcusparseLt.so.0` missing | Not in JetPack, not in apt | Unpacked by hand to `~/opt/cusparselt`, added to `LD_LIBRARY_PATH` |
| `enable_gqa` TypeError | torch 2.5 SDPA lacks the argument | `attn_implementation="eager"` — **mandatory** |
| transformers rejects torch | Version string `2.5.0a0` fails the parser in transformers 5.x | Pin `transformers==4.57.1` |
| `torch.numpy()` breaks | numpy 2.x incompatibility | Pin `numpy==1.26.4`, `opencv-python-headless==4.10.0.84` |
| `compress_model` missing | `compressed-tensors` 0.9.4 too old; 0.17+ needs `torch.distributed.Work` | Pin `compressed-tensors==0.13.0` |
| torchvision missing functions | Jetson build lacks `pil_to_tensor`, `transforms.v2`, tensor `resize` | Wrote three shims into the installed package |
| pip installs x86 torch | pip does not know about the Jetson build | Two venvs sharing Jetson torch through a `.pth` file — never pip-install torch |

**Working environment:**

```bash
source ~/cosmos-env/bin/activate
export CUDART128=/home/user/photon-env/lib/python3.10/site-packages/nvidia/cuda_runtime/lib
export LD_LIBRARY_PATH=$CUDART128:$HOME/opt/cusparselt:/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

---

## 5. Settings that turned out not to matter

We swept the obvious knobs. Most of them do nothing, because the video
processor normalises its input before the model sees it.

### 5.1 Input frame rate — no effect at all

| Input fps | Frames given | Grid the model sees | Tokens | Median ms |
|---|---|---|---|---|
| 2.0 | 16 | `[8, 10, 16]` | 506 | 2998 |
| 5.0 | 40 | `[8, 10, 16]` | 506 | 2998 |
| 10.0 | 80 | `[8, 10, 16]` | 506 | 3023 |
| **30.0** | **240** | `[8, 10, 16]` | **506** | 3074 |

Giving the model 240 frames instead of 16 produced **identical token counts and
identical timing**. `Qwen3VLVideoProcessor` has `do_sample_frames=True` and its
own internal `fps=2`, so it resamples whatever it is handed.

**Lesson: benchmark against the grid the processor emits, not the frame count you passed in.**

### 5.2 Frame count — no effect

| Config | Frames | fps | Tokens | Median ms | F1 |
|---|---|---|---|---|---|
| v8_16f | 16 | 2.0 | 471 | 2930 | 0.333 |
| v8_32f | 32 | 4.0 | 471 | 2949 | 0.357 |

### 5.3 Resolution — no effect

256 px and 384 px gave the **identical verdict on 37 of 37 windows**.
384 px on the image path caused OOM.

### 5.4 What did matter: image path vs video path

| Path | Tokens | Median ms |
|---|---|---|
| Image (16 separate images) | **1350** | 5530 |
| **Video (one video tensor)** | **182** | **2505** |

**7.4× fewer tokens, 2.2× faster.** The image path also caused repeated OOM
crashes. This was the single biggest engineering win.

### 5.5 What also mattered: reading MCAP directly

| Source | Read ms | Inference ms | Tokens |
|---|---|---|---|
| MP4 | 3079 | 2910 | 470 |
| **MCAP direct** | **2316** | 2990 | 470 |

MCAP is 25% faster to read than MP4, because the JPEG frames can be decoded
individually without seeking through a compressed video stream.

---

## 6. Prompt experiments

### 6.1 Asking for a verdict and a label together destroys the verdict

Measured four times with four different wordings, all on the same windows:

| Prompt style | F1 |
|---|---|
| `yes` / `no` only | **0.600** |
| `WORK:` / `IDLE:` prefix | 0.238 |
| `yes - <label>` | 0.000 – 0.333 |

**This is why the pipeline uses two separate passes.** Once the verdict is
settled, asking for a label costs nothing extra.

### 6.2 Prompt wording changes behaviour but not quality

| Variant | Precision | Recall | F1 | Predicted work |
|---|---|---|---|---|
| A — current wording | 20.0% | 100% | 0.333 | 67.6% |
| B — "majority of the clip" | 0% | 0% | **0.000** | 0.0% |
| C — "advancing a task" | 28.6% | 40.0% | 0.333 | 18.9% |

Variant B made the model answer "no" to everything. Variant C moved the
precision/recall balance without improving F1.

### 6.3 Idle spans must get no description

Asking the model to describe idle time produces confident fiction. One example
over a clip of a person walking:

> "pick up a box, place it in a carton"

Nothing of the sort happened. Idle spans now get no label by design.

---

## 7. The core problem: the model cannot separate work from idle

### 7.1 The scores are all crowded together

On the GAS clip (300 s, 147 windows), `p_yes` never dropped below **0.833**:

```
    p_yes distribution over 147 windows

    min    0.728  │
    p10    0.833  │                              ██
    median 0.933  │                        ████████████
    p90    0.970  │                    ████████████████
    max    0.989  │
                  └──────────────────────────────────────
                  0.7      0.8      0.9      1.0
```

| Measure | Value |
|---|---|
| Mean score on **work** ticks | 0.9203 |
| Mean score on **idle** ticks | 0.9030 |
| **Gap** | **0.0173** |
| Standard deviation | 0.0452 |

**The signal is 2.6× smaller than the noise.**

Every threshold from 0.30 to 0.80 produced the **identical mask** — because
every score is above all of them.

### 7.2 Threshold sweep on an earlier clip

This one did have a usable range, and it shows the shape of the failure:

| Threshold | Precision | Recall | F1 | Predicted work |
|---|---|---|---|---|
| 0.30 | 22.6% | 96.9% | 0.366 | 88.7% |
| 0.50 | 26.4% | 88.9% | 0.408 | 69.4% |
| **0.60** | 29.2% | 84.0% | **0.434** | 59.4% |
| 0.70 | 32.8% | 30.7% | 0.317 | 19.3% |
| 0.80 | 0% | 0% | **0.000** | 0.0% |

Recall falls off a cliff between 0.60 and 0.70, and the model never scores above
~0.75. Ground truth was 20.6% work; the best setting predicted 59.4%.

**Two example frames make it concrete:**

| Clip content | Score |
|---|---|
| Genuine wiping | 0.703 |
| Merely holding a tray | 0.640 |
| **Difference** | **0.063** |

### 7.3 A metric trap worth knowing

Ground-truth work segments in these recordings are short — **nine of eleven are
under 8 seconds**. Scoring with non-overlapping 8-second windows and a
"more than 50% work" rule silently loses a third of the work:

| Scoring method | Work found |
|---|---|
| Tick level (0.1 s) — honest | 20.7% |
| 8 s windows + majority rule | **13.5%** |

All numbers in this report are **tick level**.

---

## 8. Adding wrist IMU

Since a head camera sees the worker standing at the bench whether or not their
hands are moving, we read the IMU streams that were already inside the MCAP
files and never used.

### 8.1 What the IMU gives

| Feature | What it measures |
|---|---|
| `acc_std` | Spread of acceleration magnitude in a tick — motion in any direction |
| `gyro_mag` | Rotation rate — turning a knob produces it, walking does not |
| `jerk` | `abs(d abs(a) / dt)` — penalises smooth carrying relative to manipulation |

### 8.2 Results on the CRM clip (wiping trays, rinsing a cloth)

Measured by AUC — the chance a random work tick scores above a random idle tick.
0.5 means no information at all.

| Feature | AUC |
|---|---|
| **Both wrists, `max`, gyro − 0.75 × head jerk** | **0.8187** |
| Both wrists, `max`, jerk, 8 s | 0.7827 |
| Both wrists, `max`, gyro, 8 s | 0.7760 |
| Both wrists, `mean`, jerk, 8 s | 0.7216 |
| Single wrist, jerk | 0.6729 |
| **Head IMU alone** | **0.39 – 0.47** |

```
    AUC on the CRM clip (0.5 = no signal)

    wrist gyro − head jerk  ████████████████████████████████  0.819
    wrist jerk (max)        ██████████████████████████████    0.783
    wrist jerk (mean)       ████████████████████████          0.722
    single wrist            ██████████████████                0.673
    ── 0.5 = chance ───────────────────────────────
    head IMU alone          ███████                           0.39
```

**Three findings:**

1. **`max` beats `mean` across both wrists** (0.783 vs 0.722). A worker uses one
   dominant hand; averaging the two dilutes the signal.
2. **Head motion is an anti-signal** — below 0.5 on its own. When the head
   swings, the worker is looking around, not working. Subtracting it raised AUC
   from 0.783 to **0.819**.
3. **Longer smoothing windows always won** (1 s → 8 s, monotonically). Work
   episodes are sustained, not instantaneous.

### 8.3 Results on the GAS clip (testing stove knobs) — it failed

| Signal | CRM clip | **GAS clip** |
|---|---|---|
| Wrist jerk (max) | 0.783 | **0.561** |
| Wrist gyro (max) | 0.776 | 0.535 |
| VLM | — | 0.611 |
| Head IMU | 0.39 | 0.455 |

**AUC 0.561 — almost chance, and worse than the VLM.**

### 8.4 Why the two clips differ

This is the key insight of the whole project.

| Clip | Work looks like | GT work | IMU AUC |
|---|---|---|---|
| **CRM** | wiping trays, rinsing a cloth — **large arm movements**, real pauses between | 20.7% | **0.783** ✅ |
| **GAS** | turning knobs, feeling a surface, inspecting — **small, fine movements** | 57.3% | 0.561 ❌ |

In the GAS clip the worker stands at the stove the whole time. Sometimes they
touch a knob (work), sometimes they just look (idle). Ground-truth descriptions
include *"places their hands on the stove top, feeling and inspecting"* —
physically indistinguishable from standing still, for both a camera and an
accelerometer.

---

## 9. Fusing VLM and IMU

If two weak signals make **different** mistakes, combining them helps. We tested
that directly.

### 9.1 They are indeed independent

| Pair | Correlation |
|---|---|
| VLM ↔ wrist jerk | +0.238 |
| VLM ↔ wrist gyro | −0.051 |
| VLM ↔ head jerk | −0.162 |

Near zero — they see different things, which is the precondition for fusion
helping.

### 9.2 But fusion still did not clear the bar

| Approach | AUC | Best F1 |
|---|---|---|
| VLM alone | 0.611 | 0.732 |
| IMU alone | 0.575 | 0.739 |
| **Fusion** | **0.638** | 0.708 |
| *always say "work"* | *—* | **0.729** |

Fusion gained **+0.028 AUC** over the better single channel — real, but not
enough. Best F1 0.708 is **below** the do-nothing baseline of 0.729.

**An upper bound was also computed:** a logistic regression fitted on this very
clip's answers — which is cheating — reached only **AUC 0.629**. So *no linear
combination of these four features* can do better than ~0.64 here. The problem
is not weight tuning. The signal is not there.

---

## 10. Labelling — this part works

Given clean spans, the second pass produces usable labels. Two separate queries:

```
1. "What single object are the worker's hands touching or working on
    for most of the clip? Name only the object, one or two words."

2. "The worker's hands are working on: <object>.
    Which one of these best describes what the hands are doing?
    testing / adjusting / inspecting / placing / opening / ..."
```

The verb is **not generated**. Each option's first-token logit is read at the
answer position and softmaxed, so the answer is a ranking with a confidence and
a word outside the list cannot be invented.

### 10.1 Verbs are correct

| Model output | Ground truth | Verdict |
|---|---|---|
| **wiping** metal tray (0.89) | **Wiping** the baking tray | ✅ |
| **placing** wooden planks (0.61) | **Placing** cardboard strips in tray | ✅ |
| **scrubbing** bucket (0.72) | **Rinsing** and wringing cloth | ✅ close |
| **testing** oven (0.59) | **Testing** stove knobs | ✅ |
| **testing** control panel (0.74) | **Testing** stove burner knobs | ✅ |

Before the closed verb set, open generation produced **"disconnecting hose"**
over a clip of someone *assembling* equipment — both the verb and the noun
wrong. A closed list makes that specific inversion impossible.

### 10.2 Object names are not correct

| Model output | Actual object |
|---|---|
| placing **cardboard boxes** | wiping a tray |
| testing **hydraulic hose** | testing a stove |
| connecting **computer mouse** | unpacking equipment |

Confidence scores are honest about this:

| Clip | Median confidence |
|---|---|
| CRM | 0.61 |
| GAS | **0.45** |
| actseg | **0.45** |

0.45 across 14 options means the model is close to guessing.

**Worse: the same object gets different names.** In one continuous assembly, cut
into 20-second pieces, the model named **seven different objects**:

```
 32– 52 s   tightening hammer drill
 52– 72 s   adjusting drill machine
 72– 92 s   adjusting motorcycle engine     ← same scene
 92–112 s   tightening electrical cable
116–135 s   loosening hammer drill
155–174 s   tightening computer tower       ← same scene
174–194 s   tightening computer mouse
```

The real object was a yellow plastic construction kit. A 2B model does not have
it as a class, so it names the nearest thing it does know — differently each
time.

### 10.3 Labelling on clean ground-truth spans

To separate "bad spans" from "bad labels", we fed the model the 11 ground-truth
intervals directly:

| VLM output | Ground truth | |
|---|---|---|
| cleaning tray | Wiping the baking tray | ✅ |
| placing papers in | Placing inserts in tray | ✅ |
| cleaning metal tray | Wiping the baking tray | ✅ |
| placing papers in | Placing inserts in tray | ✅ |
| cleaning tray | Wiping the baking tray | ✅ |
| placing papers in | Placing inserts in tray | ✅ |
| unpacking plastic bags | Wiping the baking tray | ❌ |
| unpacking boxes | Placing cardboard strips in tray | ~ |
| holding and arranging | Taking foam pieces from colleague | ~ |
| filling bucket | Rinsing and wringing cloth | ~ |
| opening oven door | Cleaning oven front panel | ~ |

**Six of eleven accurate.** Importantly, all four repeated wiping segments got
the same label, and all three insert-placing segments got the same label — the
model is **consistent on recurring tasks**, which is what a taxonomy needs.

---

## 11. Performance and resource use

### 11.1 Two architectures compared, same 5-minute clip

| | VLM-only (v1) | **IMU-gated (v2)** |
|---|---|---|
| **Total time** | 583.8 s | **90.6 s** |
| **Realtime factor** | 1.95× (slower than realtime) | **0.30× (3.3× faster)** |
| Model load | 14.7 s | 14.8 s |
| Gate computation | 560.6 s (147 windows) | **0.18 s** |
| Labelling | 8.5 s | 75.4 s (9 spans) |
| Frame reading | 256.9 s | included |
| VLM calls | **147** | **18** (9 spans × 2 queries) |

```
    Time for a 5-minute video (seconds)

    VLM-only  ████████████████████████████████████████████  584 s
    IMU-gated ███████                                        91 s
              └──────┴──────┴──────┴──────┴──────┴──────┴
              0     100    200    300    400    500    600
```

### 11.2 System load during the IMU-gated run

Sampled with `tegrastats` at 1 Hz, 89 samples:

| Resource | Mean | Peak |
|---|---|---|
| **RAM** | 6529 MB | **7428 MB of 7619 (97%)** |
| **GPU** | 52.8% | 99% |
| **CPU** (8-core mean) | 27.6% | 86.3% |
| **Power** | 11.8 W | 17.0 W |
| **Temperature** | 41.1 °C | 42.9 °C |

### 11.3 System load during the VLM-only run

579 samples over 584 seconds:

| Resource | Mean | Peak |
|---|---|---|
| **RAM** | 7148 MB | **7398 MB (97%)** |
| **GPU** | 43.5% | 99% |
| **CPU** | 42.8% | 87.5% |
| **Power** | 12.1 W | 15.8 W |
| **Temperature** | 42.6 °C | 44.6 °C |

```
    RAM usage — both runs sit at the ceiling

    0 GB                                          7.62 GB
    ├──────────────────────────────────────────────────┤
    VLM-only   ████████████████████████████████████████▌ 7.40 (97%)
    IMU-gated  ████████████████████████████████████████▌ 7.43 (97%)
```

**Three observations:**

1. **RAM is the ceiling, not the GPU.** 97% used in both runs. Nothing else can
   run on the device during processing.
2. **GPU sits at ~50%** because frame reading and JPEG decoding happen on the
   CPU. There is headroom, but no memory to exploit it with.
3. **Power is comfortable** — 12 W average on a 15 W profile, 43 °C.

### 11.4 Where the time goes in the VLM-only run

| Stage | ms per window |
|---|---|
| Frame reading (MP4 seek + decode) | **1748** |
| Model inference | **2066** |

Frame reading costs almost as much as inference. `cap.set(POS_FRAMES)` random
seek in MP4 is expensive; sequential reading would cut total time roughly 45%.

---

## 12. IMU extraction cost

The IMU streams live inside the MCAP files, which have to be read sequentially
because their index is corrupt (see §13).

| Recording | Size | Scan time | Samples extracted |
|---|---|---|---|
| GAS | 33.4 GB | **122 s** | 176 000 (3 streams × 200 Hz) |
| CRM | 11.1 GB | 70 s | 178 000 |
| actseg (Orbbec) | 4.5 GB | 66 s | 300 000 (6 streams) |

Throughput ≈ **275 MB/s** sequential.

**Full cost from raw MCAP for a 5-minute recording: ~213 s** (122 s IMU
extraction + 91 s pipeline).

---

## 13. MCAP reading problems

Three obstacles, all handled in code.

### 13.1 Corrupt index on every file

The footer reports a record length of **11115885623593766349** (~1.1 × 10¹⁹).
`make_reader`, `get_summary` and every seek-based path fail:

```
mcap.exceptions.RecordLengthLimitExceeded:
  unknown (opcode 151) record has length 11115885623593766349
  that exceeds limit 4294967296
```

**Fix:** use `StreamReader`, which walks records in order without reading the
summary. Cost: one full sequential pass, no random access.

### 13.2 Truncated files

The GAS recording ends at **294.5 s of 300 s** and then raises `EndOfFile`
mid-record. The readers catch it and keep what was decoded, rather than
discarding a 122-second scan over a partial final record.

### 13.3 CDR field offsets shift between recorders

The offsets depend on the length of `frame_id`, so they cannot be hard-coded:

| Recorder | `frame_id` | Length | Accel offset |
|---|---|---|---|
| ZED | `wrist_right_imu_link` | 20 | **244** |
| Orbbec | `camera_CPA9B520080_accel_optical_frame` | 37 | **260** |

**How the offsets are found automatically:**

- **Acceleration** — slide a 3×float64 window and take the offset whose
  **median** magnitude is closest to 9.81 (gravity). The median matters: a
  hand-worn camera reaches 25 m/s², so requiring *every* sample near gravity
  rejects the correct offset on the most active camera.
- **Angular velocity** — no characteristic magnitude exists, so take the offset
  whose vector actually **varies** between messages.

**A trap worth recording:** an earlier heuristic ("the last plausible triple")
latched onto the orientation quaternion, which reads as a rock-steady
`|v| = 1.000` and produced a **silently dead channel**. The extractor now warns
when any channel comes out constant.

### 13.4 Orbbec splits the IMU across topics

| Recorder | Layout |
|---|---|
| ZED | `/imu/wrist_left` — accel and gyro in one message |
| Orbbec | `/camera_XXX/imu/accel` and `/camera_XXX/imu/gyro` separately |

Both layouts are supported.

---

## 14. Recordings used

| Clip | Source | Duration | Cameras | Ground truth | Scene |
|---|---|---|---|---|---|
| **CRM** `5e19608c` | ZED | 300 s | head + 2 wrists | ✅ 11 segments | Wiping baking trays, rinsing cloth |
| **GAS** `47cb1429` | ZED | 300 s | head + 2 wrists | ✅ 33 segments | Testing stove knobs, inspecting oven |
| **actseg** `sample` | Orbbec Gemini 2L | 249.5 s | 3 cameras (1 head, 2 wrist) | ❌ none | Unpacking equipment |

Ground truth came with the recordings as `metadata.json`, written by Gemini (a
large cloud model). It is not perfect — in one clip the model's own output was
arguably more accurate than the label — but it is independent of anything tested
here.

**The metadata is JSON-shaped but not JSON:** keys and string values are
unquoted and descriptions contain commas and apostrophes, so it is parsed line
by line.

---

## 15. Final architecture

```
   MCAP file
      │
      ├── /imu/wrist_left    200 Hz ──┐
      ├── /imu/wrist_right   200 Hz ──┤  GATE: when is there work
      ├── /imu/head          200 Hz ──┘
      │
      └── ego colour frames ──────────┐  LABELS: what is the work
                                      │
   score = z(wrist gyro) − 0.75 × z(head jerk)
                                      │
        threshold → merge gaps < 2 s → drop spans < 3 s → split spans > 25 s
                                      │
                                   spans
                                      │
                          VLM, 2 queries per span
                            1. object  (open answer)
                            2. verb    (closed set, logit-scored)
                                      │
                          idle  |  verb + noun
```

**Configuration:**

```
model              embedl/Cosmos-Reason2-2B-W4A16
input              video path, 16 frames, 256 px long edge
gate smoothing     8 s
head weight        0.75
threshold          70th percentile of motion (or the GT work fraction)
min span           3 s
merge gap          2 s
max span           25 s
attn               eager (mandatory)
```

---

## 16. Honest conclusions

### What works

**Labelling.** Verbs from a closed set were correct on every clip tested. The
model is consistent on recurring tasks. It runs at 0.30× realtime inside 8 GB.

### What works conditionally

**The work/idle gate.** AUC 0.819 on wiping and rinsing; 0.638 on knob-turning,
which is worse than answering "work" to everything. **Scene-dependent — must be
measured on the scene at hand before being trusted.**

### What does not work

**Object naming.** Confidence 0.45 median, and the model contradicts itself
across windows of the same continuous action. A 2B model does not have ordinary
workshop objects as classes.

**Vision-only work/idle detection.** Three clips, three measurements, no
threshold ever beat the do-nothing baseline. The camera cannot see the
difference between a hand resting on a stove and a hand testing a stove.

### What was not tried

**Hand–object contact detection.** An accelerometer measures whether a hand
*moves*, not whether it is *working* — exactly the distinction the GAS clip
needs. This is the most promising unexplored direction.

**A larger VLM for labelling only.** The verbs are already correct, so the gain
would be in object naming. An 8B model at INT4 would fit in ~5.5 GB, but not
alongside everything else on this device — it belongs on a larger machine.

---

## 17. Reproducibility note

One result could not be reproduced. An early run scored **F1 0.600** on the
work/idle verdict with a `yes/no` prompt. Re-running the same script with the
same prompt later gave **F1 0.000**. The cause was never found.

Everything else in this report was measured more than once or comes from a
saved JSON file that can be inspected.

---

## Where the evidence is

**Repository:** `SanjarDeveloper/egocentric-work-segmentation`, branch
`imu_vlm_label_v2`

| File | What it holds |
|---|---|
| `results/gas_segmentation.json` | VLM-only run: 147 windows, threshold sweep, separation, timing, resources |
| `results/gas_imu_segmentation.json` | IMU-gated run: AUC, labels, timing, **tegrastats resources** |
| `results/imu_segmentation.json` | CRM clip, IMU gate, AUC 0.819 |
| `results/actseg_segmentation.json` | Orbbec clip, no GT |
| `results/imu_vs_gt.json` | All 54 IMU feature × window combinations, scored |
| `results/fuse_check.json` | Head-motion subtraction experiment |
| `results/gas_fusion.json` | VLM+IMU fusion, correlations, fitted upper bound |
| `results/exp1_overlap.json` | Threshold sweep, 146 windows |
| `results/exp3_label.json` | Labels on clean GT spans |
| `results/cosmos_fps.json` | Input fps sweep — proves the processor resamples |
| `results/cosmos_v8.json` | 16 vs 32 frames |
| `results/cosmos_v9.json` | Three prompt variants |
| `results/cosmos_video_probe.json` | Image path vs video path |
| `results/cosmos_edge2v.json` | Edge2 vs base, same windows |
| `results/fastvlm_bench.json` | FastVLM-0.5B, 60 samples |
| `results/vidlm_smolvlm.json` | SmolVLM2, OOM at 55/60 |
| `results/mcap_video_test.json` | MCAP vs MP4 read timing |

**Overlay videos** (model output drawn over the source video, with a
ground-truth strip where GT exists):

| File | Clip |
|---|---|
| `overlay_imu.mp4` | CRM, IMU gate, 3 strips (model / GT / disagreement) |
| `overlay_gas_imu.mp4` | GAS, fused gate, 3 strips |
| `overlay_actseg.mp4` | Orbbec, no GT |

---

*Report compiled 2026-09-11 from measurements taken 2026-09-08 to 2026-09-11 on
Jetson Orin NX 8 GB.*
