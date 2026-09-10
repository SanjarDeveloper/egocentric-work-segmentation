# Egocentric work segmentation — IMU gate + VLM labels

Temporal action segmentation from head-mounted video on a **Jetson Orin NX 8 GB**.
Output is `idle` when the worker is not working, and `verb + noun` when they are.

This branch replaces the VLM-only approach of the previous work. A vision-language
model is asked *what* the worker is doing; **wrist IMU** decides *when* they are
doing it. The split came out of measurement, not preference — the numbers are below,
including the case where the gate does not work.

> The previous approach — sliding-window VLM for both the verdict and the label — is
> kept in `scripts/` and documented in [README_v1_vlm_only.md](README_v1_vlm_only.md).
> It is the baseline the timings here are compared against, and its findings on
> quantisation, model sizes and prompt wording still hold.

---

## Results

Three recordings, all scored at tick level (0.1 s) against the recorder's own labels
where those exist. **AUC** is the probability that a random work tick scores above a
random idle tick; 0.5 means the score carries no information.

| Recording | Scene | GT work | VLM AUC | Wrist IMU AUC | Fused AUC | Beats always-yes? |
|---|---|---|---|---|---|---|
| CRM `5e19608c` | wiping trays, rinsing cloth | 20.7 % | — | **0.783** | **0.819** | **yes**, F1 0.343 → 0.579 |
| GAS `47cb1429` | testing stove knobs, inspecting | 57.3 % | 0.611 | 0.575 | 0.638 | **no**, F1 0.729 → 0.633 |
| actseg `sample` | unpacking equipment | no labels | — | — | — | not measurable |

**The gate works on large-amplitude work and fails on fine work.** Wiping and rinsing
swing the wrist; turning a knob and feeling a surface do not, and on the GAS clip no
channel — vision, motion, or their combination — separates work from idle better than
predicting "work" everywhere.

Labelling, by contrast, was stable on every clip.

### Why the always-yes column matters

A clip that is 57 % work gives F1 0.729 to a model that knows nothing and answers
"yes" to everything. Any F1 quoted without that baseline is misleading, so
`common.always_yes()` computes it and every script prints it.

### What each channel contributes

Measured on the CRM clip (`analysis/score_imu_features.py`):

| Feature | AUC |
|---|---|
| both wrists, max, jerk, 8 s | **0.783** |
| both wrists, max, gyro, 8 s | 0.776 |
| both wrists, **mean**, jerk, 8 s | 0.722 |
| single wrist, jerk | 0.673 |
| **head IMU, any feature** | **0.39 – 0.47** |

Two findings worth keeping:

- **`max` beats `mean` across both wrists** (0.783 vs 0.722). A worker uses one
  dominant hand; averaging the two dilutes the signal.
- **Head motion is an anti-signal.** Below 0.5 on its own, so subtracting it helps:
  0.783 → 0.819. When the head swings the worker is looking around or walking, which
  is exactly the case a camera cannot distinguish from bending over a task.

---

## Architecture

```
MCAP
 ├── /imu/wrist_left    200 Hz ─┐
 ├── /imu/wrist_right   200 Hz ─┤   gate: when is there work
 ├── /imu/head          200 Hz ─┘
 │
 └── ego colour images ─────────┐   labels: what is the work
                                │
   score = z(wrist gyro/jerk) − 0.75·z(head jerk)   [+ z(VLM p_yes) when fused]
                                │
              threshold → merge gaps → drop short → split long
                                │
                            spans ──→ VLM, 2 queries each
                                        1. object   (open answer)
                                        2. verb     (closed set, logit-scored)
                                │
                       idle  |  verb + noun
```

### The gate

Three signals, z-normalised and combined:

```python
score = z(wrist_gyro) - 0.75 * z(head_jerk)              # sensor-only
score = z(vlm_p_yes) + 0.50 * z(wrist_jerk) - 0.50 * z(head_jerk)   # fused
```

- **jerk** — `|d|a|/dt|`, smoothed over 4–8 s. Penalises smooth carrying relative to
  manipulation.
- **gyro magnitude** — rotation, which turning a knob produces and walking does not.
- **head jerk, subtracted** — the anti-signal described above.

Longer smoothing windows monotonically beat shorter ones (1 s → 8 s), because work
episodes are sustained rather than instantaneous.

### The labels

Two separate queries per span, never one:

```
1. "What single object are the worker's hands touching or working on
    for most of the clip?  Name only the object, one or two words."

2. "The worker's hands are working on: <object>.
    Which one of these best describes what the hands are doing?
    testing / adjusting / inspecting / placing / opening / closing / ..."
```

The verb is **not generated**. Each option's first-token logit is read at the answer
position and softmaxed, so the output is a ranking with a confidence, and a word
outside the list cannot be invented.

Both choices were forced by measurement:

- **Asking for verdict and label together destroys the verdict.** `yes/no` scored
  F1 0.600; `yes - <label>` scored 0.000–0.333; `WORK:/IDLE:` scored 0.238. Measured
  four times.
- **Open verb generation inverts actions.** On an assembly clip the model produced
  *"disconnecting hose"* — wrong verb, hallucinated noun. A closed set makes
  `disconnecting` impossible to pick when `assembling` is on the list, and it fixed
  the verb. It did **not** fix the noun.

### Known limitation: nouns are still unreliable

Closed-set verbs corrected the verb but object identification remains weak. On one
continuous assembly the model named seven different objects across seven 20 s windows.
Confidence scores are honest about it — mostly 0.28–0.54.

Verbs are domain-independent and correct; nouns need either a domain-specific closed
list or a detector. Where only one is needed, take the verb.

---

## Timing and resource use

A 5-minute (300 s) recording on Jetson Orin NX 8 GB, measured with `tegrastats`:

| Stage | Time |
|---|---|
| IMU extraction (33 GB MCAP, sequential) | 122 s |
| Gate computation | **0.18 s** |
| VLM load | 15 s |
| Labelling (9 spans × 2 queries) | 76 s |
| **Total, IMU already extracted** | **90.6 s — 0.30× realtime** |
| **Total from raw MCAP** | ~213 s |

| Resource | Mean | Peak |
|---|---|---|
| RAM | 6529 MB | **7428 / 7619 MB (97 %)** |
| GPU | 52.8 % | 99 % |
| CPU (8-core mean) | 27.6 % | 86.3 % |
| Power | 11.8 W | 17.0 W |
| Temperature | 41.1 °C | 42.9 °C |

**RAM is the binding constraint** — 97 % used, nothing else can run alongside. GPU sits
at ~53 % because frame reading and JPEG decode are on the CPU.

For comparison, the VLM-only approach this replaces took **584 s** for the same clip
(1.95× realtime) — 147 sliding windows instead of 9 spans — and produced no usable
gate.

---

## Repository layout

```
common.py                     ground truth parsing, metrics, motion features,
                              span post-processing, tegrastats sampler

pipeline/
  extract_imu.py              IMU streams out of an MCAP (offsets auto-detected)
  extract_frames.py           colour frames + contact sheets per camera
  run_zed_wrist_imu.py        gate + labels, ZED layout  (CRM clip)
  run_orbbec_multicam.py      gate + labels, Orbbec layout (actseg clip)
  run_fused_gate.py           gate + labels, VLM+IMU fusion (GAS clip)
  overlay_with_gt.py          overlay video with a ground-truth strip
  overlay_no_gt.py            overlay video without one

analysis/
  score_imu_features.py       AUC of every IMU feature × smoothing window
  score_head_fusion.py        does subtracting head motion help
  score_vlm_imu_fusion.py     are VLM and IMU errors independent
  score_vlm_thresholds.py     is there any signal in p_yes at all
  vlm_only_baseline.py        the sliding-window VLM approach, for comparison

tools/
  survey_mcap.py              list topics and rates when the index is corrupt
  find_accel_offset.py        locate linear_acceleration by finding gravity
  find_gyro_offset.py         locate angular_velocity by finding variance

results/                      the JSON each run produced
```

---

## Reading these MCAPs

Three practical obstacles, all handled in code:

**1. The index is corrupt on every file in this set.** The footer reports a record
length of ~1.1e19, so `make_reader`, `get_summary` and every seek-based path fail.
`StreamReader` walks records in order without touching the summary. Some files are
also truncated — the GAS recording ends at 294.5 s of 300 s — so the readers catch
`EndOfFile` and keep what was decoded rather than discarding the scan.

**2. CDR field offsets shift between recordings**, because they depend on the length
of `frame_id`:

| Recorder | frame_id | accel offset |
|---|---|---|
| ZED | `wrist_right_imu_link` | 244 |
| Orbbec | `camera_CPA9B520080_accel_optical_frame` | 260 |

Nothing is hard-coded. `find_accel_offset` slides a 3×float64 window and takes the
offset whose **median** magnitude is closest to 9.81 — the median, not every sample,
because a hand-worn camera reaches 25 m/s² and requiring every sample near gravity
rejects the correct offset on the most active camera.

`find_gyro_offset` cannot use that trick, since angular velocity has no characteristic
magnitude. It takes the offset whose vector actually **varies** between messages. An
earlier heuristic — "the last plausible triple" — latched onto the orientation
quaternion, which reads as a rock-steady `|v| = 1.000` and silently produces a dead
channel. `extract_imu.py` now warns when any channel comes out constant.

**3. Orbbec splits the IMU across topics** (`/imu/accel` and `/imu/gyro`); ZED puts
both in one message. Both layouts are supported.

---

## Environment

Jetson Orin NX 8 GB, JetPack 6.2.1, CUDA 12.6, `torch 2.5.0a0`, sm_87.

Model: [`embedl/Cosmos-Reason2-2B-W4A16`](https://huggingface.co/embedl/Cosmos-Reason2-2B-W4A16)
— Qwen3-VL 2B, INT4 weights with FP16 compute, visual encoder unquantised.

Two virtual environments share the Jetson-native torch through a `.pth` file rather
than pip-installing it (pip fetches an x86 wheel and breaks CUDA):

```bash
source ~/cosmos-env/bin/activate
export CUDART128=/home/user/photon-env/lib/python3.10/site-packages/nvidia/cuda_runtime/lib
export LD_LIBRARY_PATH=$CUDART128:$HOME/opt/cusparselt:/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

Pinned versions and the reasons: `transformers==4.57.1` (5.x rejects the `2.5.0a0`
version string), `numpy==1.26.4` (2.x breaks `torch.numpy()`),
`compressed-tensors==0.13.0` (0.9.4 lacks `compress_model`, 0.17+ needs
`torch.distributed.Work`), and `attn_implementation="eager"` is mandatory — torch 2.5
has no `enable_gqa` in SDPA.

On quantisation: **W4A16 is the only workable format on sm_87.** NVFP4 needs Blackwell,
FP8 needs sm_89 or newer.

---

## Running it

```bash
# 1. what is in the file
python tools/survey_mcap.py recording.mcap

# 2. IMU streams (one sequential pass; offsets found automatically)
python pipeline/extract_imu.py recording.mcap imu.npz \
       /imu/head /imu/wrist_left /imu/wrist_right

# 3. how well the gate can possibly do, if you have labels
python analysis/score_imu_features.py imu.npz metadata.json

# 4. gate + labels
python pipeline/run_zed_wrist_imu.py

# 5. overlay
python pipeline/overlay_with_gt.py
```

Paths are constants at the top of each pipeline script — they were written against
specific recordings and are not yet argument-driven.

---

## Honest summary

**What works:** labelling. Verbs from a closed set were correct on every clip;
`wiping metal tray` against a ground truth of *"Wiping the baking tray"*, and
`assembling` where open generation had said `disconnecting`. It runs at 0.30× realtime
inside 8 GB.

**What works conditionally:** the work/idle gate. AUC 0.819 on wiping and rinsing;
0.638 on knob-turning, which is worse than answering "yes" to everything. It is
scene-dependent and should not be trusted without measuring it on the scene at hand.

**What does not work:** object naming. Confidence is low and the model contradicts
itself across windows of the same continuous action.

**What has not been tried:** hand–object contact detection. An accelerometer measures
whether the hand *moves*, not whether it is *working* — which is exactly the
distinction the GAS clip needs, and the reason no amount of threshold tuning rescued
it there.
