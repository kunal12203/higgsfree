<div align="center">

 <img src="assets/logo.png" alt="higgsfree logo" width="220" height="160" />

# higgsfree

**Open-source AI talking-avatar video generation pipeline.**

Turn a short consent video + a script into a photorealistic talking-head video — with
face identity preserved, voice cloned, and lips synced to generated speech.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)
![GPU](https://img.shields.io/badge/GPU-required%20(16--20GB%20VRAM)-76B900.svg)
![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)

[Quick Start](#-quick-start) · [How It Works](#-how-it-works) · [Architecture](#-architecture) · [Contributing](#-contributing)

</div>

---

## 🌍 Our Vision

**We're building the open-source, end-to-end video production pipeline — for everyone.**

Talking avatars are just the beginning. Our mission is to grow `higgsfree` into a **complete,
production-grade video creation stack** that takes you from raw idea to finished film: identity and
avatar generation, voice cloning, lipsync, B-roll, motion, scene compositing, restoration, and
final post — all **free, modular, and community-owned**.

The best creative tools shouldn't be locked behind closed APIs and paywalls. We believe
**studio-grade video generation should be open** — auditable, self-hostable, and shaped by the
people who use it. Every step is a reusable building block, every contribution makes the whole
ecosystem stronger, and every quality gain is measured in the open.

> **Build in the open. Create without limits. Own your pipeline.**

If that excites you, [come build it with us](#-contributing).

---

## ✨ What is higgsfree?

You give it **a short "consent video"** of a person and **a script of text**. It returns a video
of that person speaking the script:

- 🧬 **Face identity preserved** — the avatar genuinely looks like the source person (PuLID SDXL).
- 🎙️ **Voice cloned** — zero-shot voice cloning from the consent video's audio (Chatterbox).
- 👄 **Lip-synced** — mouth movement driven by the generated speech (Sonic, CVPR 2025).
- 🎬 **Scene compositing** — the talking head is placed into a full studio/cafe/desk scene.

It's designed to be **modular and contributor-friendly**: heavy ML steps are reusable building
blocks, and contributors compose them into new "pipelines". Every PR is automatically run on a
**real GPU** and scored for quality.

## 🎯 Pipelines

| Pipeline | Description | VRAM |
|----------|-------------|------|
| `avatar_studio` | Seated studio portrait with full scene compositing | 20 GB |
| `portrait` | Head-only talking head, minimal setup, faster previews | 16 GB |

## 🚀 Quick Start

> **Requirements:** an NVIDIA GPU with 16–20 GB VRAM (tested on g5.2xlarge / A10G 24 GB),
> CUDA 12.1, FFmpeg, and Python 3.10+.

```bash
# 1. Install dependencies
bash deploy/install_pipeline.sh

# 2. Run the avatar_studio pipeline
python pipelines/avatar_studio/run.py \
    consent_video.mp4 output.mp4 \
    --text "Hey! This is my talking avatar." \
    --scene studio --aspect 9:16
```

**CLI options**

| Flag | Values | Default |
|------|--------|---------|
| `--text` *(required)* | The script the avatar will speak | — |
| `--scene` | `portrait` · `studio` · `cafe` · `outdoor` · `desk` | `studio` |
| `--aspect` | `9:16` · `1:1` · `16:9` · `4:5` · `4:3` | `9:16` |

Stages are **cached** by output file — rerunning resumes from the last completed step.

### Run with Docker

```bash
docker build -t higgsfree -f deploy/Dockerfile .
docker run --gpus all higgsfree
```

## 🧠 How It Works

The `avatar_studio` pipeline runs **9 cached stages**:

```
consent_video.mp4 + "script text"
        │
  1. Extract best face frame   ──► face.jpg        InsightFace: sharpest, frontal, mouth-closed
  2. Generate avatar portrait  ──► avatar.png      PuLID SDXL identity + RealVisXL img2img realism
  3. Extract voice profile     ──► ref_audio.wav   Chatterbox: loudest ~10s window
  4. Synthesize speech         ──► cloned_audio.wav Chatterbox clone (Kokoro TTS fallback)
  5. Crop head+shoulders       ──► avatar_crop.png  removes hands → avoids SVD distortion
  6. Sonic lipsync             ──► sonic_face.mp4   SVD audio-driven talking head
  7. CodeFormer polish         ──► polished.mp4     per-frame face restoration
  8. Composite onto scene      ──► composited.mp4   feathered back-projection into full scene
  9. Mux audio (FFmpeg)        ──► output.mp4       final H.264 + AAC
```

**Design highlights**

- **Subprocess + multi-venv isolation** — each large model runs in its own virtualenv as a
  subprocess. This resolves conflicting dependencies *and* fully frees GPU VRAM between stages.
- **Graceful degradation** — Chatterbox → Kokoro, img2img → raw PuLID, CodeFormer → passthrough,
  local GPU → Replicate cloud fallback. The pipeline rarely hard-fails.
- **Idempotent caching** — every stage writes an artifact; reruns skip completed work.

## 🏗 Architecture

```
higgsfree/
├── pipelines/              # One folder per pipeline variant
│   ├── avatar_studio/      # config.yaml (declarative) + run.py (orchestrator)
│   └── portrait/
├── core/
│   ├── steps/              # Reusable ML steps (PuLID, Sonic, CodeFormer, Chatterbox...)
│   └── utils/              # gpu_env.py — CUDA/LD_LIBRARY_PATH wiring
├── worker/                 # Production SQS worker (S3 + Postgres + CloudFront)
├── eval/
│   └── score_pipeline.py   # Quality scoring: face similarity + lipsync confidence
├── ci/
│   ├── Jenkinsfile         # GPU pipeline — runs & scores PRs
│   └── workflows/          # GitHub Actions — wakes EC2 on approved PRs
├── deploy/                 # Dockerfile, install_pipeline.sh, requirements.txt
└── docs/
    └── adding_a_pipeline.md
```

**Layers**

- **Pipelines** — `config.yaml` declares steps + env (VRAM, venv); `run.py` orchestrates them with
  caching and fallbacks. Contributors compose `core.steps.*`, never duplicate ML code.
- **Core steps** — the reusable model primitives (see below).
- **Worker** — the SaaS-style backend: long-polls SQS, reads/writes Postgres, stores results in
  S3/CloudFront, processes multi-segment timelines, and self-stops its EC2 instance when idle.
- **Eval** — objective, automated quality score for any output video.
- **CI/CD** — maintainer label → GitHub Action wakes GPU EC2 → Jenkins runs + scores → posts a
  comparison comment on the PR.

### Core steps

| Module | What it does |
|--------|--------------|
| `avatar_gen` | Face extraction, PuLID portrait generation, img2img refinement, crop |
| `voiceclone` | Chatterbox zero-shot voice cloning |
| `tts` | Kokoro TTS fallback |
| `sonic_lipsync` | Sonic audio-driven lipsync |
| `codeformer_polish` | Per-frame face restoration |
| `face_composite` | Feathered face compositing onto scene background |
| `replicate_fallback` | Cloud fallback for lipsync when no local GPU is available |

## 📊 Quality Scoring

Every output is scored from **0.0 – 1.0** by `eval/score_pipeline.py`:

```
score = 0.5 × face_identity_similarity   # cosine sim of InsightFace embeddings (source vs output)
      + 0.5 × lipsync_confidence         # variance of mouth-opening across frames
```

```bash
python eval/score_pipeline.py source_video.mp4 output.mp4
```

## 🤝 Contributing

Contributions are very welcome — especially new pipelines, new `core/steps`, and quality
improvements that move the score.

### Ways to contribute

- **Add a new pipeline** — a new way to wire the existing steps together. Start with
  [`docs/adding_a_pipeline.md`](docs/adding_a_pipeline.md).
- **Add a new step** — integrate a new model into `core/steps/` as a self-contained function.
- **Improve quality** — beat the `avatar_studio` baseline score (face identity + lipsync).
- **Fix bugs / docs** — smaller PRs are great too.

### Contribution workflow

1. **Read** [`docs/adding_a_pipeline.md`](docs/adding_a_pipeline.md).
2. **Fork** the repo and create a feature branch (`git checkout -b my-pipeline`).
3. **Build** your change — import from `core.steps.*`, never duplicate step code.
4. **Open a PR** with a clear description of what changed and why.
5. A **maintainer reviews** and adds the `approved-for-ci` label.
6. **CI runs your pipeline on a real GPU** and posts a quality score comparison on the PR.
7. PRs that **match or beat the baseline** get merged; regressions are flagged.

> ℹ️ CI is gated behind the `approved-for-ci` label because every run consumes paid GPU time.
> A maintainer must approve before the GPU instance wakes — this keeps the project affordable and
> protects secrets from untrusted PRs.

### Ground rules

- ✅ **Import from `core.steps`** — never copy step code into a pipeline folder.
- ✅ **All config via `os.environ`** — never hardcode paths, keys, or secrets.
- ✅ **Declare honest `vram_gb`** in `config.yaml` so CI can skip if under-resourced.
- ✅ **`run.py` must accept** `--text`, `--scene`, and `--aspect` for CI compatibility.
- ✅ **Keep steps self-contained** — one clear function, no global side effects.
- ❌ **Never commit** `.env`, credentials, model weights, or generated media.

### Local development tips

- Stages are cached by output file, so iterate fast by reusing a workdir.
- No GPU? Most steps require CUDA, but the worker supports a **Replicate** cloud fallback for
  lipsync (`core/steps/replicate_fallback.py`).

## ⚙️ CI / CD

- **GitHub Actions** (`ci/workflows/wake-ec2.yml`) wakes the GPU EC2 instance when a maintainer
  adds the `approved-for-ci` label (or on push to `main`).
- **Jenkins** (`ci/Jenkinsfile`) runs the pipeline on a test video, scores it, and posts a
  PR comment comparing the PR score against the `main` baseline.
- Merges to `main` update the baseline score.

## 📜 License

[MIT](LICENSE) — free to use, modify, and distribute.

---

<div align="center">
<sub>Built for the open-source community. Face identity preservation · voice cloning · lipsync · face restoration.</sub>
</div>
