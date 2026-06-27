<div align="center">

<img src="assets/logo.png" alt="higgsfree logo" width="220" height="160" />

# higgsfree

**Open-source AI talking-avatar video generation pipeline.**

Turn a short consent video and a script into a photorealistic talking-head video —
face identity preserved, voice cloned, lips synced to generated speech.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)
![GPU](https://img.shields.io/badge/GPU-required%20(16--20GB%20VRAM)-76B900.svg)
![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)

[Quick Start](#quick-start) · [How It Works](#how-it-works) · [Architecture](#architecture) · [Contributing](#contributing)

</div>

---

## Overview

higgsfree is a modular, self-hostable pipeline for generating talking-avatar videos from a person's consent video and a text script. Every stage is a reusable building block; contributors compose them into new pipelines without duplicating model code.

- **Face identity preserved** — the avatar looks like the source person (PuLID SDXL + RealVisXL img2img)
- **Voice cloned** — zero-shot voice cloning from the consent video audio (Chatterbox TTS)
- **Lip-synced** — audio-driven mouth animation (Sonic, CVPR 2025)
- **Scene compositing** — talking head placed into a full studio, cafe, or desk scene

Every PR is run automatically on a real GPU and scored for quality before merge.

---

## Pipelines

| Pipeline | Description | VRAM |
|----------|-------------|------|
| `avatar_studio` | Seated studio portrait with full scene compositing | 20 GB |
| `portrait` | Head-only talking head, minimal setup, faster previews | 16 GB |
| `text_to_video` | CogVideoX-5b text→video (or image+prompt→video) | 24 GB |

---

## Quick Start

**Requirements:** NVIDIA GPU with 16–20 GB VRAM (tested on g5.2xlarge / A10G 24 GB), CUDA 12.1, FFmpeg, Python 3.10+.

```bash
# Provision the avatar_studio (Sonic) stack on a GPU box:
#   venv-sonic + Sonic/PuLID/CodeFormer + venv-chatterbox + Kokoro
bash deploy/install_sonic_stack.sh

# (legacy EchoMimicV2 / F5-TTS stack)
bash deploy/install_pipeline.sh

# Run the avatar_studio pipeline
python pipelines/avatar_studio/run.py \
    consent_video.mp4 output.mp4 \
    --text "Hey! This is my talking avatar." \
    --scene studio --aspect 9:16

# Add --upscale 2 for Real-ESRGAN 2x super-resolution (~4K output)
python pipelines/avatar_studio/run.py \
    consent_video.mp4 output.mp4 \
    --text "Hey!" --scene studio --aspect 9:16 --upscale 2

# Text-to-video (no avatar needed)
python pipelines/text_to_video/run.py output.mp4 \
    --text "A golden retriever running on a beach at sunset, cinematic, 4k" \
    --aspect 16:9
```

**CLI options**

| Flag | Values | Default |
|------|--------|---------|
| `--text` *(required)* | Script text the avatar will speak | — |
| `--scene` | `portrait` · `studio` · `cafe` · `outdoor` · `desk` | `studio` |
| `--aspect` | `9:16` · `1:1` · `16:9` · `4:5` · `4:3` | `9:16` |

Stages are cached by output file — rerunning resumes from the last completed step.

### Docker

```bash
docker build -t higgsfree -f deploy/Dockerfile .
docker run --gpus all higgsfree
```

---

## How It Works

The `avatar_studio` pipeline runs 9 cached stages:

```
consent_video.mp4 + "script text"
        │
  1. Extract best face frame   ──► face.jpg         InsightFace: sharpest, frontal, mouth-closed
  2. Generate avatar portrait  ──► avatar.png       PuLID SDXL identity + RealVisXL img2img realism
  3. Extract voice profile     ──► ref_audio.wav    Chatterbox: loudest ~10s window
  4. Synthesize speech         ──► cloned_audio.wav Chatterbox clone (Kokoro TTS fallback)
  5. Crop head+shoulders       ──► avatar_crop.png  removes hands, avoids SVD distortion
  6. Sonic lipsync             ──► sonic_face.mp4   SVD audio-driven talking head
  7. CodeFormer polish         ──► polished.mp4     per-frame face restoration
  8. Composite onto scene      ──► composited.mp4   feathered back-projection into full scene
  9. Mux audio (FFmpeg)        ──► output.mp4       final H.264 + AAC
```

**Design highlights**

- **Subprocess + multi-venv isolation** — each large model runs in its own virtualenv as a subprocess. This resolves conflicting dependencies and fully frees GPU VRAM between stages.
- **Graceful degradation** — Chatterbox falls back to Kokoro TTS, img2img falls back to raw PuLID, CodeFormer falls back to passthrough, local GPU falls back to Replicate. The pipeline rarely hard-fails.
- **Idempotent caching** — every stage writes a named artifact; reruns skip completed work automatically.

---

## Architecture

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
│   └── Jenkinsfile         # GPU pipeline — runs and scores PRs
├── .github/workflows/      # GitHub Actions — wakes EC2 on approved PRs
├── deploy/                 # Dockerfile, install_*.sh, requirements.txt
└── docs/
    └── adding_a_pipeline.md
```

**Layers**

- **Pipelines** — `config.yaml` declares steps and environment requirements (VRAM, venv); `run.py` orchestrates them with caching and fallbacks. Contributors compose `core.steps.*` and never duplicate ML code.
- **Core steps** — the reusable model primitives listed below.
- **Worker** — production backend: long-polls SQS, reads/writes Postgres, stores results in S3/CloudFront, processes multi-segment timelines, self-stops the EC2 instance when idle.
- **Eval** — objective, automated quality score for any output video.
- **CI/CD** — maintainer adds label → GitHub Action wakes GPU EC2 → Jenkins runs + scores → posts comparison comment on the PR.

### Core steps

| Module | What it does |
|--------|--------------|
| `avatar_gen` | Face extraction, PuLID portrait generation, img2img refinement, head+shoulders crop |
| `voiceclone` | Chatterbox zero-shot voice cloning |
| `tts` | Kokoro TTS fallback |
| `sonic_lipsync` | Sonic audio-driven lipsync |
| `codeformer_polish` | Per-frame face restoration |
| `face_composite` | Feathered face compositing onto scene background |
| `replicate_fallback` | Cloud lipsync fallback when no local GPU is available |
| `soul_id` | Persistent face-identity embedding per avatar (identity-locked frame selection + QA) |
| `text_to_video` | CogVideoX-5b text→video / image→video generation |
| `video_sr` | Real-ESRGAN 2x super-resolution upscale |

---

## Quality Scoring

Every output is scored from 0.0 to 1.0 by `eval/score_pipeline.py`:

```
score = 0.5 × face_identity_similarity   # InsightFace cosine similarity (source vs output)
      + 0.5 × lipsync_confidence         # variance of mouth-opening across frames
```

```bash
python eval/score_pipeline.py source_video.mp4 output.mp4
```

---

## Contributing

Contributions are welcome — new pipelines, new core steps, and quality improvements that move the score.

**Ways to contribute**

- Add a new pipeline — compose existing steps in a new way. Start with [`docs/adding_a_pipeline.md`](docs/adding_a_pipeline.md).
- Add a new step — integrate a new model into `core/steps/` as a self-contained function.
- Improve quality — beat the `avatar_studio` baseline score on face identity or lipsync.
- Fix bugs or documentation.

**Contribution workflow**

1. Read [`docs/adding_a_pipeline.md`](docs/adding_a_pipeline.md).
2. Fork the repo and create a feature branch.
3. Build your change — import from `core.steps.*`, never duplicate step code.
4. Open a PR with a clear description of what changed and why.
5. A maintainer reviews and adds the `approved-for-ci` label.
6. CI runs your pipeline on a real GPU and posts a quality score comparison on the PR.
7. PRs that match or beat the baseline get merged; regressions are flagged.

> CI is gated behind the `approved-for-ci` label because every run consumes paid GPU time.
> A maintainer must approve before the GPU instance wakes — this keeps the project sustainable
> and protects secrets from untrusted PRs.

**Ground rules**

- Import from `core.steps` — never copy step code into a pipeline folder.
- All configuration via `os.environ` — never hardcode paths, keys, or secrets.
- Declare honest `vram_gb` in `config.yaml` so CI can skip under-resourced runs.
- `run.py` must accept `--text`, `--scene`, and `--aspect` for CI compatibility.
- Keep steps self-contained — one clear function, no global side effects.
- Never commit `.env`, credentials, model weights, or generated media.

**Local development tips**

- Stages cache by output file — iterate fast by reusing a workdir across runs.
- No GPU? The worker supports a Replicate cloud fallback for lipsync via `core/steps/replicate_fallback.py`.

---

## CI / CD

- **GitHub Actions** (`.github/workflows/wake-ec2.yml`) wakes the GPU EC2 instance when a maintainer adds the `approved-for-ci` label to a PR.
- **Jenkins** (`ci/Jenkinsfile`) checks out the branch, runs the full pipeline on a test video, scores the output, and posts a comparison comment against the `main` baseline.
- Merges to `main` update the baseline score.

---

## License

[MIT](LICENSE) — free to use, modify, and distribute.

---

<div align="center">
<sub>Face identity preservation · voice cloning · lipsync · face restoration</sub>
</div>
