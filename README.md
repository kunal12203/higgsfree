# higgsfree

Open-source video generation pipeline. Face identity preservation, voice cloning, lipsync, face restoration — modular and contributor-friendly.

## Pipelines

| Pipeline | Description | VRAM |
|----------|-------------|------|
| `avatar_studio` | Seated studio portrait, full scene compositing | 20GB |
| `portrait` | Head-only talking head, minimal setup | 16GB |

## Quick Start

```bash
# Install dependencies
bash deploy/install_pipeline.sh

# Run avatar_studio pipeline
python pipelines/avatar_studio/run.py \
    consent_video.mp4 output.mp4 \
    --text "Hey! This is my talking avatar." \
    --scene studio --aspect 9:16
```

## Project Structure

```
higgsfree/
├── pipelines/              # One folder per pipeline variant
│   ├── avatar_studio/      # config.yaml + run.py
│   └── portrait/           # config.yaml + run.py
├── core/
│   ├── steps/              # Reusable pipeline steps (PuLID, Sonic, CodeFormer...)
│   └── utils/              # GPU env, shared helpers
├── eval/
│   └── score_pipeline.py   # Quality scoring: face similarity + lipsync confidence
├── ci/
│   ├── Jenkinsfile         # Jenkins pipeline (runs on GPU, scores PRs)
│   └── workflows/          # GitHub Actions (wakes EC2 on approved PRs)
├── deploy/
│   ├── Dockerfile
│   ├── install_pipeline.sh
│   └── requirements.txt
└── docs/
    └── adding_a_pipeline.md
```

## Contributing

1. Read `docs/adding_a_pipeline.md`
2. Fork the repo, create a branch
3. Open a PR — a maintainer will review and add the `approved-for-ci` label
4. CI runs your pipeline on real GPU and posts a quality score on the PR

## CI / CD

- **Jenkins** runs on the GPU instance at every approved PR
- **Score**: face identity similarity (50%) + lip sync confidence (50%)
- PRs that improve the score get merged; regressions are flagged

## Core Steps

| Module | What it does |
|--------|-------------|
| `avatar_gen` | Face extraction, PuLID portrait generation, img2img refinement, crop |
| `sonic_lipsync` | Sonic audio-driven lipsync |
| `codeformer_polish` | Per-frame face restoration |
| `voiceclone` | Chatterbox zero-shot voice cloning |
| `tts` | Kokoro TTS fallback |
| `face_composite` | Feathered face compositing onto scene background |
| `video_postproc` | FFmpeg post-processing filters |
| `lipsync` | LatentSync alternative lipsync |
| `hallo2` | Hallo2 audio-driven talking head |
