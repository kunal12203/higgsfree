# Adding a New Pipeline

Each pipeline lives in its own folder under `pipelines/`. It needs two files:

```
pipelines/your_pipeline_name/
├── config.yaml   — declares steps, params, environment requirements
└── run.py        — the actual runner script
```

## Step 1 — Create the folder

```bash
mkdir pipelines/your_pipeline_name
```

## Step 2 — Write config.yaml

```yaml
name: your_pipeline_name
version: "1.0"
description: >
  One paragraph describing what this pipeline does differently.

steps:
  - id: extract_face
    module: core.steps.avatar_gen
    fn: extract_best_face_frame

  - id: lipsync
    module: core.steps.sonic_lipsync
    fn: run_sonic
    params:
      dynamic_scale: 1.0

  # Add more steps...

environment:
  venv: /home/ubuntu/venv-sonic
  gpu: required
  vram_gb: 16        # honest estimate
```

## Step 3 — Write run.py

Copy `pipelines/avatar_studio/run.py` as a starting point. Import from `core.steps.*` — do not copy step code into your pipeline folder.

```python
import sys, os
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

from core.steps.avatar_gen import extract_best_face_frame
from core.steps.sonic_lipsync import run_sonic
# ...

def run(consent_video, output_path, workdir, text, **kwargs):
    # your pipeline logic here
    pass
```

## Step 4 — Add a new step (optional)

If your pipeline needs a step that doesn't exist in `core/steps/`:

1. Add the module to `core/steps/your_step.py`
2. Keep it self-contained — one clear function, no hardcoded paths
3. Add it to `core/steps/__init__.py`

## Step 5 — Open a PR

- The CI will run `eval/score_pipeline.py` against your pipeline
- A maintainer adds `approved-for-ci` label to trigger the GPU test
- Score is posted as a PR comment — aim to match or beat the `avatar_studio` baseline

## Rules

- Import from `core.steps` — never duplicate step code
- All secrets via `os.environ` — never hardcode paths or keys
- `config.yaml` must declare honest `vram_gb` so CI can skip if under-resourced
- `run.py` must accept `--text`, `--scene`, `--aspect` at minimum for CI compatibility
