# Project Website + Hosting Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship a static showcase site (`site/`, Next.js, Vercel-ready) with an embeddable HF Space
demo package, hosting docs, and a publish-ready public repo — per
`docs/superpowers/specs/2026-07-17-website-hosting-design.md`.

**Architecture:** Monorepo. `site/` is a fully static Next.js App Router export (no server code).
The Gradio demo is packaged for a Hugging Face Space (CPU tier) by a staging script; the
GPU-only single-photo tab is gated at build time in `app.py` (visible but disabled with an honest
note when SSAT/CUDA is absent). Docs (`docs/HOSTING.md`) carry the $0 hosting plan + the manual
deploy runbook (Vercel, HF, DNS, repo-public flip) — those final clicks need the user's accounts.

**Tech Stack:** Next.js 15 (static export) + React 19, plain CSS (no Tailwind); Python/Gradio
(existing pipeline); PowerShell-free scripts (Python) so they run anywhere.

## Global Constraints

- Launch scope: showcase + hosted demo only. NO custom upload UI, NO owned inference API, NO auth/storage backend (spec "Out of scope").
- Site is fully static — no API routes, no server components that need a server (`output: 'export'`).
- Domain: `makeup-transfer.edwardlin.dev` (CNAME → `cname.vercel-dns.com`).
- HF Space: free CPU Basic tier; SSAT weights EXCLUDED from the Space; MediaPipe assets included. (Spec said BiSeNet too, but code inspection shows BiSeNet is only used by the GPU removal path — exclude it from the Space; note the correction in the progress log.)
- Engineering-notes section MUST include the failed Phase-4 prior gate, presented plainly.
- FFHQ-Makeup images: CC BY-NC-SA — attribution required everywhere they appear (repo README, site credits, Space README).
- Nothing tracked in git may be ≥ 100 MB; nothing here should even approach 10 MB (HF LFS threshold) so no LFS setup is needed.
- Vercel project Root Directory = `site/` (documented in runbook; not configurable from the repo).
- Existing test suite must stay green (`python -m pytest tests/ -q` from repo root, venv `.venv`).

**Executor environment notes (verified 2026-07-17):**
- Node v24.16.0 + npm 11.13.0 installed. Python venv at `.venv` (Windows: `.venv/Scripts/python.exe`).
- Tests run with `PYTHONPATH=src` via existing `conftest.py` (repo root on sys.path already handled).
- `weights/`, `data/ffhq_makeup/*`, `looks/*/`, `third_party/*/`, `preprocess_out/` are currently gitignored.
- `removal.py` exposes `SSAT_WEIGHTS` (Path) at module level and does NOT import torch at module level; `extract.py` imports `.removal` lazily inside `extract_look_single` — a torch-less machine can import `app.py` and run tabs 1/3/4.

---

### Task 1: Track the runtime assets the public repo needs

A fresh clone of the public repo must be able to run the CPU demo. Currently `weights/`,
sample data, and seed looks are all gitignored, and the README claims "weights already in the
repo" — false for a clone. Track the small runtime assets; keep `third_party/`, big outputs,
and the smoke-test npz out.

**Files:**
- Modify: `.gitignore`
- Track (git add): `weights/face_landmarker.task` (3.7 MB), `weights/canonical_face_model.obj` (46 KB), `weights/makeup_prior_pca.npz` (6.3 MB), `data/ffhq_makeup/*.jpg` (18 files, 635 KB total), `looks/ffhq-002386-03/`, `looks/ffhq-003125-02/`, `looks/ffhq-003417-05/` (~700 KB total)

**Interfaces:**
- Produces: a repo where `git clone` + `pip install -r requirements.txt` (no torch needed) runs the CPU tabs of `app.py`. Task 3's Space build script copies these same tracked paths.

- [ ] **Step 1: Update `.gitignore`**

Replace the current blanket ignores with targeted ones. New full content of `.gitignore`:

```gitignore
.venv/
__pycache__/
*.pyc
third_party/*/
!third_party/.gitkeep
# looks: only the three seed looks are tracked; everything else is user output
looks/*/
!looks/.gitkeep
!looks/ffhq-002386-03/
!looks/ffhq-003125-02/
!looks/ffhq-003417-05/
data/samples/*
!data/samples/.gitkeep
data/ffhq_makeup_train/
data/mt/*
!data/mt/.gitkeep
preprocess_out/
space_build/
site/node_modules/
site/.next/
site/out/
*.zip
# weights: MediaPipe assets + fitted PCA prior are tracked (small, needed at runtime);
# third-party .pth/.onnx stay out
weights/prior_smoke.npz
*.pth
*.onnx
*.tar
```

(Note: the old `weights/`, `data/ffhq_makeup/*`, `*.zip` before looks are replaced/kept as shown. `*.zip` stays — look exports; the site never needs zips.)

- [ ] **Step 2: Stage the assets and verify sizes**

```bash
git add .gitignore weights/face_landmarker.task weights/canonical_face_model.obj weights/makeup_prior_pca.npz data/ffhq_makeup/ looks/ffhq-002386-03 looks/ffhq-003125-02 looks/ffhq-003417-05
git status --short
# verify nothing staged is >= 10 MB:
git diff --cached --stat
```

Expected: ~25 new files, largest `weights/makeup_prior_pca.npz` (~6.3 MB). If anything ≥ 10 MB appears, STOP — that's a plan violation.

- [ ] **Step 3: Run the test suite (nothing should change, but verify)**

Run: `.venv/Scripts/python.exe -m pytest tests/ -q`
Expected: same pass/skip counts as before the change (GPU tests skip on CPU boxes).

- [ ] **Step 4: Commit**

```bash
git commit -m "Publish prep: track runtime assets (MediaPipe weights, PCA prior, FFHQ samples, seed looks)"
```

---

### Task 2: Gate the GPU-only tab in `app.py` (visible but disabled on CPU boxes)

Spec: on the CPU Space the single-photo tab "stays visible but disabled, with an honest
'needs a GPU — run locally' note" instead of failing on click.

**Files:**
- Modify: `src/makeup_transfer/app.py`
- Test: `tests/test_app_gating.py` (new)

**Interfaces:**
- Produces: `app._single_photo_available() -> tuple[bool, str]` (module-level, monkeypatchable); `build_demo()` unchanged signature. Task 3's Space runs this app unmodified.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_app_gating.py`:

```python
"""The single-photo (GPU) tab must be visible but disabled when SSAT/CUDA is absent."""
import gradio as gr
import pytest

from makeup_transfer import app


def _gpu_buttons(demo):
    return [b for b in demo.blocks.values()
            if isinstance(b, gr.Button) and b.value == "Extract look (GPU)"]


def test_single_photo_available_returns_tuple():
    ok, why = app._single_photo_available()
    assert isinstance(ok, bool)
    assert isinstance(why, str)
    if not ok:
        assert why  # a disabled tab must say why


def test_gpu_button_disabled_when_unavailable(monkeypatch):
    monkeypatch.setattr(app, "_single_photo_available",
                        lambda: (False, "needs a CUDA GPU"))
    demo = app.build_demo()
    btns = _gpu_buttons(demo)
    assert len(btns) == 1
    assert btns[0].interactive is False


def test_gpu_button_enabled_when_available(monkeypatch):
    monkeypatch.setattr(app, "_single_photo_available", lambda: (True, ""))
    demo = app.build_demo()
    btns = _gpu_buttons(demo)
    assert len(btns) == 1
    assert btns[0].interactive is not False  # None or True both mean enabled
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_app_gating.py -v`
Expected: FAIL — `AttributeError: module 'makeup_transfer.app' has no attribute '_single_photo_available'`.

- [ ] **Step 3: Implement the gate in `app.py`**

Add after the `_REGIONS` line:

```python
def _single_photo_available() -> tuple[bool, str]:
    """Can the SSAT single-photo tab actually run here? (ok, reason-if-not)."""
    from .removal import SSAT_WEIGHTS  # safe: removal.py imports torch lazily
    if not SSAT_WEIGHTS.exists():
        return False, ("SSAT weights are not present on this machine "
                       "(third_party/SSAT). Clone the repo and run locally with a GPU.")
    try:
        import torch
    except ImportError:
        return False, "PyTorch is not installed here."
    if not torch.cuda.is_available():
        return False, "No CUDA GPU is available here."
    return True, ""
```

In `build_demo()`, inside the Tab 2 block: call the gate once before building components,
adjust the intro markdown and the button:

```python
            with gr.Tab("Extract from one photo"):
                gpu_ok, gpu_why = _single_photo_available()
                note = "" if gpu_ok else (
                    f"\n\n⚠️ **Disabled on this machine:** {gpu_why} "
                    "The other tabs run fine on CPU."
                )
                gr.Markdown(
                    "One photo only. SSAT makeup **removal** synthesizes the bare face, then the "
                    "same pair pipeline runs. **Requires a CUDA GPU.**" + note
                )
                ...
                s_btn = gr.Button("Extract look (GPU)", variant="primary",
                                  interactive=gpu_ok)
```

(Only the markdown string and the `s_btn` line change; everything else in the tab stays.)

- [ ] **Step 4: Run the tests**

Run: `.venv/Scripts/python.exe -m pytest tests/test_app_gating.py tests/ -q`
Expected: new tests PASS, full suite green.

- [ ] **Step 5: Commit**

```bash
git add src/makeup_transfer/app.py tests/test_app_gating.py
git commit -m "app: disable the GPU single-photo tab up front on CPU machines (honest note, no click-to-fail)"
```

---

### Task 3: HF Space packaging (staging script + Space requirements + Space README)

The Space is a separate HF git repo. We stage its full contents into `space_build/`
(gitignored) with a script, so deploying is "run script, push folder". No torch on the
Space — tabs 1/3/4 don't need it (verified: `extract.py` imports `.removal` lazily).

**Files:**
- Create: `scripts/build_space.py`
- Create: `scripts/space_assets/README.md` (Space card w/ HF YAML frontmatter)
- Create: `scripts/space_assets/requirements.txt` (Space-only deps)
- Test: `tests/test_build_space.py` (new)

**Interfaces:**
- Consumes: tracked asset paths from Task 1; gated `app.py` from Task 2.
- Produces: `space_build/` laid out as: `app.py` (entry shim), `src/makeup_transfer/*.py`, `weights/{face_landmarker.task,canonical_face_model.obj}`, `looks/<3 seeds>/`, `data/ffhq_makeup/*.jpg + README.md`, `requirements.txt`, `README.md`. `build_space.py` exposes `stage(dest: Path) -> None`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_build_space.py`:

```python
"""build_space.stage() must assemble a self-contained CPU Space folder."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from build_space import stage  # noqa: E402


def test_stage_layout(tmp_path):
    dest = tmp_path / "space"
    stage(dest)
    # entry + code
    assert (dest / "app.py").is_file()
    assert (dest / "src" / "makeup_transfer" / "app.py").is_file()
    assert (dest / "src" / "makeup_transfer" / "extract.py").is_file()
    # CPU weights in, GPU weights + third_party out
    assert (dest / "weights" / "face_landmarker.task").is_file()
    assert (dest / "weights" / "canonical_face_model.obj").is_file()
    assert not (dest / "third_party").exists()
    assert not list(dest.rglob("*.pth"))
    # seeds + samples + metadata
    assert (dest / "looks" / "ffhq-002386-03" / "layer.png").is_file()
    assert (dest / "data" / "ffhq_makeup" / "003417_bare.jpg").is_file()
    reqs = (dest / "requirements.txt").read_text()
    assert "torch" not in reqs
    assert "mediapipe" in reqs
    card = (dest / "README.md").read_text(encoding="utf-8")
    assert card.startswith("---")          # HF YAML frontmatter
    assert "sdk: gradio" in card


def test_stage_entry_shim_imports_app(tmp_path):
    dest = tmp_path / "space"
    stage(dest)
    shim = (dest / "app.py").read_text()
    assert "makeup_transfer.app" in shim
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_build_space.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'build_space'`.

- [ ] **Step 3: Write the Space assets**

`scripts/space_assets/requirements.txt`:

```text
numpy<2.0
opencv-python-headless>=4.9
Pillow>=10.0
scipy>=1.11
scikit-image>=0.22
mediapipe>=0.10.14
gradio>=4.0
```

`scripts/space_assets/README.md`:

```markdown
---
title: AI Makeup Transfer
emoji: 💄
colorFrom: pink
colorTo: purple
sdk: gradio
sdk_version: "4.44.1"
app_file: app.py
pinned: false
license: mit
short_description: Extract makeup as a reusable RGBA layer; re-apply to any face.
---

# AI Makeup Transfer — live demo

Extract makeup from a face as a translucent **RGBA layer in 3D UV space**, then re-apply it
to any face with a skin-tone-aware blend (skin shows through; the look adapts per person).

**On this CPU Space:** extract-from-a-pair, apply-a-look, and the look library all work.
The *extract-from-one-photo* tab needs a CUDA GPU (SSAT makeup removal) and is disabled
here — clone the [GitHub repo](https://github.com/edwardlin-dev/ai-makeup-transfer) and
run locally to use it.

Sample FFHQ-Makeup images © their authors, **CC BY-NC-SA** —
[cyberagent/FFHQ-Makeup](https://huggingface.co/datasets/cyberagent/FFHQ-Makeup).
Website: https://makeup-transfer.edwardlin.dev
```

- [ ] **Step 4: Write `scripts/build_space.py`**

```python
"""Stage the Hugging Face Space folder (CPU demo) into space_build/.

Usage:  python scripts/build_space.py  [dest]

The Space is torch-free: only the CPU tabs' code + assets are copied. SSAT/BiSeNet
weights and third_party/ never ship (the GPU tab is disabled by app.py's runtime gate).
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ASSETS = Path(__file__).resolve().parent / "space_assets"

SEED_LOOKS = ["ffhq-002386-03", "ffhq-003125-02", "ffhq-003417-05"]
WEIGHTS = ["face_landmarker.task", "canonical_face_model.obj"]

ENTRY_SHIM = '''"""HF Space entrypoint: run the packaged Gradio app."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from makeup_transfer.app import build_demo

demo = build_demo()

if __name__ == "__main__":
    demo.launch()
'''


def stage(dest: Path) -> None:
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)

    # code (no __pycache__)
    shutil.copytree(ROOT / "src" / "makeup_transfer", dest / "src" / "makeup_transfer",
                    ignore=shutil.ignore_patterns("__pycache__"))
    # assets
    (dest / "weights").mkdir()
    for w in WEIGHTS:
        shutil.copy2(ROOT / "weights" / w, dest / "weights" / w)
    for look in SEED_LOOKS:
        shutil.copytree(ROOT / "looks" / look, dest / "looks" / look)
    shutil.copytree(ROOT / "data" / "ffhq_makeup", dest / "data" / "ffhq_makeup",
                    ignore=shutil.ignore_patterns("*.zip"))
    # space metadata + entry
    shutil.copy2(ASSETS / "requirements.txt", dest / "requirements.txt")
    shutil.copy2(ASSETS / "README.md", dest / "README.md")
    (dest / "app.py").write_text(ENTRY_SHIM, encoding="utf-8")


if __name__ == "__main__":
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "space_build"
    stage(out)
    n = sum(1 for _ in out.rglob("*") if _.is_file())
    print(f"Staged {n} files -> {out}")
```

- [ ] **Step 5: Run the tests**

Run: `.venv/Scripts/python.exe -m pytest tests/test_build_space.py -v`
Expected: 2 PASS.

- [ ] **Step 6: Smoke-test the staged Space actually builds its UI**

```bash
.venv/Scripts/python.exe scripts/build_space.py
cd space_build && ../.venv/Scripts/python.exe -c "import app; print(type(app.demo).__name__)" && cd ..
```

Expected: prints `Blocks`. (This runs the real `build_demo()` against staged assets — looks gallery seeds from the staged `looks/`, gate disables the GPU tab.)

Note: `LookStore` default resolves to `parents[2]` of `storage.py` = the staged root — correct by construction.

- [ ] **Step 7: Commit**

```bash
git add scripts/ tests/test_build_space.py
git commit -m "HF Space packaging: torch-free staging script + Space card/requirements"
```

---

### Task 4: Site scaffold (`site/`, Next.js static export, builds clean)

**Files:**
- Create: `site/package.json`, `site/next.config.mjs`, `site/tsconfig.json`, `site/app/layout.tsx`, `site/app/page.tsx`, `site/app/globals.css`, `site/.gitignore`

**Interfaces:**
- Produces: `npm run build` in `site/` emits a static `out/` with `index.html`. Task 5 fills `page.tsx` with the real sections; Task 5 relies on `site/public/` for images and `HF_SPACE_URL`/`GITHUB_URL` consts exported from `site/lib/links.ts`.

- [ ] **Step 1: Write the scaffold files**

`site/package.json`:

```json
{
  "name": "makeup-transfer-site",
  "private": true,
  "scripts": {
    "dev": "next dev",
    "build": "next build",
    "start": "next start"
  },
  "dependencies": {
    "next": "^15.4.5",
    "react": "^19.1.0",
    "react-dom": "^19.1.0"
  },
  "devDependencies": {
    "@types/node": "^22",
    "@types/react": "^19",
    "typescript": "^5"
  }
}
```

`site/next.config.mjs`:

```js
/** @type {import('next').NextConfig} */
const nextConfig = {
  output: "export",
  images: { unoptimized: true },
};

export default nextConfig;
```

`site/tsconfig.json`:

```json
{
  "compilerOptions": {
    "target": "ES2022",
    "lib": ["dom", "dom.iterable", "esnext"],
    "allowJs": true,
    "skipLibCheck": true,
    "strict": true,
    "noEmit": true,
    "esModuleInterop": true,
    "module": "esnext",
    "moduleResolution": "bundler",
    "resolveJsonModule": true,
    "isolatedModules": true,
    "jsx": "preserve",
    "incremental": true,
    "plugins": [{ "name": "next" }],
    "paths": { "@/*": ["./*"] }
  },
  "include": ["next-env.d.ts", "**/*.ts", "**/*.tsx", ".next/types/**/*.ts"],
  "exclude": ["node_modules"]
}
```

`site/.gitignore`:

```gitignore
node_modules/
.next/
out/
next-env.d.ts
*.tsbuildinfo
```

`site/lib/links.ts`:

```ts
// Single place to update once the Space/repo/domain go live (see docs/HOSTING.md runbook).
export const HF_SPACE_URL = "https://edwardlindev-makeup-transfer.hf.space";
export const HF_SPACE_PAGE = "https://huggingface.co/spaces/edwardlindev/makeup-transfer";
export const GITHUB_URL = "https://github.com/edwardlin-dev/ai-makeup-transfer";
```

`site/app/layout.tsx`:

```tsx
import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "AI Makeup Transfer — makeup as a portable RGBA layer",
  description:
    "Extract makeup from a photo as a translucent RGBA layer in 3D UV space; re-apply it skin-tone-aware to any face.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
```

`site/app/page.tsx` (placeholder for Task 5 — minimal but real):

```tsx
export default function Home() {
  return (
    <main>
      <h1>AI Makeup Transfer</h1>
    </main>
  );
}
```

`site/app/globals.css` (baseline; Task 5 replaces with the real design):

```css
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: system-ui, sans-serif; }
```

- [ ] **Step 2: Install and build**

```bash
cd site && npm install && npm run build
```

Expected: `✓ Compiled successfully`, `out/index.html` exists. If npm install fails on network, STOP and report.

- [ ] **Step 3: Commit**

```bash
git add site/
git commit -m "Site scaffold: Next.js static-export app (Vercel root dir = site/)"
```

(`site/.gitignore` keeps `node_modules/`, `out/` untracked; root `.gitignore` from Task 1 also covers them.)

---

### Task 5: Site content — hero, pipeline, results, demo embed, notes, credits

Single page, anchored sections, per spec order. **Invoke the frontend-design skill before
writing the CSS/JSX** — portfolio-quality, distinctive, no template feel; but the section
structure, copy, and assets below are fixed content requirements.

**Files:**
- Create: `site/public/results/*.png|jpg` (copied from `preprocess_out/` + `data/`)
- Create: `site/components/Pipeline.tsx` (SVG diagram)
- Modify: `site/app/page.tsx`, `site/app/globals.css`
- Consumes: `site/lib/links.ts` constants from Task 4.

- [ ] **Step 1: Copy the result assets**

```bash
mkdir -p site/public/results
cp preprocess_out/phase2/verification_grid.png site/public/results/
cp preprocess_out/phase1/ffhq-002386-03_side_by_side.png site/public/results/
cp preprocess_out/phase1/ffhq-003125-02_side_by_side.png site/public/results/
cp preprocess_out/phase25/spike_grid.png site/public/results/
cp preprocess_out/phase4/prior_grid.png site/public/results/
cp preprocess_out/phase1/ffhq-002386-03_layer_preview.png site/public/results/
cp data/ffhq_makeup/002386_makeup_03.jpg data/ffhq_makeup/002386_bare.jpg site/public/results/
```

- [ ] **Step 2: Build the page**

Required sections, in order, all reachable via nav anchors:

1. `#hero` — "AI Makeup Transfer" + the one-liner: *"Extract makeup as a translucent RGBA layer in 3D UV space; re-apply it skin-tone-aware to any face."* CTA buttons → `#demo`, GitHub.
2. `#how` — `<Pipeline/>` SVG: `photo pair → preprocess (MediaPipe 468 landmarks) → UV unwrap → illumination-aware difference → RGBA layer` then `layer + target → UV warp → per-region blend (multiply/soft-light/additive) → delta composite → result`.
3. `#results` — gallery: `verification_grid.png` (3 looks × 4 targets, captioned), the two `side_by_side` images, `layer_preview` next to its source pair (the "what a look IS" exhibit), `spike_grid.png` (single-photo extraction).
4. `#demo` — `<iframe src={HF_SPACE_URL} …/>` + direct link (`HF_SPACE_PAGE`) + honest note: pair-extract/apply/library run on CPU; single-photo needs a GPU → run locally; Space may cold-start after idle.
5. `#notes` — engineering notes, MUST include: Phase-4 PCA prior gate **FAILED** (false-alpha 0.1734 → 0.1122, target was ≤0.04; residual knob monotonically harmful; prior ships opt-in) with `prior_grid.png`; the 003417 SSAT skin-tone-drift failure; the mouth-interior alpha-bleed bug ("plum teeth"); sandbox-blocked 3DDFA_V2 build → canonical-UV workaround. Link to BUILD_PLAN.md on GitHub for numbers.
6. `#credits` — links: GitHub repo; papers/repos: makeup-extract, makeup-priors (CVPR 2024), SSAT (AAAI 2022), face-parsing.PyTorch (BiSeNet, MIT), MediaPipe; datasets: FFHQ-Makeup (cyberagent, CC BY-NC-SA), MT dataset. Footer: built by Edward Lin.

All `<img>` get `loading="lazy"` except the hero-adjacent one; the iframe gets `loading="lazy"` too (grids are ~2 MB each).

- [ ] **Step 3: Build and verify sections**

```bash
cd site && npm run build
grep -c 'id="hero"\|id="how"\|id="results"\|id="demo"\|id="notes"\|id="credits"' out/index.html
```

Expected: build passes; grep finds all 6 section ids (count ≥ 6 across nav+sections).

- [ ] **Step 4: Visual check**

Serve `site/out/` (`npx serve out` or `python -m http.server -d out`), screenshot/eyeball: readable at 375 px and 1440 px widths, images load, iframe placeholder renders (Space not live yet — a broken iframe is expected; verify the fallback link shows).

- [ ] **Step 5: Commit**

```bash
git add site/
git commit -m "Site: showcase page (hero, pipeline, results, demo embed, honest engineering notes, credits)"
```

---

### Task 6: Hosting docs + README/BUILD_PLAN updates + deploy runbook

**Files:**
- Create: `docs/HOSTING.md`
- Modify: `README.md` (fix the stale "weights in repo" claim, add site/demo links + license/credits section)
- Modify: `docs/BUILD_PLAN.md` (progress-log entry)

- [ ] **Step 1: Write `docs/HOSTING.md`** — must contain, per spec: current $0 setup table (Vercel Hobby / HF Space CPU Basic / GitHub public), DNS record (`makeup-transfer` CNAME → `cname.vercel-dns.com`), tier limits (no online GPU; Space sleeps → cold start; Vercel Hobby non-commercial), upgrade path (1. HF PRO/ZeroGPU → GPU tab online; 2. Modal/Replicate/RunPod serverless API + upload UI = full-web-app phase, ~$0.001–0.01/run; 3. real storage backend behind `storage.py` interface), and the **manual deploy runbook**:

  1. GitHub: run publish checklist (Task 7) → flip repo public.
  2. HF: `python scripts/build_space.py` → create Space `makeup-transfer` (Gradio, CPU Basic) → push `space_build/` contents → verify CPU tabs in browser.
  3. Vercel: import GitHub repo → Root Directory `site/` → deploy → check preview URL.
  4. DNS: add CNAME at the edwardlin.dev provider → add domain in Vercel → wait for TLS.
  5. Post-launch: update `site/lib/links.ts` if the Space/repo slugs differ, redeploy.

- [ ] **Step 2: Update `README.md`** — status line points at the site + Space + HOSTING.md; setup section says torch is optional (only single-photo/GPU path) and third_party is fetch-on-demand for the GPU tab; add "Licenses & credits" (FFHQ-Makeup CC BY-NC-SA attribution, MT dataset research-scoped, MediaPipe Apache-2.0, vendored repos listed with licenses, note repo content is for research/portfolio use).

- [ ] **Step 3: BUILD_PLAN progress-log entry** — dated 2026-07-17: website + hosting built (site/, Space packaging, gating change, tracked assets); spec correction noted (BiSeNet NOT needed by CPU tabs — excluded from Space); what remains manual (runbook in HOSTING.md).

- [ ] **Step 4: Commit**

```bash
git add docs/HOSTING.md README.md docs/BUILD_PLAN.md
git commit -m "Docs: hosting plan + deploy runbook; README licenses/credits; BUILD_PLAN progress entry"
```

---

### Task 7: Publish checklist + final verification

**Files:** none new (verification only; fixes go where the failure is)

- [ ] **Step 1: File-size gate**

```bash
git ls-files -z | xargs -0 -I{} du -b "{}" | sort -rn | head -5
```

Expected: max ≈ 6.3 MB (`makeup_prior_pca.npz`). Nothing ≥ 100 MB (GitHub), nothing ≥ 10 MB (keeps LFS out of play).

- [ ] **Step 2: Secrets/personal-data scan**

```bash
git log --all --oneline | wc -l
git grep -iE "api[_-]?key|secret|token|password" -- . ':!docs/superpowers' ':!site/package-lock.json' || echo CLEAN
```

Also scan history: `git log --all -p -- .env* */secrets* 2>/dev/null | head` → expect empty.
Expected: CLEAN (matches only in innocuous contexts, reviewed by eye).

- [ ] **Step 3: License inventory sanity**

Confirm: tracked content = own code (choose MIT — add `LICENSE` file if absent), MediaPipe assets (Apache-2.0, redistribution OK), FFHQ-Makeup samples + PCA prior derived from them (CC BY-NC-SA, attributed in README + site + Space card), seed looks (derived from FFHQ-Makeup, same license note). `third_party/` is NOT tracked → no vendored-license blockers. If `LICENSE` is missing, create MIT with "Edward Lin" and note the data-license exception ("sample images and derived looks: CC BY-NC-SA, see README").

- [ ] **Step 4: Full suite + site build, one last time**

```bash
.venv/Scripts/python.exe -m pytest tests/ -q
cd site && npm run build
```

Expected: all green, build clean.

- [ ] **Step 5: Commit any checklist fixes + final commit**

```bash
git add -A && git commit -m "Publish checklist: license file + final verification"
```

- [ ] **Step 6: Report to user** — everything below requires their accounts and stays manual (per HOSTING.md runbook): flip repo public, create the HF Space and push `space_build/`, import to Vercel (root dir `site/`), add the DNS CNAME, then update `site/lib/links.ts` slugs if they differ and push.

---

## Self-review (done at writing time)

- **Spec coverage:** architecture pieces 1–4 → Tasks 4/5 (site), 3 (Space), 6 (DNS documented; actual record is a user action); site content sections 1–6 → Task 5 step 2; hosting plan doc → Task 6; publish checklist → Tasks 1 & 7; error handling → static site (build-time) + existing gr.Error + Task 2 gate; testing/verification 1–5 → Task 4 step 2, Task 5 steps 3–4, Task 7 (items needing live accounts land in the runbook as user actions).
- **Spec corrections found during fact-gathering (recorded in Task 6 step 3):** (a) BiSeNet is used only by the GPU removal path — Space excludes it, contra spec §3; (b) weights/data were never git-tracked — publish checklist's "SSAT ~42 MB, BiSeNet ~53 MB in repo" premise is moot; Task 1 tracks only the small CPU assets.
- **Type consistency:** `_single_photo_available` name matches between Task 2 impl and tests; `stage(dest: Path)` matches test import; `HF_SPACE_URL`/`HF_SPACE_PAGE`/`GITHUB_URL` defined Task 4, consumed Task 5.
- **Placeholder scan:** Task 5's page JSX/CSS is intentionally specified as content requirements + design-skill invocation rather than inline JSX (a full page dump would lock in design before the frontend-design pass); all testable behavior around it is concrete.
