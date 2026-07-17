---
title: AI Makeup Transfer
emoji: 💄
colorFrom: pink
colorTo: purple
sdk: gradio
sdk_version: "6.19.0"
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
