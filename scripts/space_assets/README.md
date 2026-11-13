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
short_description: Borrow the makeup from any photo and try it on your own face.
---

# AI Makeup Transfer — live demo

Pick a photo of someone wearing makeup you like, add a photo of yourself, and see the look on
your own face. The makeup is rendered by a reference-based transfer network (CSD-MT, CVPR 2024)
and composited back at full resolution, so your skin texture, lighting and complexion stay yours.

**On this CPU Space** everything works: *Try on*, *Create a look* (one photo), *My looks*.
A transfer takes a few seconds.

Sample FFHQ-Makeup images © their authors, **CC BY-NC-SA** —
[cyberagent/FFHQ-Makeup](https://huggingface.co/datasets/cyberagent/FFHQ-Makeup).
CSD-MT model © its authors, CC BY-NC-SA 4.0 — [Snowfallingplum/CSD-MT](https://github.com/Snowfallingplum/CSD-MT).
Code: [GitHub](https://github.com/Edward9292/ai-makeup-transfer) · Website: https://makeup-transfer.edwardlin.dev
