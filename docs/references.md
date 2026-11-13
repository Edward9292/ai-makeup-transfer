# References — Papers & Code to Implement

**2026-11-13 update:** the shipped engine is now **CSD-MT** (below). The sections after it are
the original curation for the "makeup as a translucent RGBA layer, in 3D UV space" design,
kept for the legacy pipeline's history.

## ⭐ SHIPPED — reference-based transfer engine

### CSD-MT — Content-Style Decoupling for Unsupervised Makeup Transfer (CVPR 2024)
- **Why it's the engine:** takes `(source face, reference face)` directly — no same-person pair,
  no pseudo ground truth — and is tiny (ngf=16 generator, ~28 MB) so it runs on **CPU in ~1 s** at
  256 px, which is what the free Space has. Content/style decoupling keeps the target's identity;
  cross-attention over face-parsing semantics handles pose/expression differences.
- Authors: Zhaoyang Sun, Shengwu Xiong, Yaxiong Chen, Yi Rong
- Paper: https://arxiv.org/abs/2405.17240
- **GitHub:** https://github.com/Snowfallingplum/CSD-MT — weights: Google Drive folder linked in
  `quick_start/` (`CSD_MT.pth`, `79999_iter.pth`); fetched by `makeup_transfer.fetch_weights`.
- License: **CC BY-NC-SA 4.0** (non-commercial).
- **How we use it (`engine.py`):** dlib crop replaced by MediaPipe landmarks with the same
  margins; spectacles mapped to *background* in the semantic encoding so a reference's frames
  are never sampled as eye shadow; output applied as a **delta** onto the full-res photo, gated
  by BiSeNet regions + the segmenter skin mask; complexion (uniform L / colour cast on skin)
  damped in Lab so the target keeps its own skin tone.

### Stable-Makeup — diffusion engine (SIGGRAPH 2025) — SHIPPED as `--backend diffusion`
- **Why:** the GAN edits pixels; this *generates* the made-up face, so makeup reads as pigment
  on skin (gloss, soft blush edges, blended shadow) instead of a tint. SD 1.5-sized, so it fits
  a 16 GB Apple-silicon laptop in fp16.
- Authors: Yuxuan Zhang et al. · Paper: https://arxiv.org/abs/2403.07764
- **GitHub:** https://github.com/Xiaojiu-z/Stable-Makeup (Apache-2.0) — weights: Google Drive
  folder in its README (`pytorch_model{,_1,_2}.bin`), fetched by `fetch_weights --diffusion`.
- Architecture: SD 1.5 UNet (frozen) + a **content encoder** and a **structure encoder** (both
  ControlNets, fed the target face and a 68-landmark contour drawing, with the empty-prompt
  text embedding) + a **detail-preserving makeup encoder** (fine-tuned CLIP ViT-L, 12 hidden
  layers x 257 tokens -> one resampler block -> the UNet cross-attention through its own
  key/value projections). DDIM, 30 steps, CFG 1.6 over the makeup tokens.
- **How we use it (`diffusion.py`):** ported onto current `diffusers` (upstream bundles a 2023
  fork); SPIGA + facelib landmark stack replaced by MediaPipe -> 68-point mapping, drawn with
  upstream's exact matplotlib recipe; FFHQ-style *portrait* crop instead of the GAN's tight
  crop; the 512 px output goes through the same delta composite / region gate / occlusion mask
  as CSD-MT, which also keeps the wearer's own eyes, hair and background.

### Heavier tiers evaluated (not shipped — documented upgrade path)
- **FLUX-Makeup** (2025, Apache-2.0 weights at `qihoo360/FLUX-Makeup`): DiT on FLUX.1-Kontext-dev
  (gated, non-commercial, 12B → 24 GB+ VRAM). Best reported fidelity; cloud-GPU only.
  https://github.com/360CVGroup/FLUX-Makeup
- **SHMT** (NeurIPS 2024, CC BY-NC-SA): latent diffusion, needs user-supplied seg + 3D maps and
  the LDM VQ-f4 autoencoder; 50 DDIM steps. https://github.com/Snowfallingplum/SHMT
- **EleGANt** (ECCV 2022, CC BY-NC-SA): GAN alternative to CSD-MT, needs dlib; CUDA-only demo.
- **General image editors via API** (Gemini image, FLUX Kontext, Qwen-Image-Edit): multi-image
  edit prompts can do "apply the makeup from image 2 to image 1" well, at ~$0.03/image and with
  no local weights — a reasonable paid "HQ" mode later.

## ⭐ PRIMARY — extract makeup as a UV layer + alpha matte (our core idea)

### Makeup Extraction of 3D Representation via Illumination-Aware Image Decomposition
- **Why it matters:** Extracts **bare skin + makeup + alpha matte** in **UV space** from a makeup
  portrait — literally "makeup as a transparent layer." This is the closest published work to
  what we want and the main blueprint for the Extraction NN.
- Venue: Eurographics 2023 (Computer Graphics Forum)
- Authors: Xingchao Yang, Takafumi Taketomi, Yoshihiro Kanamori
- Paper (arXiv): https://arxiv.org/abs/2302.13279
- PDF: https://arxiv.org/pdf/2302.13279
- Project page: https://yangxingchao.github.io/makeup-extract-page/
- **GitHub:** https://github.com/YangXingchao/makeup-extract
- Journal: https://onlinelibrary.wiley.com/doi/10.1111/cgf.14762

### Makeup Prior Models for 3D Facial Makeup Estimation and Applications
- **Why it matters:** Same authors as makeup-extract, one year later. Instead of per-image
  optimization / raw differencing, they **fit a learned makeup prior** (PCA-based or
  StyleGAN2-based, in UV space) — regularizes away shadows/noise, much faster, more robust.
  **This is the primary Phase 1 blueprint** (makeup-extract is the concept; this is the method).
- Venue: CVPR 2024
- Authors: Xingchao Yang, Takafumi Taketomi, Yuki Endo, Yoshihiro Kanamori
- Paper (arXiv): https://arxiv.org/abs/2403.17761
- Project page: https://yangxingchao.github.io/makeup-priors-page/
- **GitHub:** https://github.com/YangXingchao/makeup-priors

### CPM — Color-Pattern Makeup Transfer ("Lipstick ain't enough")
- **Why it matters:** Separates makeup into independent **color** and **pattern** branches
  (pattern = glitter/drawn designs as an overlay). Pretrained weights + datasets. Good init for
  the color/pattern parts of extraction.
- Venue: CVPR 2021
- Authors: Thao Nguyen, Anh Tran, Minh Hoai (VinAI Research)
- Paper (CVPR open access): https://openaccess.thecvf.com/content/CVPR2021/papers/Nguyen_Lipstick_Aint_Enough_Beyond_Color_Matching_for_In-the-Wild_Makeup_Transfer_CVPR_2021_paper.pdf
- Project page: https://thaoshibe.github.io/CPM
- **GitHub:** https://github.com/VinAIResearch/CPM

### SOGAN — 3D-Aware Shadow and Occlusion Robust GAN for Makeup Transfer
- **Why it matters:** Does makeup transfer in **3D/UV space**, robust to pose, shadow, occlusion.
  Reference for the UV-warp + application design.
- Venue: ACM MM 2021
- Paper (arXiv): https://arxiv.org/abs/2104.10567
- PDF: https://arxiv.org/pdf/2104.10567

## REUSE — pretrained building blocks (no training needed)

### 3DDFA_V2 — fast, stable 3D dense face alignment (3DMM fit + UV)
- **Use for:** fitting a 3D face model and producing the canonical UV texture.
- Venue: ECCV 2020
- **GitHub:** https://github.com/cleardusk/3DDFA_V2

### DECA — Detailed Expression Capture and Animation (alternative 3D fit + UV)
- **Use for:** alternative 3DMM/UV pipeline; richer detail. Evaluate vs 3DDFA_V2 in Phase 0.
- Venue: SIGGRAPH 2021
- **GitHub:** https://github.com/yfeng95/DECA

### BiSeNet face parsing — per-region segmentation (lips, eyes, brows, skin…)
- **Use for:** masking makeup regions; clipping the extracted layer to valid areas.
- **GitHub:** https://github.com/zllrunning/face-parsing.PyTorch

### MediaPipe Face Mesh — 468-point facial landmarks
- **Use for:** alignment, warping, lightweight preprocessing.
- Docs: https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker

### FreeUV — Ground-Truth-Free Realistic Facial UV Texture Recovery
- **Use for:** upgrading `uv_unwrap.py` if UV quality becomes the bottleneck — high-fidelity UV
  texture recovery that handles makeup, facial hair, occlusion. Same group as makeup-extract.
- Venue: CVPR 2025
- PDF: https://openaccess.thecvf.com/content/CVPR2025/papers/Yang_FreeUV_Ground-Truth-Free_Realistic_Facial_UV_Texture_Recovery_via_Cross-Assembly_Inference_CVPR_2025_paper.pdf

## REFERENCE / OPTIONAL — comparison or future phases

### SSAT — Symmetric Semantic-Aware Transformer (transfer + REMOVAL)
- **Use for:** **single-image extraction (future phase)** — its makeup-removal estimates the bare
  face when we only have a with-makeup photo (no pair).
- Venue: AAAI 2022
- Paper (arXiv): https://arxiv.org/abs/2112.03631 · PDF: https://arxiv.org/pdf/2112.03631
- **GitHub:** https://github.com/Snowfallingplum/SSAT
- ncnn mobile infer port: https://github.com/FeiGeChuanShu/ncnn_MakeupTransfer
- **Practical notes (Phase 2.5, adopted 2026-07-03):**
  - **Weights ship in-repo** at `quick_start/weights/SSAT.pth` (~42 MB) — pure-PyTorch inference,
    no build. (The README's Baidu link is for the **root-repo** SSAT_D variant only; the
    `quick_start` weights are all Phase 2.5 needs.)
  - Needs **face-parsing.PyTorch** (BiSeNet, MIT) label maps as semantic input: **18 classes**;
    map hat (`label ≥ 18`) → background.
  - **Successor repos checked 2026-07-03:** SSAT++ and CSD-MT (same author) are **Baidu-weights-
    only** and **transfer-only** (no usable removal branch) — not useful here.

### EleGANt — Exquisite and Locally Editable GAN for Makeup Transfer
- **Use for:** optional high-realism neural renderer / quality baseline; supports local editing.
- Venue: ECCV 2022
- Paper (arXiv): https://arxiv.org/abs/2207.09840 · PDF: https://arxiv.org/pdf/2207.09840
- **GitHub:** https://github.com/Chenyu-Yang-2000/EleGANt

### SHMT — Self-supervised Hierarchical Makeup Transfer via Latent Diffusion
- **Use for:** quality baseline for diffusion-era transfer; self-supervised (no pseudo-pair
  labels). Reported to preserve identity/detail better than Stable-Makeup. End-to-end though —
  no portable layer — so baseline/refiner inspiration only, not a pipeline component.
- Venue: NeurIPS 2024
- Paper (arXiv): https://arxiv.org/html/2412.11058v1

### Stable-Makeup — diffusion-based real-world makeup transfer
- **Use for:** optional high-realism renderer for complex/artistic looks (heavy, GPU).
  Caveat: comparisons (e.g. in SHMT/FLUX-Makeup papers) report identity/color shift and loss of
  high-frequency makeup detail — evaluate before adopting.
- Venue: SIGGRAPH 2025
- Paper (arXiv): https://arxiv.org/abs/2403.07764 · HTML: https://arxiv.org/html/2403.07764v2
- Project page: https://xiaojiu-z.github.io/Stable-Makeup.github.io/
- **GitHub:** https://github.com/Xiaojiu-z/Stable-Makeup

### FLUX-Makeup / DreamMakeup / BeautyDiffusion — newer diffusion transfer (2025)
- **Use for:** tracking the diffusion SOTA; same caveat as above (end-to-end, no layer).
- FLUX-Makeup (DiT-based): https://arxiv.org/html/2508.05069v1
- DreamMakeup (latent-diffusion customization): https://arxiv.org/html/2510.10918v1
- BeautyDiffusion (generative latent decomposition): https://www.sciencedirect.com/science/article/abs/pii/S1566253525003148

### BeautyGAN / PSGAN — classic baselines
- BeautyGAN (ACM MM 2018): http://liusi-group.com/projects/BeautyGAN
- BeautyGAN paper: https://www.researchgate.net/publication/328375007
- PSGAN (CVPR 2020): https://github.com/wtjiang98/PSGAN

## Datasets (for extraction training/eval + fine-tuning the harmonization refiner)
- **FFHQ-Makeup (2025)** — ⭐ **synthetic same-identity with/without-makeup pairs**, multiple
  styles, identity-consistent. Directly unblocks Phase 1's "needs real same-person pairs"
  dependency. Paper: https://arxiv.org/pdf/2508.03241
- **MT (Makeup Transfer) dataset** — standard paired makeup/no-makeup set (via BeautyGAN project page).
  NOTE: pairs are different people (style pairs), not same-person pairs.
- **CPM-Real / CPM-Synt-1 / CPM-Synt-2 / Stickers** — from the CPM repo (`about-data.md`).
- **BeautyFace** — in-the-wild makeup evaluation set.

## Commercial APIs (build-vs-buy context only — none expose a portable UV/PNG layer)
- Perfect Corp / YouCam AI Beauty API: https://yce.perfectcorp.com/ai-api
- Banuba Makeup SDK: https://www.banuba.com/blog/best-makeup-virtual-try-on-sdks-compared-and-tested
- ZEGOCLOUD AI Effects, Orbo BeautyGPT, TRYONNIX — similar black-box try-on services.
