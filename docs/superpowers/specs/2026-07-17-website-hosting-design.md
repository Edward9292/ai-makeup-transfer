# Design: Project Website + Hosting (makeup-transfer.edwardlin.dev)

**Date:** 2026-07-17
**Status:** approved by user (conversation, 2026-07-17)

## Goal

Turn the ai-makeup-transfer project into a public-facing website: a polished showcase site
with an embedded live demo, at **makeup-transfer.edwardlin.dev**, with the repo public on
GitHub and the site hosted on Vercel. A written hosting plan becomes part of the project docs.

**Launch scope (decided):** showcase + hosted demo. NOT a full product web app — no custom
upload UI, no owned inference API, no auth/storage backend. Those remain a documented upgrade
path only.

## Architecture

Three free pieces, one repo:

1. **GitHub — public monorepo (this repo).**
   - The website lives in a new `site/` directory in this same repo (decision: monorepo over
     a separate site repo — one source of truth; the docs/results the site showcases live
     next to it; Vercel builds from a subdirectory natively).
   - Pushing to `master` auto-deploys the site via Vercel's GitHub integration.

2. **Vercel — frontend, free Hobby tier.**
   - **Next.js** (App Router), project Root Directory set to `site/`.
   - Rationale: Vercel's zero-config default; leaves the door open for the future
     "full web app with own API" phase. The launch site is fully static pages — no server
     code, no API routes.
   - Custom domain `makeup-transfer.edwardlin.dev` added to the Vercel project.

3. **Hugging Face Space — live demo, free CPU Basic tier.**
   - The existing Gradio `src/makeup_transfer/app.py` deployed as a public HF Space,
     embedded (iframe) and linked from the site's demo section.
   - CPU-capable tabs (extract-from-pair, apply, look library) work as-is.
   - The single-photo extraction tab requires CUDA (SSAT). On the CPU Space it stays
     visible but disabled, with an honest "needs a GPU — run locally" note. (The tab
     already fails loudly via `gr.Error` on CPU-only machines; the Space build makes the
     limitation visible up front instead of on click.)
   - Space gets its own copy of app code + weights. SSAT weights are **excluded** (only
     the disabled GPU tab uses them); BiSeNet + MediaPipe assets are included (the live
     tabs need them). Weights go via HF LFS, which handles these sizes fine.

4. **DNS.**
   - CNAME record: `makeup-transfer` → `cname.vercel-dns.com` at the edwardlin.dev DNS
     provider; Vercel provisions TLS automatically.

## Site content (showcase)

Single-page structure with anchored sections, in order:

1. **Hero** — project name + the one-line design:
   "extract makeup as a translucent RGBA layer in 3D UV space; re-apply it
   skin-tone-aware to any face."
2. **How it works** — pipeline diagram (preprocess → UV unwrap → extract → apply),
   adapted from BUILD_PLAN's ASCII pipeline into a real graphic.
3. **Results gallery** — Phase 2 verification grids (3 looks × 4 targets), cross-identity
   before/after examples, single-photo extraction results.
4. **Live demo** — embedded HF Space iframe + direct link, with a note on what runs on
   CPU vs what needs the repo locally.
5. **Engineering notes** — honest build log highlights, **including the failed Phase-4
   prior gate** (negative results presented plainly; links to BUILD_PLAN for the numbers).
6. **Attributions + links** — GitHub repo, papers/repos used (makeup-extract, makeup-priors,
   SSAT, BiSeNet, MediaPipe), dataset credits (FFHQ-Makeup, MT dataset).

Design intent: portfolio-quality, distinctive (frontend-design skill applies at
implementation time), fast/static, no client-side heaviness beyond the demo iframe.

## Hosting plan (to be written into project docs)

New `docs/HOSTING.md`, plus README + BUILD_PLAN goal-section updates pointing at it:

- **Current setup:** Vercel Hobby ($0) + HF Space CPU Basic ($0) + GitHub public ($0).
  Total: **$0/month**. DNS config recorded (CNAME → Vercel).
- **Limits at this tier:** no GPU inference online (single-photo tab disabled on the
  Space); HF CPU Spaces sleep after inactivity (cold-start delay); Vercel Hobby is
  non-commercial use.
- **Upgrade path (documented, not built):**
  1. HF PRO / ZeroGPU Space → enables the GPU single-photo tab online.
  2. Modal / Replicate / RunPod serverless GPU API + custom upload/apply UI in the
     Next.js app → the "full web app" phase (per-inference cost ~$0.001–0.01/run,
     cold starts apply).
  3. Real storage backend replacing the `storage.py` local-file fake (already designed
     as a swappable interface).

## Publish checklist (repo goes public)

Must pass before the GitHub repo is made public:

- **File sizes:** no file ≥ 100 MB (GitHub hard limit). Known large files: SSAT ~42 MB,
  BiSeNet ~53 MB — fine. Verify nothing else (data, preprocess_out) exceeds limits;
  prune or LFS anything that does.
- **Licenses of vendored `third_party/`:** confirm each vendored repo's license permits
  redistribution (face-parsing.PyTorch is MIT; check SSAT, 3DDFA_V2, makeup-extract, CPM);
  keep their LICENSE files intact; note anything that must be removed before publishing.
- **Dataset licensing:** FFHQ-Makeup is CC BY-NC-SA — redistribution of the sample images
  in the repo and on the site requires attribution and non-commercial use (a portfolio
  site qualifies). MT-dataset samples: attribute; keep usage research/portfolio-scoped.
- **No secrets/personal data** in history (repo has none known; verify before flip).

## Error handling

- Site is static: failures are build-time (Vercel build fails → deploy blocked, site
  stays on last good version).
- Demo Space: pipeline errors already surface as `gr.Error` with actionable messages;
  Space sleeping is acceptable at this tier and noted on the site.
- DNS/TLS: handled by Vercel; misconfiguration is visible in the Vercel dashboard.

## Testing / verification

1. `next build` passes locally and on Vercel; site renders at the Vercel preview URL.
2. Custom domain resolves: https://makeup-transfer.edwardlin.dev serves the site with TLS.
3. HF Space builds and the CPU tabs work end-to-end in a browser (extract from the
   bundled sample pair → save → apply to a sample target → download).
4. Embedded iframe loads the Space from the site.
5. Publish checklist above fully checked off before the repo is flipped public.

## Out of scope (YAGNI, explicitly)

- Custom inference API, job queues, auth, user storage, sharing service.
- Realtime/AR, video.
- Paid tiers of anything at launch.
