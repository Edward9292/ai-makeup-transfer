# Hosting plan — makeup-transfer.edwardlin.dev

**Status:** built and verified locally (2026-07-17); the final deploy clicks need the
owner's accounts — see the runbook below. Design doc:
`docs/superpowers/specs/2026-07-17-website-hosting-design.md`.

## Current setup ($0/month)

| Piece | Service | Tier | Cost | What it serves |
|---|---|---|---|---|
| Showcase site | Vercel | Hobby | $0 | `site/` — static Next.js export, custom domain |
| Live demo | Hugging Face Spaces | CPU Basic | $0 | Gradio app staged by `scripts/build_space.py` |
| Source + assets | GitHub | public repo | $0 | this monorepo (site builds from `site/`) |
| DNS | edwardlin.dev provider | — | $0 | `makeup-transfer` CNAME → `cname.vercel-dns.com` |

TLS is provisioned automatically by Vercel once the domain is added to the project.

## Limits at this tier (accepted, stated on the site)

- **CPU only, and that is enough.** The reference-based engine (CSD-MT + BiSeNet, ~80 MB,
  CPU torch wheel) runs every tab on the free Space at 1–4 s per transfer. The Space build
  copies the two `.pth` files from `weights/`, so run `python -m makeup_transfer.fetch_weights`
  before `scripts/build_space.py`; they go up through HF's normal git push (<100 MB each).
- **Space sleeps** after ~48h of inactivity; first visit after that cold-starts (up to a
  minute). Noted next to the demo embed.
- **Vercel Hobby is non-commercial** use only — fine for a portfolio/research showcase.

## Deploy runbook (manual, owner's accounts)

1. **GitHub — make the repo public.** Re-run the publish checklist first (see
   `docs/superpowers/plans/2026-07-17-website-hosting.md` Task 7): no tracked file ≥ 100 MB,
   no secrets in history, licenses/attribution in README. Then Settings → General →
   Change visibility → Public.
2. **Hugging Face — create the Space.**
   ```bash
   python scripts/build_space.py            # stages space_build/
   # create Space "makeup-transfer" (SDK: Gradio, hardware: CPU Basic, public), then:
   cd space_build
   git init && git remote add origin https://huggingface.co/spaces/<user>/makeup-transfer
   git add -A && git commit -m "demo" && git push origin main --force
   ```
   Verify in a browser: create a look from a bundled sample photo → save → try it on a
   sample face → download; a transfer should take a few seconds on CPU Basic.
3. **Vercel — import the repo.** New Project → import the GitHub repo →
   **Root Directory = `site/`** (framework auto-detects Next.js) → Deploy. Check the
   `*.vercel.app` preview renders.
4. **DNS.** At the edwardlin.dev DNS provider add `makeup-transfer` CNAME →
   `cname.vercel-dns.com`. In Vercel: Project → Settings → Domains → add
   `makeup-transfer.edwardlin.dev`; wait for TLS to go green.
5. **Post-launch link fixup.** If the actual HF username / repo slug differ from the
   defaults, update `site/lib/links.ts` (single source for the Space/repo URLs) and push —
   Vercel redeploys automatically. Verify the iframe loads the Space from the live site.

## Upgrade path (documented, deliberately not built)

1. **HF PRO / ZeroGPU Space** → room for a diffusion-quality engine (Stable-Makeup on SD1.5,
   or FLUX-Makeup on FLUX Kontext) behind the same `engine.py` crop → delta path, as an
   opt-in "HQ" mode next to the instant CPU result. See `docs/references.md`.
2. **Serverless GPU API** (Modal / Replicate / RunPod) + a custom upload/apply UI in the
   Next.js app → the "full web app" phase. Ballpark ~$0.001–0.01 per inference, cold starts
   apply. The site is already Next.js on Vercel so API routes can be added incrementally.
3. **Real storage backend** replacing the `storage.py` local-folder fake — it was designed
   as a swappable interface precisely for this step (looks → object store + DB).

## Failure modes

- Site: build-time only — a failed Vercel build blocks the deploy, last good version stays up.
- Demo: pipeline errors surface as `gr.Error` with actionable messages; Space sleeping is
  expected behavior at this tier.
- DNS/TLS: managed by Vercel; misconfiguration is visible in the Vercel dashboard's Domains tab.
