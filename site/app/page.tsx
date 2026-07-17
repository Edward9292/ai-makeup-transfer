import Pipeline from "@/components/Pipeline";
import { GITHUB_URL, HF_SPACE_PAGE, HF_SPACE_URL } from "@/lib/links";

function Gate({ verdict }: { verdict: "passed" | "failed" | "partial" }) {
  const text = { passed: "GATE · PASSED", failed: "GATE · FAILED", partial: "GATE · 2 OF 3" }[verdict];
  return <span className={`gate gate-${verdict}`}>{text}</span>;
}

export default function Home() {
  return (
    <main>
      <nav className="topnav" aria-label="Sections">
        <span className="topnav-brand">makeup-transfer</span>
        <div className="topnav-links">
          <a href="#how">Method</a>
          <a href="#results">Results</a>
          <a href="#demo">Demo</a>
          <a href="#notes">Notes</a>
          <a href={GITHUB_URL} target="_blank" rel="noreferrer">GitHub ↗</a>
        </div>
      </nav>

      {/* ---------------- hero ---------------- */}
      <section id="hero" className="hero">
        <p className="eyebrow">AI MAKEUP TRANSFER — RESEARCH BUILD, EDWARD LIN</p>
        <h1>
          A makeup look<br />is a <em>file</em>.
        </h1>
        <p className="hero-sub">
          Extract makeup from a photo as a translucent <strong>RGBA layer in 3D UV space</strong>;
          re-apply it skin-tone-aware to any face — the skin underneath shows through, so the
          same look adapts to every person who wears it.
        </p>
        <div className="hero-ctas">
          <a className="btn btn-primary" href="#demo">Try the live demo</a>
          <a className="btn" href={GITHUB_URL} target="_blank" rel="noreferrer">Read the code ↗</a>
        </div>

        <figure className="specimen">
          <div className="specimen-strip">
            <div className="specimen-cell">
              <img src="/results/002386_makeup_03.jpg" alt="Source photo: a face wearing red lipstick, brow fill and eye shadow" />
              <figcaption>1 · the look, on its owner</figcaption>
            </div>
            <div className="specimen-arrow" aria-hidden="true">→</div>
            <div className="specimen-cell">
              <img src="/results/ffhq-002386-03_layer_preview.png" alt="The extracted makeup as a translucent RGBA layer in UV space, shown over a checkerboard" />
              <figcaption>2 · extracted: layer.png, UV space</figcaption>
            </div>
            <div className="specimen-arrow" aria-hidden="true">→</div>
            <div className="specimen-cell">
              <div className="crop-right">
                <img src="/results/ffhq-002386-03_side_by_side.png" alt="The same look re-applied to a different person's face" />
              </div>
              <figcaption>3 · re-applied to someone else</figcaption>
            </div>
          </div>
          <figcaption className="specimen-note">
            Real pipeline output, not a mockup — FFHQ-Makeup identity 002386, style 03,
            re-applied cross-identity. The checkerboard is the point: the middle image is
            a genuinely transparent PNG.
          </figcaption>
        </figure>
      </section>

      {/* ---------------- how it works ---------------- */}
      <section id="how" className="band">
        <p className="eyebrow">METHOD</p>
        <h2>Two photos in, one portable layer out</h2>
        <p className="lede">
          Both photos are unwrapped into a canonical <strong>UV texture space</strong> (pose- and
          face-shape-invariant), where makeup becomes a per-texel difference. Chromaticity
          differencing with Retinex normalization keeps lighting out of the layer; UV symmetry
          suppresses shadows. Applying is the reverse trip: warp the layer to the target&apos;s UV,
          blend per region — lipstick absorbs (multiply), powder scatters (soft-light) — then
          composite only the <em>delta</em> back onto the untouched photo, so skin grain and lip
          highlights survive.
        </p>
        <Pipeline />
        <p className="method-foot">
          No makeup GAN is trained. Pretrained pieces (MediaPipe landmarks, SSAT removal for
          single-photo mode) plus classical numerics do the work; every stage is independently
          runnable and inspectable.
        </p>
      </section>

      {/* ---------------- results ---------------- */}
      <section id="results" className="band">
        <p className="eyebrow">RESULTS <Gate verdict="passed" /></p>
        <h2>Three looks × four faces</h2>
        <p className="lede">
          The Phase-2 verification matrix: each extracted look applied to every test identity —
          pale, freckled, olive, male. Identity is preserved, teeth stay clean on open smiles,
          and the same lipstick reads differently on each face because each face keeps its own
          luminance base.
        </p>
        <figure className="frame">
          <img src="/results/verification_grid.png" alt="Grid of three extracted looks applied to four different faces" loading="lazy" />
          <figcaption>3 looks × 4 targets, cross-identity. Left column: the source of each look.</figcaption>
        </figure>

        <div className="duo">
          <figure className="frame">
            <img src="/results/ffhq-002386-03_side_by_side.png" alt="Before and after: red lipstick look transferred" loading="lazy" />
            <figcaption>Pair extraction → cross-applied. Lipstick, brow fill, smoky eye.</figcaption>
          </figure>
          <figure className="frame">
            <img src="/results/ffhq-003125-02_side_by_side.png" alt="Before and after: a second look transferred to another identity" loading="lazy" />
            <figcaption>Second held-out identity, same pipeline, no tuning.</figcaption>
          </figure>
        </div>

        <h3>One photo is enough <span className="muted">(usually)</span> <Gate verdict="partial" /></h3>
        <p className="lede">
          Single-photo mode synthesizes the bare face with SSAT makeup removal, then runs the
          identical pair pipeline. On held-out identities, 2 of 3 match the ground-truth-pair
          extraction; the third (male face, glasses) fails via skin-tone drift — documented
          below, not hidden.
        </p>
        <figure className="frame">
          <img src="/results/spike_grid.png" alt="Comparison grid: pseudo-pair extraction versus ground-truth-pair extraction on three identities" loading="lazy" />
          <figcaption>Pseudo-pair vs ground-truth pair, three held-out identities.</figcaption>
        </figure>
      </section>

      {/* ---------------- demo ---------------- */}
      <section id="demo" className="band band-dark">
        <p className="eyebrow">LIVE DEMO</p>
        <h2>Steal a look, right here</h2>
        <p className="lede">
          Extract a look from a photo pair, apply any saved look to a new face with per-region
          intensity sliders, and browse the look library — running free on a Hugging Face CPU
          Space. Two honest caveats: the Space sleeps when idle (first load can take a minute),
          and the <em>single-photo</em> tab needs a CUDA GPU, so it&apos;s disabled online — clone the
          repo to run it locally.
        </p>
        <div className="demo-frame">
          <iframe
            src={HF_SPACE_URL}
            title="AI Makeup Transfer — live Gradio demo"
            loading="lazy"
            allow="clipboard-write"
          />
        </div>
        <p className="demo-fallback">
          Iframe not loading? Open the Space directly:{" "}
          <a href={HF_SPACE_PAGE} target="_blank" rel="noreferrer">huggingface.co/spaces — makeup-transfer ↗</a>
        </p>
      </section>

      {/* ---------------- engineering notes ---------------- */}
      <section id="notes" className="band">
        <p className="eyebrow">ENGINEERING NOTES</p>
        <h2>The build log, including the parts that failed</h2>
        <p className="lede">
          This project ran on explicit go/no-go gates. Most passed. One didn&apos;t — and shipping
          that result honestly is part of the work.
        </p>

        <ol className="log">
          <li>
            <div className="log-head">
              <span className="log-phase">PHASE 0</span>
              <Gate verdict="passed" />
            </div>
            <h3>The layer concept survives contact with reality</h3>
            <p>
              A hand-derived layer from one with/without pair re-applied convincingly to a third
              face — red lipstick onto much paler skin, lip texture and highlights showing
              through. Bonus finding: MediaPipe&apos;s canonical mesh + per-triangle barycentric
              warping gave a workable UV unwrap with <em>zero</em> third-party model code, after the
              sandbox blocked 3DDFA_V2&apos;s native build.
            </p>
          </li>
          <li>
            <div className="log-head">
              <span className="log-phase">PHASE 1–2</span>
              <Gate verdict="passed" />
            </div>
            <h3>Chromaticity beats shading; deltas beat pastes</h3>
            <p>
              The #1 risk was alpha picking up shadows instead of makeup. Differencing in Lab
              chromaticity with Retinex-normalized luminance, plus UV-symmetry suppression, kept
              the mattes clean on held-out pairs. On the apply side, compositing only the makeup
              <em> delta</em> back onto the untouched photo cancelled double-resample blur — the
              planned neural refiner had nothing left to fix and was dropped. Best bug: a final
              alpha blur bled lip color into the mouth-interior sliver, tinting teeth plum on
              open-mouthed targets. Hard exclusions are now re-asserted after every blur.
            </p>
          </li>
          <li>
            <div className="log-head">
              <span className="log-phase">PHASE 2.5</span>
              <Gate verdict="partial" />
            </div>
            <h3>Single-photo mode works — with a documented failure mode</h3>
            <p>
              SSAT removal makes the pair requirement optional. But on a male face with glasses,
              SSAT (trained on a mostly-female dataset) drifts the whole face&apos;s skin tone, and
              the pipeline reads that drift as foundation: false-alpha 0.17, ~8× the passing
              identities. Called a pass for the concept, logged as a limitation, not averaged away.
            </p>
          </li>
          <li>
            <div className="log-head">
              <span className="log-phase">PHASE 4</span>
              <Gate verdict="failed" />
            </div>
            <h3>The PCA makeup prior missed its gate — so it ships off by default</h3>
            <p>
              A 64-component PCA prior (2,000 FFHQ-Makeup pairs, robust IRLS fit against an
              explicit nuisance basis) was built to reject that skin-tone drift. It helped —
              false-alpha 0.1734 → 0.1122, −35% — but the gate demanded ≤ 0.04, and one clean
              identity mildly regressed. The residual re-add knob made things monotonically
              worse at every setting. Verdict: a stronger prior is not the lever; the drift is an
              SSAT-side defect. The prior remains in the codebase behind <code>--prior</code>,
              and the baseline extractor is byte-for-byte unchanged by default.
            </p>
            <figure className="frame">
              <img src="/results/prior_grid.png" alt="Evaluation grid for the PCA prior across residual weights and identities" loading="lazy" />
              <figcaption>The failed gate&apos;s evidence: prior vs baseline across residual weights.</figcaption>
            </figure>
            <p className="log-link">
              Full numbers live in{" "}
              <a href={`${GITHUB_URL}/blob/master/docs/BUILD_PLAN.md`} target="_blank" rel="noreferrer">
                docs/BUILD_PLAN.md ↗
              </a>{" "}
              — the metric tables are committed alongside the code that produced them.
            </p>
          </li>
        </ol>
      </section>

      {/* ---------------- credits ---------------- */}
      <section id="credits" className="band band-credits">
        <p className="eyebrow">ATTRIBUTIONS</p>
        <h2>Built on open research</h2>
        <div className="credit-cols">
          <div>
            <h3>Papers &amp; code</h3>
            <ul>
              <li><a href="https://github.com/kyotovision-public/makeup-extract" target="_blank" rel="noreferrer">makeup-extract</a> — bare skin + makeup + alpha in UV (the concept)</li>
              <li><a href="https://yangxueyan.github.io/publication/makeup-prior/" target="_blank" rel="noreferrer">Makeup Priors</a> (CVPR 2024) — learned makeup priors in UV space</li>
              <li><a href="https://github.com/Snowfallingplum/SSAT" target="_blank" rel="noreferrer">SSAT</a> (AAAI 2022) — makeup transfer &amp; removal</li>
              <li><a href="https://github.com/zllrunning/face-parsing.PyTorch" target="_blank" rel="noreferrer">face-parsing.PyTorch</a> — BiSeNet face parsing (MIT)</li>
              <li><a href="https://developers.google.com/mediapipe" target="_blank" rel="noreferrer">MediaPipe</a> — 468-point face landmarks + canonical mesh</li>
            </ul>
          </div>
          <div>
            <h3>Data</h3>
            <ul>
              <li>
                <a href="https://huggingface.co/datasets/cyberagent/FFHQ-Makeup" target="_blank" rel="noreferrer">FFHQ-Makeup</a>{" "}
                (CyberAgent) — synthetic same-identity with/without pairs. Sample images on this
                page © their authors, <strong>CC BY-NC-SA</strong>; used non-commercially with attribution.
              </li>
              <li>MT dataset — research-scoped samples used during early phases.</li>
            </ul>
            <h3>Project</h3>
            <ul>
              <li><a href={GITHUB_URL} target="_blank" rel="noreferrer">GitHub repository ↗</a></li>
              <li><a href={HF_SPACE_PAGE} target="_blank" rel="noreferrer">Hugging Face Space ↗</a></li>
            </ul>
          </div>
        </div>
        <footer className="foot">
          <span>Built by Edward Lin · no makeup GAN was trained in the making of this site</span>
        </footer>
      </section>
    </main>
  );
}
