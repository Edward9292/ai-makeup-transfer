import Pipeline from "@/components/Pipeline";
import { GITHUB_URL, HF_SPACE_PAGE, HF_SPACE_URL } from "@/lib/links";

/* Pigment chips are the mean lip / lid / skin colours parsed off each reference face,
   so a look's swatches are literally the colours the engine reads from it. */
const LOOKS = [
  {
    code: "002386 · 03",
    name: "Bright red lip",
    swatches: ["#BD5759", "#A88179", "#C8A196"],
    image: "/results/ffhq-002386-03_side_by_side.png",
    alt: "A man in glasses before and after the bright red lip look is applied",
    note: "Lip, lid and brow — on a face that never wore it, glasses left alone.",
  },
  {
    code: "003125 · 02",
    name: "Crimson & smoke",
    swatches: ["#C5454F", "#574A4B", "#A98882"],
    image: "/results/ffhq-003125-02_side_by_side.png",
    alt: "A freckled face before and after the crimson and smoke look is applied",
    note: "A second identity, same engine, no per-photo tuning. Freckles survive.",
  },
];

function Swatches({ colors }: { colors: string[] }) {
  return (
    <span className="swatches" aria-hidden="true">
      {colors.map((c) => (
        <span key={c} className="swatch" style={{ background: c }} />
      ))}
    </span>
  );
}

export default function Home() {
  return (
    <>
      <header className="nav">
        <div className="nav-inner">
          <a className="brand" href="#top">
            <span className="brand-mark" aria-hidden="true" />
            Makeup Transfer
          </a>
          <nav className="nav-links" aria-label="Sections">
            <a href="#how">How it works</a>
            <a href="#looks">Looks</a>
            <a href="#build">Behind the build</a>
          </nav>
          <a className="btn btn-primary btn-sm" href="#demo">
            Try the demo
          </a>
        </div>
      </header>

      <main id="top">
        {/* ---------------- hero ---------------- */}
        <section className="hero">
          <div className="wrap">
            <p className="kicker">Makeup from any photo</p>
            <h1>
              One look.
              <br />
              Any face.
            </h1>
            <p className="hero-sub">
              Borrow the makeup from any photo and wear it on anyone — their own skin,
              lighting and complexion stay theirs.
            </p>
            <div className="cta-row">
              <a className="btn btn-primary" href="#demo">
                Try the demo
              </a>
              <a className="btn" href={GITHUB_URL} target="_blank" rel="noreferrer">
                View the code
              </a>
            </div>

            <figure className="specimen">
              <div className="specimen-cell">
                <img
                  src="/results/002386_makeup_03.jpg"
                  alt="Source photo of a face wearing a red lip, filled brows and soft eye shadow"
                />
                <p className="step">
                  <span>1</span> Source look
                </p>
              </div>
              <div className="specimen-cell">
                <img
                  src="/results/ffhq-002386-03_reference.png"
                  alt="The saved look: the reference face the engine reads the makeup from"
                />
                <p className="step">
                  <span>2</span> Saved look
                  <Swatches colors={LOOKS[0].swatches} />
                </p>
              </div>
              <div className="specimen-cell">
                <div className="crop-right">
                  <img
                    src="/results/ffhq-002386-03_side_by_side.png"
                    alt="The same look worn by a different person"
                  />
                </div>
                <p className="step">
                  <span>3</span> Worn by someone else
                </p>
              </div>
            </figure>
          </div>
        </section>

        {/* ---------------- how it works ---------------- */}
        <section id="how" className="section">
          <div className="wrap">
            <p className="kicker">How it works</p>
            <h2>Only the makeup travels.</h2>
            <p className="lede">
              Skin tone, texture and lighting stay the wearer&apos;s own, so one look reads
              differently on every face.
            </p>

            <ol className="steps">
              <li>
                <h3>Pick</h3>
                <p>One photo of anyone wearing makeup you like. That face becomes the look.</p>
              </li>
              <li>
                <h3>Transfer</h3>
                <p>
                  A makeup-transfer network reads the makeup off that face and paints it onto
                  yours — matched region to region, whatever the pose.
                </p>
              </li>
              <li>
                <h3>Wear</h3>
                <p>
                  Only the change is composited back onto your full-resolution photo, per
                  region, with intensity and complexion you control.
                </p>
              </li>
            </ol>

            <details className="disclosure">
              <summary>See the full pipeline</summary>
              <Pipeline />
              <p className="disclosure-note">
                The network (CSD-MT, CVPR 2024) works on a 256-pixel face crop. Rather than
                returning that crop, we add its makeup <em>difference</em> onto the untouched
                photo, gated by face parsing and a skin mask, so grain, hair and glasses
                survive. A complexion control keeps the reference&apos;s foundation from
                lightening your skin.
              </p>
            </details>
          </div>
        </section>

        {/* ---------------- looks ---------------- */}
        <section id="looks" className="section section-tint">
          <div className="wrap">
            <p className="kicker">Looks</p>
            <h2>Three looks, three faces.</h2>
            <p className="lede">
              Every look applied to every test face — pale, freckled, male in glasses.
            </p>

            <div className="look-grid">
              {LOOKS.map((look) => (
                <figure key={look.code} className="look">
                  <div className="ba">
                    <img src={look.image} alt={look.alt} loading="lazy" />
                    <span className="ba-tag ba-before">Before</span>
                    <span className="ba-tag ba-after">After</span>
                  </div>
                  <figcaption>
                    <div className="look-head">
                      <Swatches colors={look.swatches} />
                      <div>
                        <h3>{look.name}</h3>
                        <p className="look-code">{look.code}</p>
                      </div>
                    </div>
                    <p>{look.note}</p>
                  </figcaption>
                </figure>
              ))}
            </div>

            <figure className="matrix">
              <img
                src="/results/verification_grid.png"
                alt="Matrix of three looks applied to three different faces"
                loading="lazy"
              />
              <figcaption>Every look, every face. Left column is where each look came from.</figcaption>
            </figure>
          </div>
        </section>

        {/* ---------------- demo ---------------- */}
        <section id="demo" className="section section-dark">
          <div className="wrap">
            <p className="kicker">Live demo</p>
            <h2>Take a look for a spin.</h2>
            <p className="lede">
              Make a look from any photo, try it on a new face, and adjust each region. Free, in
              the browser, a few seconds per face.
            </p>
            <div className="demo-frame">
              <iframe
                src={HF_SPACE_URL}
                title="Makeup Transfer — live demo"
                loading="lazy"
                allow="clipboard-write"
              />
            </div>
            <p className="demo-note">
              The demo sleeps when idle, so the first load can take a minute. Single-photo mode
              needs a GPU and runs locally only.{" "}
              <a href={HF_SPACE_PAGE} target="_blank" rel="noreferrer">
                Open it in a new tab
              </a>
              .
            </p>
          </div>
        </section>

        {/* ---------------- behind the build ---------------- */}
        <section id="build" className="section">
          <div className="wrap">
            <p className="kicker">Behind the build</p>
            <h2>What worked, and what didn&apos;t.</h2>
            <p className="lede">
              The project ran on go/no-go gates. Three passed, one didn&apos;t — and the first
              design was replaced once real results came back flat.
            </p>

            <div className="log">
              <details open>
                <summary>
                  <span className="log-title">Replacing the extractor with a transfer network</span>
                  <span className="tag tag-pass">Shipped</span>
                </summary>
                <p>
                  The original design lifted makeup off a with/without pair as a transparent
                  layer. On real photos the layer kept only a faint lip tint — shadow, liner and
                  blush were filtered out as lighting — so results read as a colour wash. A
                  reference-based network (CSD-MT) now decides what is makeup; the
                  full-resolution compositing, occlusion masking and per-region controls from the
                  first design were kept around it. Two fixes came from looking: a reference in
                  glasses leaked its frames as eye shadow (spectacles are now treated as
                  background), and the reference&apos;s foundation whitened faces (a complexion
                  control damps the uniform skin shift in Lab, keeping blush).
                </p>
              </details>
              <details>
                <summary>
                  <span className="log-title">Keeping shadows out of the layer</span>
                  <span className="tag tag-pass">Passed</span>
                </summary>
                <p>
                  The main risk was reading shading as makeup. Differencing in chromaticity with
                  normalized luminance, plus left/right symmetry suppression, kept the mattes clean
                  on held-out faces.
                </p>
              </details>
              <details>
                <summary>
                  <span className="log-title">Applying without blurring the photo</span>
                  <span className="tag tag-pass">Passed</span>
                </summary>
                <p>
                  Compositing only the makeup difference back onto the untouched photo cancelled
                  resampling blur — which retired a planned neural refiner. A late bug tinted teeth
                  on open smiles; region exclusions are now re-asserted after every blur.
                </p>
              </details>
              <details>
                <summary>
                  <span className="log-title">Working from a single photo</span>
                  <span className="tag tag-partial">2 of 3</span>
                </summary>
                <p>
                  Synthesizing the bare face makes the photo pair optional. Two of three held-out
                  faces matched the paired result. The third — a male face in glasses — drifted in
                  skin tone, and the pipeline read that drift as foundation.
                </p>
              </details>
              <details>
                <summary>
                  <span className="log-title">A learned prior to fix that drift</span>
                  <span className="tag tag-fail">Missed gate</span>
                </summary>
                <p>
                  A 64-component prior cut the false makeup signal by 35%, but the gate asked for
                  far more and one clean face regressed. Verdict: the drift is upstream, not a
                  weak prior. The prior stays in the codebase behind a flag, off by default.
                </p>
                <figure className="matrix">
                  <img
                    src="/results/prior_grid.png"
                    alt="Evaluation grid comparing the prior against the baseline across settings and faces"
                    loading="lazy"
                  />
                  <figcaption>
                    The evidence behind the failed gate.{" "}
                    <a
                      href={`${GITHUB_URL}/blob/master/docs/BUILD_PLAN.md`}
                      target="_blank"
                      rel="noreferrer"
                    >
                      Full numbers
                    </a>
                    .
                  </figcaption>
                </figure>
              </details>
            </div>
          </div>
        </section>
      </main>

      {/* ---------------- footer ---------------- */}
      <footer className="footer">
        <div className="wrap footer-inner">
          <div className="footer-brand">
            <a className="brand" href="#top">
              <span className="brand-mark" aria-hidden="true" />
              Makeup Transfer
            </a>
            <p>Built by Edward Lin.</p>
          </div>
          <div>
            <h4>Project</h4>
            <ul>
              <li>
                <a href={GITHUB_URL} target="_blank" rel="noreferrer">
                  GitHub
                </a>
              </li>
              <li>
                <a href={HF_SPACE_PAGE} target="_blank" rel="noreferrer">
                  Hugging Face Space
                </a>
              </li>
            </ul>
          </div>
          <div>
            <h4>Built on</h4>
            <ul>
              <li>
                <a
                  href="https://github.com/kyotovision-public/makeup-extract"
                  target="_blank"
                  rel="noreferrer"
                >
                  makeup-extract
                </a>
              </li>
              <li>
                <a
                  href="https://yangxueyan.github.io/publication/makeup-prior/"
                  target="_blank"
                  rel="noreferrer"
                >
                  Makeup Priors (CVPR 2024)
                </a>
              </li>
              <li>
                <a href="https://github.com/Snowfallingplum/SSAT" target="_blank" rel="noreferrer">
                  SSAT (AAAI 2022)
                </a>
              </li>
              <li>
                <a
                  href="https://github.com/zllrunning/face-parsing.PyTorch"
                  target="_blank"
                  rel="noreferrer"
                >
                  face-parsing.PyTorch
                </a>
              </li>
              <li>
                <a href="https://developers.google.com/mediapipe" target="_blank" rel="noreferrer">
                  MediaPipe
                </a>
              </li>
            </ul>
          </div>
          <div>
            <h4>Images</h4>
            <ul>
              <li>
                <a
                  href="https://huggingface.co/datasets/cyberagent/FFHQ-Makeup"
                  target="_blank"
                  rel="noreferrer"
                >
                  FFHQ-Makeup
                </a>{" "}
                (CyberAgent)
              </li>
              <li>© their authors, CC BY-NC-SA. Used non-commercially with attribution.</li>
            </ul>
          </div>
        </div>
      </footer>
    </>
  );
}
