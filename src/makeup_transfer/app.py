"""Phase 3: Gradio web demo (``app.py``).

A thin browser UI over the already-built pipeline — it only calls existing public functions
(``extract_look`` / ``extract_look_single`` / ``apply_look`` / ``LookStore``), reimplementing no
pipeline logic.

The UI follows the order a first-time visitor actually works in:

1. **Try a look** — pick a saved look + a target face -> before/after result (CPU).
2. **Make a look** — with-makeup + bare photos of the same person -> saved RGBA look (CPU).
   Its "one photo only" panel uses SSAT removal (Phase 2.5) and **requires CUDA**.
3. **Your looks** — browse / delete / zip-export saved looks.

Uploads are handled as file *paths* (``gr.Image(type="filepath")``) to match the pipeline's
path-based API. Pipeline outputs are OpenCV BGR arrays, converted to RGB for display here.
Visual language (porcelain/ink/pigment, Jost + Karla, pigment-chip swatches) matches the
product site so the demo reads as part of the same product.

Run (from the repo root)::

    $env:PYTHONPATH = "src"          # package lives under src/
    python -m makeup_transfer.app
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import gradio as gr

from .apply import DEFAULT_REGION_PARAMS, apply_look, region_weights_uv
from .extract import _checker_preview, extract_look, extract_look_single
from .storage import LookStore

_ROOT = Path(__file__).resolve().parents[2]
WEBAPP_OUT = _ROOT / "preprocess_out" / "webapp"
WEBAPP_OUT.mkdir(parents=True, exist_ok=True)
SAMPLES = _ROOT / "data" / "ffhq_makeup"

store = LookStore()

# Slider order used everywhere strengths are read/written: lips, eyes, brows, skin.
_REGIONS = ["lips", "eyes", "brows", "skin"]


def _single_photo_available() -> tuple[bool, str]:
    """Can the SSAT single-photo panel actually run here? (ok, reason-if-not)."""
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


# --- small helpers ---------------------------------------------------------------------------
def _bgr2rgb(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _preview_rgb(layer_rgba: np.ndarray) -> np.ndarray:
    """RGBA UV layer -> viewable RGB checkerboard composite (alpha shows through)."""
    return _bgr2rgb(_checker_preview(layer_rgba))


def _looks() -> list[str]:
    return store.list_looks()


def _gallery_items() -> list[tuple[np.ndarray, str]]:
    items: list[tuple[np.ndarray, str]] = []
    for name in _looks():
        try:
            items.append((_preview_rgb(store.load_layer(name)), name))
        except Exception:  # a half-written look shouldn't break the gallery
            continue
    return items


def _look_strengths(look: str | None) -> tuple[float, float, float, float]:
    """Per-region strengths for a look (its meta merged over defaults), in _REGIONS order."""
    regions = {k: dict(v) for k, v in DEFAULT_REGION_PARAMS.items()}
    if look:
        for k, v in (store.load_meta(look).blend_params or {}).get("regions", {}).items():
            regions.setdefault(k, {}).update(v)
    return tuple(float(regions[r].get("strength", 1.0)) for r in _REGIONS)  # type: ignore[return-value]


# --- pigment chips ----------------------------------------------------------------------------
@lru_cache(maxsize=1)
def _uv_regions(uv_size: int) -> dict[str, np.ndarray]:
    return region_weights_uv(uv_size)


def _region_pigment(layer_rgba: np.ndarray, region: np.ndarray) -> tuple[str, float] | None:
    """Alpha-weighted mean colour of a look inside one region -> (#hex, coverage 0-1).

    ``region`` is a normalized weight map, so its sum is the region's area in pixels and
    ``coverage`` reads as "how much of this region the look actually paints".
    """
    alpha = (layer_rgba[..., 3].astype(np.float32) / 255.0) * region
    total = float(alpha.sum())
    coverage = total / max(float(region.sum()), 1.0)
    if total < 1.0 or coverage < 0.02:
        return None
    rgb = layer_rgba[..., :3].astype(np.float32)  # LookStore.load_layer returns RGBA
    r, g, b = ((rgb * alpha[..., None]).reshape(-1, 3).sum(axis=0) / total)
    return "#{:02x}{:02x}{:02x}".format(*(int(np.clip(c, 0, 255)) for c in (r, g, b))), coverage


def _swatch_html(look: str | None) -> str:
    """The look's real pigments, sampled per region — what you get before you apply it."""
    if not look:
        return ""
    try:
        layer = store.load_layer(look)
        regions = _uv_regions(layer.shape[0])
    except Exception:
        return ""
    chips = []
    for name in _REGIONS:
        weights = regions.get(name)
        if weights is None:
            continue
        sample = _region_pigment(layer, weights)
        if sample is None:
            continue
        hex_colour, coverage = sample
        chips.append(
            f'<div class="chip"><span class="chip-dot" style="background:{hex_colour}"></span>'
            f'<span class="chip-name">{name}</span>'
            f'<span class="chip-hex">{hex_colour} · {round(coverage * 100)}%</span></div>'
        )
    if not chips:
        return ""
    return (f'<div class="chips"><p class="chips-title">Pigments in {look}</p>'
            f'<div class="chip-row">{"".join(chips)}</div></div>')


def _look_panel(look: str | None):
    """Everything that follows from selecting a look: its recipe sliders + its pigments."""
    return (*_look_strengths(look), _swatch_html(look))


# --- sample images ----------------------------------------------------------------------------
def _sample(name: str) -> str | None:
    path = SAMPLES / name
    return str(path) if path.exists() else None


def _sample_rows(rows: list[list[str | float]]) -> list[list]:
    """Keep only example rows whose sample images shipped with this build."""
    kept = []
    for row in rows:
        resolved = [_sample(v) if isinstance(v, str) and v.endswith(".jpg") else v for v in row]
        if all(v is not None for v in resolved):
            kept.append(resolved)
    return kept


# --- tab callbacks ---------------------------------------------------------------------------
def do_extract_pair(with_path, bare_path, name, sym):
    if not with_path or not bare_path:
        raise gr.Error("Upload both a with-makeup photo and a bare photo of the same person.")
    name = (name or "").strip()
    if not name:
        raise gr.Error("Give the look a name.")
    try:
        layer, _, _ = extract_look(with_path, bare_path, name, sym=float(sym), store=store)
    except ValueError as e:  # detect_face raises ValueError when no face is found
        raise gr.Error(f"Could not process the photos: {e}. Use frontal, well-lit faces.")
    looks = _looks()
    items = _gallery_items()
    return (
        gr.update(value=_preview_rgb(layer), visible=True),
        f"Saved **{name}**. Pick it in Try a look to wear it on another face.",
        gr.update(choices=looks, value=name),
        gr.update(choices=looks, value=name),
        items,
        items,
    )


def do_extract_single(with_path, ref_path, name, sym):
    if not with_path:
        raise gr.Error("Upload a with-makeup photo.")
    name = (name or "").strip()
    if not name:
        raise gr.Error("Give the look a name.")
    pseudo_path = WEBAPP_OUT / f"{name}_pseudo_bare.png"
    try:
        layer, _, _ = extract_look_single(
            with_path, name, sym=float(sym), store=store,
            reference=ref_path or None, pseudo_out=pseudo_path,
        )
    except RuntimeError as e:  # SSAT removal asserts CUDA (removal.py)
        raise gr.Error(
            f"Single-photo mode needs a CUDA GPU ({e}). "
            "On a CPU-only machine, use two photos instead."
        )
    except ValueError as e:
        raise gr.Error(f"Could not process the photo: {e}. Use a frontal, well-lit face.")
    pseudo = cv2.imread(str(pseudo_path))
    looks = _looks()
    items = _gallery_items()
    return (
        gr.update(value=_bgr2rgb(pseudo) if pseudo is not None else None,
                  visible=pseudo is not None),
        gr.update(value=_preview_rgb(layer), visible=True),
        f"Saved **{name}** from one photo. Pick it in Try a look to wear it on another face.",
        gr.update(choices=looks, value=name),
        gr.update(choices=looks, value=name),
        items,
        items,
    )


def do_apply(look, target_path, lips, eyes, brows, skin, composite):
    if not look:
        raise gr.Error("Pick a look first.")
    if not target_path:
        raise gr.Error("Add a face to wear the look.")
    strengths = {"lips": float(lips), "eyes": float(eyes),
                 "brows": float(brows), "skin": float(skin)}
    try:
        result_bgr, debug = apply_look(
            look, target_path, strengths=strengths, composite=composite, store=store,
        )
    except ValueError as e:
        raise gr.Error(f"Could not apply the look: {e}")
    out_path = WEBAPP_OUT / f"{look}_on_target.png"
    cv2.imwrite(str(out_path), result_bgr)
    return (gr.update(value=(_bgr2rgb(debug["target"]), _bgr2rgb(result_bgr)), visible=True),
            gr.update(value=str(out_path), visible=True))


def do_pick_from_gallery(evt: gr.SelectData):
    names = _looks()
    return names[evt.index] if 0 <= evt.index < len(names) else None


def do_delete(look):
    if not look:
        raise gr.Error("Pick a look to delete.")
    store.delete(look)
    looks = _looks()
    items = _gallery_items()
    return (
        gr.update(choices=looks, value=None),
        gr.update(choices=looks, value=None),
        items,
        items,
        f"Deleted **{look}**.",
    )


def do_export(look):
    if not look:
        raise gr.Error("Pick a look to export.")
    out = WEBAPP_OUT / f"{look}.zip"
    store.export_zip(look, out)
    return gr.update(value=str(out), visible=True)


def do_refresh():
    looks = _looks()
    items = _gallery_items()
    return gr.update(choices=looks), gr.update(choices=looks), items, items


# --- look & feel ------------------------------------------------------------------------------
THEME = gr.themes.Base(
    primary_hue=gr.themes.Color(
        c50="#fdf2f5", c100="#fbe3ea", c200="#f3c2d0", c300="#e694ac",
        c400="#d45f83", c500="#b0173f", c600="#9d1338", c700="#8d0f31",
        c800="#6f0c27", c900="#520a1d", c950="#360613",
    ),
    neutral_hue=gr.themes.Color(
        c50="#fbf9f8", c100="#f2ecea", c200="#e8dedb", c300="#d3c6c3",
        c400="#a6969b", c500="#6b5b63", c600="#54454c", c700="#3a2f3c",
        c800="#241c28", c900="#17121a", c950="#0f0b11",
    ),
    font=(gr.themes.GoogleFont("Karla"), "ui-sans-serif", "system-ui", "sans-serif"),
    font_mono=(gr.themes.GoogleFont("IBM Plex Mono"), "ui-monospace", "monospace"),
    radius_size=gr.themes.sizes.radius_sm,
).set(
    body_background_fill="#fbf9f8",
    body_background_fill_dark="#17121a",
    block_background_fill="#ffffff",
    block_background_fill_dark="#241c28",
    block_border_color="#e8dedb",
    block_border_color_dark="#3a2f3c",
    block_label_text_weight="600",
    button_primary_background_fill="#b0173f",
    button_primary_background_fill_hover="#8d0f31",
    button_primary_text_color="#ffffff",
    button_large_radius="6px",
    button_small_radius="6px",
)

CSS = """
:root, .dark { --display: 'Jost', ui-sans-serif, system-ui, sans-serif; }
.gradio-container { max-width: 1180px !important; }

/* masthead ------------------------------------------------------------------ */
#masthead { border-bottom: 1px solid var(--block-border-color); padding-bottom: 1.1rem; }
#masthead h1 {
  font-family: var(--display); font-weight: 300; letter-spacing: -0.03em;
  font-size: clamp(2.4rem, 6vw, 3.6rem); line-height: 1.02; margin: 0 0 .5rem;
}
#masthead p { max-width: 42rem; color: var(--body-text-color-subdued); margin: 0; }
#masthead .rule { color: var(--primary-500); font-weight: 500; }

/* step eyebrows: the tabs really are a sequence, so they are numbered -------- */
.kicker { padding: .3rem 0 .9rem; }
.kicker .eyebrow {
  font-family: var(--display); font-size: .72rem; font-weight: 600;
  text-transform: uppercase; letter-spacing: .18em;
  color: var(--primary-500); margin: 0 0 .4rem;
}
.kicker .eyebrow .n { color: var(--body-text-color-subdued); margin-right: .6rem; }
.kicker .sub {
  max-width: 46rem; color: var(--body-text-color-subdued);
  font-size: .95rem; margin: 0;
}
.kicker em { font-style: normal; color: var(--body-text-color); font-weight: 600; }
.note p { color: var(--body-text-color-subdued); font-size: .92rem; margin: 0 0 .2rem; }

/* look picker --------------------------------------------------------------- */
.looks .caption-label { font-family: var(--display); font-size: .72rem; }
/* narrow screens: wrap the picker into rows instead of clipping looks offscreen */
@media (max-width: 700px) {
  .looks, .looks .grid-wrap { height: 340px !important; max-height: none !important; }
  .looks .grid-container {
    grid-template-columns: repeat(2, minmax(0, 1fr)) !important;
  }
}

/* tabs ---------------------------------------------------------------------- */
.tab-nav button {
  font-family: var(--display) !important; font-weight: 500 !important;
  letter-spacing: .01em; font-size: 1rem !important;
}
button.selected { color: var(--primary-500) !important; }

/* pigment chips — the look's real colours, sampled from the layer ------------ */
.chips { padding: .1rem 0 .2rem; }
.chips-title {
  font-family: var(--display); font-size: .68rem; font-weight: 600;
  text-transform: uppercase; letter-spacing: .16em;
  color: var(--body-text-color-subdued); margin: 0 0 .5rem;
}
.chip-row { display: flex; flex-wrap: wrap; gap: .4rem; }
.chip {
  display: flex; align-items: center; gap: .45rem;
  border: 1px solid var(--block-border-color); border-radius: 999px;
  padding: .26rem .7rem .26rem .3rem; background: var(--block-background-fill);
}
.chip-dot {
  width: 1.05rem; height: 1.05rem; border-radius: 999px; display: inline-block;
  box-shadow: inset 0 0 0 1px rgba(23, 18, 26, .16);
}
.chip-name { font-family: var(--display); font-weight: 500; font-size: .8rem; }
.chip-hex {
  font-family: var(--font-mono); font-size: .68rem;
  color: var(--body-text-color-subdued);
}

/* actions ------------------------------------------------------------------- */
.go button, .go { font-family: var(--display) !important; font-weight: 500 !important;
  letter-spacing: .02em; }
button.stop {
  background: transparent !important; color: var(--primary-600) !important;
  border: 1px solid var(--primary-200) !important;
}
button.stop:hover { background: var(--primary-50) !important; }
.dark button.stop { border-color: var(--primary-800) !important; }
.dark button.stop:hover { background: var(--primary-950) !important; }
.dark .chip-dot { box-shadow: inset 0 0 0 1px rgba(255, 255, 255, .22); }
footer { display: none !important; }
"""

_FONT_LINK = (
    '<link rel="preconnect" href="https://fonts.googleapis.com">'
    '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
    '<link href="https://fonts.googleapis.com/css2?family=Jost:wght@300;400;500;600'
    '&display=swap" rel="stylesheet">'
)


def _step(number: str, title: str, sub: str) -> str:
    """Step eyebrow + one line of guidance. The tabs are a real sequence, so they're numbered."""
    return (f'<div class="kicker"><p class="eyebrow"><span class="n">{number}</span>{title}</p>'
            f'<p class="sub">{sub}</p></div>')


# --- UI --------------------------------------------------------------------------------------
def build_demo() -> gr.Blocks:
    initial = _looks()
    first = initial[0] if initial else None
    l0, e0, b0, s0 = _look_strengths(first)

    with gr.Blocks(title="Makeup Transfer", theme=THEME, css=CSS, head=_FONT_LINK) as demo:
        with gr.Column(elem_id="masthead"):
            gr.HTML(
                "<h1>Makeup&nbsp;Transfer</h1>"
                "<p>Lift a makeup look off one face and wear it on another. The look is stored as "
                "a <span class='rule'>translucent layer in 3D UV space</span>, so skin tone shows "
                "through and the same look adapts to whoever wears it.</p>"
            )

        with gr.Tabs():
            # --- Step 1: try a look ---------------------------------------------------------
            with gr.Tab("Try a look"):
                gr.HTML(_step("01", "Pick a look, add a face",
                              "A saved look plus the face that will wear it. Frontal, well-lit "
                              "photos work best."))

                with gr.Row(equal_height=False):
                    with gr.Column(scale=1):
                        a_gallery = gr.Gallery(
                            value=_gallery_items(), label="Looks — click one",
                            columns=4, height=168, object_fit="cover",
                            allow_preview=False, elem_classes="looks",
                        )
                        a_look = gr.Dropdown(choices=initial, value=first, label="Selected look")
                        a_swatch = gr.HTML(_swatch_html(first))

                    with gr.Column(scale=1):
                        a_target = gr.Image(label="Face to make up", type="filepath", height=300)
                        target_rows = _sample_rows([["002386_bare.jpg"], ["003125_bare.jpg"],
                                                    ["003417_bare.jpg"]])
                        if target_rows:
                            gr.Examples(target_rows, inputs=a_target,
                                        label="Or start with a sample face", examples_per_page=3)

                with gr.Accordion("Fine-tune", open=False):
                    gr.Markdown("Each slider scales one region of the look. 1.0 is the strength "
                                "the look was saved at.", elem_classes="note")
                    with gr.Row():
                        a_lips = gr.Slider(0.0, 1.5, value=l0, step=0.05, label="Lips")
                        a_eyes = gr.Slider(0.0, 1.5, value=e0, step=0.05, label="Eyes")
                        a_brows = gr.Slider(0.0, 1.5, value=b0, step=0.05, label="Brows")
                        a_skin = gr.Slider(0.0, 1.5, value=s0, step=0.05, label="Skin")
                    a_comp = gr.Radio(
                        ["delta", "plain"], value="delta", label="Composite",
                        info="delta keeps the target's own skin grain; plain paints over it.",
                    )

                a_btn = gr.Button("Apply look", variant="primary", size="lg", elem_classes="go")
                a_side = gr.ImageSlider(
                    label="Before / after — drag the handle", visible=False,
                    type="numpy", slider_position=50, height=460,
                )
                a_file = gr.DownloadButton("Download result", size="lg", visible=False,
                                           elem_classes="go")

            # --- Step 2: make a look --------------------------------------------------------
            with gr.Tab("Make a look"):
                gr.HTML(_step("02", "Extract a look from photos",
                              "Two photos of the <em>same person</em> — one wearing makeup, one "
                              "bare. The difference between them, in UV space, becomes a "
                              "reusable layer."))
                with gr.Row():
                    p_with = gr.Image(label="With makeup", type="filepath", height=280)
                    p_bare = gr.Image(label="Bare — same person", type="filepath", height=280)
                pair_rows = _sample_rows([
                    ["002386_makeup_03.jpg", "002386_bare.jpg", "rose-satin"],
                    ["003125_makeup_02.jpg", "003125_bare.jpg", "warm-nude"],
                    ["003417_makeup_05.jpg", "003417_bare.jpg", "deep-plum"],
                ])
                with gr.Row():
                    p_name = gr.Textbox(label="Name this look", placeholder="e.g. smoky-red")
                    p_sym = gr.Slider(0.0, 1.0, value=0.5, step=0.05,
                                      label="Shadow suppression",
                                      info="Mirrors the UV texture to cancel lighting that "
                                           "falls on one side of the face.")
                if pair_rows:
                    gr.Examples(pair_rows, inputs=[p_with, p_bare, p_name],
                                label="Or start with a sample pair", examples_per_page=3)
                p_btn = gr.Button("Extract look", variant="primary", size="lg",
                                  elem_classes="go")
                p_status = gr.Markdown()
                p_preview = gr.Image(label="Extracted layer — checkerboard shows what's "
                                           "transparent", height=320, visible=False)

                with gr.Accordion("Only have one photo? (needs a CUDA GPU)", open=False):
                    gpu_ok, gpu_why = _single_photo_available()
                    if gpu_ok:
                        note = ("SSAT removes the makeup to synthesize a bare face, then the same "
                                "pair pipeline runs.")
                    else:
                        note = (f"**Unavailable here:** {gpu_why} Use two photos above — that "
                                "path runs on CPU.")
                    gr.Markdown(note, elem_classes="note")
                    with gr.Row():
                        s_with = gr.Image(label="With makeup", type="filepath", height=260)
                        s_ref = gr.Image(label="Bare reference — optional", type="filepath",
                                         height=260)
                    with gr.Row():
                        s_name = gr.Textbox(label="Name this look",
                                            placeholder="e.g. stolen-look")
                        s_sym = gr.Slider(0.0, 1.0, value=0.5, step=0.05,
                                          label="Shadow suppression")
                    s_btn = gr.Button("Extract look on GPU", variant="primary",
                                      interactive=gpu_ok, elem_classes="go")
                    s_status = gr.Markdown()
                    with gr.Row():
                        s_pseudo = gr.Image(label="Synthesized bare face", height=280,
                                            visible=False)
                        s_preview = gr.Image(label="Extracted layer", height=280, visible=False)

            # --- Step 3: the library --------------------------------------------------------
            with gr.Tab("Your looks"):
                gr.HTML(_step("03", "Keep, export, or clear out",
                              "Every saved look lives here as an RGBA layer plus its blend "
                              "recipe. Export one to reuse it outside this demo."
                              if initial else
                              "No looks saved yet. Extract one in <em>Make a look</em> — it "
                              "shows up here and in <em>Try a look</em>."))
                l_gallery = gr.Gallery(value=_gallery_items(), label="Saved looks",
                                       columns=4, height=280, object_fit="cover")
                with gr.Row():
                    l_pick = gr.Dropdown(choices=initial, label="Selected look", value=first,
                                         scale=3)
                    l_exp = gr.Button("Export .zip", scale=1)
                    l_refresh = gr.Button("Refresh", scale=1)
                    l_del = gr.Button("Delete", variant="stop", scale=1)
                l_status = gr.Markdown()
                l_file = gr.File(label="Exported look", visible=False)

        gr.Markdown(
            "Pretrained models only — nothing is trained here. Sample faces: "
            "[FFHQ-Makeup](https://huggingface.co/datasets/cyberagent/FFHQ-Makeup) (CC BY-NC-SA) · "
            "[source](https://github.com/Edward9292/ai-makeup-transfer)",
            elem_classes="note",
        )

        # --- wiring (all components exist now) ----------------------------------------------
        _look_outputs = [a_lips, a_eyes, a_brows, a_skin, a_swatch]

        a_gallery.select(do_pick_from_gallery, None, a_look)
        a_look.change(_look_panel, a_look, _look_outputs)
        a_btn.click(do_apply, [a_look, a_target, a_lips, a_eyes, a_brows, a_skin, a_comp],
                    [a_side, a_file], scroll_to_output=True)

        p_btn.click(do_extract_pair, [p_with, p_bare, p_name, p_sym],
                    [p_preview, p_status, a_look, l_pick, a_gallery, l_gallery])
        s_btn.click(do_extract_single, [s_with, s_ref, s_name, s_sym],
                    [s_pseudo, s_preview, s_status, a_look, l_pick, a_gallery, l_gallery])

        l_del.click(do_delete, l_pick, [a_look, l_pick, a_gallery, l_gallery, l_status])
        l_exp.click(do_export, l_pick, l_file)
        l_refresh.click(do_refresh, None, [a_look, l_pick, a_gallery, l_gallery])

    return demo


def main() -> None:
    build_demo().launch()


if __name__ == "__main__":
    main()
