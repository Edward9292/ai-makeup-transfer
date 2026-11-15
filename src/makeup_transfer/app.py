"""Phase 3: Gradio web app (``app.py``).

A thin browser UI over the already-built pipeline — it only calls existing public functions
(``engine.create_look`` / ``apply_look`` / ``LookStore``), reimplementing no pipeline logic.

Designed for a first-time visitor:

1. **Try on** — the landing screen. Three numbered steps on one page: choose a look (shown as
   the face it came from), add a photo, see the result. The result applies itself as soon as
   both are chosen — there is no button to find.
2. **Create a look** — one photo of anyone wearing makeup -> saved look (CPU, a few seconds).
3. **My looks** — pick a look, then try it on / download it / delete it.

The transfer itself is the reference-based engine (``engine.py``), with two qualities on the
Try-on screen: **Instant** (CSD-MT, about a second on CPU) and **Realistic** (Stable-Makeup
diffusion, ``diffusion.py`` — generates the face anew, about a minute on an M-series Mac or a
GPU; offered only where its weights are installed). Legacy UV-layer looks still apply through
the same ``apply_look`` call.

Uploads are handled as file *paths* (``gr.Image(type="filepath")``) to match the pipeline's
path-based API. Pipeline outputs are OpenCV BGR arrays, converted to RGB for display here.
Visual language: a "compact case" — looks are round pans, the result sits in a ringed mirror;
white paper + cornflower blue, Bricolage Grotesque + Figtree, light-only.

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
from .diffusion import DEFAULT_GUIDANCE, DEFAULT_STRENGTH
from .engine import (DEFAULT_COMPLEXION, DEFAULT_MATCH_SKIN, backend_available, create_look,
                     reference_swatches)
from .engine import available as engine_available
from .engine import missing_weights_message
from .extract import _checker_preview
from .storage import LookStore

_ROOT = Path(__file__).resolve().parents[2]
WEBAPP_OUT = _ROOT / "preprocess_out" / "webapp"
WEBAPP_OUT.mkdir(parents=True, exist_ok=True)
THUMBS = WEBAPP_OUT / "thumbs"
THUMBS.mkdir(parents=True, exist_ok=True)
SAMPLES = _ROOT / "data" / "ffhq_makeup"

store = LookStore()

# Slider order used everywhere strengths are read/written: lips, eyes, brows, skin.
_REGIONS = ["lips", "eyes", "brows", "skin"]

# The bundled looks are named after their dataset ids; show people a name instead.
_SEED_TITLES = {
    "ffhq-002386-03": "Rose Satin",
    "ffhq-003125-02": "Warm Nude",
    "ffhq-003417-05": "Deep Plum",
}
_THUMB_PX = 320


def _engine_available() -> tuple[bool, str]:
    """Can the makeup engine run here? (ok, reason-if-not). Legacy looks still work without it."""
    if not engine_available():
        return False, missing_weights_message()
    try:
        import torch  # noqa: F401
    except ImportError:
        return False, "PyTorch is not installed here (pip install torch)."
    return True, ""


_INSTANT, _REALISTIC = "Instant", "Realistic (AI-generated, about a minute)"


def _diffusion_available() -> tuple[bool, str]:
    """Can the diffusion engine run here? (ok, reason-if-not)."""
    return backend_available("diffusion")


# --- small helpers ---------------------------------------------------------------------------
def _bgr2rgb(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _preview_rgb(layer_rgba: np.ndarray) -> np.ndarray:
    """RGBA UV layer -> viewable RGB checkerboard composite (alpha shows through)."""
    return _bgr2rgb(_checker_preview(layer_rgba))


def _title(name: str | None) -> str:
    if not name:
        return ""
    return _SEED_TITLES.get(name) or name.replace("-", " ").replace("_", " ").strip().title()


def _square(img: np.ndarray, size: int = _THUMB_PX) -> np.ndarray:
    """Centre-crop to a square and shrink — thumbnails for the look picker."""
    h, w = img.shape[:2]
    s = min(h, w)
    y, x = (h - s) // 2, (w - s) // 2
    return cv2.resize(img[y:y + s, x:x + s], (size, size), interpolation=cv2.INTER_AREA)


def _save_thumb(name: str, photo_path: str | None) -> None:
    """Cache a square thumbnail of the face the look came from (uploads are temp files)."""
    img = cv2.imread(str(photo_path)) if photo_path else None
    if img is not None:
        cv2.imwrite(str(THUMBS / f"{name}.jpg"), _square(img))


def _thumb_rgb(name: str) -> np.ndarray:
    """A face wearing the look. Engine looks *are* that face (the stored reference crop);
    legacy looks fall back to their source photo, a sample face, then the UV layer."""
    if (ref := store.load_reference(name)) is not None:
        return _square(ref)
    cached = THUMBS / f"{name}.jpg"
    if not cached.exists():
        src = store.load_meta(name).source_with
        if src:
            path = Path(src) if Path(src).is_absolute() else _ROOT / src
            if path.exists():
                _save_thumb(name, str(path))
    if not cached.exists() and (fallback := _sample("003417_bare.jpg")):
        try:  # source photo gone (e.g. a temp upload): show the look worn by a sample face
            worn, _ = apply_look(name, fallback, store=store)
            cv2.imwrite(str(cached), _square(worn))
        except Exception:
            pass
    img = cv2.imread(str(cached)) if cached.exists() else None
    if img is None:
        return _preview_rgb(store.load_layer(name))
    return _bgr2rgb(img)


def _library() -> list[tuple[str, np.ndarray]]:
    """(name, thumbnail) for every loadable look — one list so gallery index == name index."""
    out: list[tuple[str, np.ndarray]] = []
    for name in store.list_looks():
        try:
            out.append((name, _thumb_rgb(name)))
        except Exception:  # a half-written look shouldn't break the gallery
            continue
    return out


def _look_names() -> list[str]:
    return [name for name, _ in _library()]


def _gallery(selected: str | None = None):
    """Gallery value (+ highlighted selection) built from the current library."""
    lib = _library()
    names = [n for n, _ in lib]
    return gr.update(value=[(img, _title(n)) for n, img in lib],
                     selected_index=names.index(selected) if selected in names else None)


def _look_strengths(look: str | None) -> tuple[float, float, float, float, float]:
    """Per-region strengths for a look (its meta merged over defaults), in _REGIONS order,
    followed by its complexion setting (engine looks; ignored by legacy looks)."""
    regions = {k: dict(v) for k, v in DEFAULT_REGION_PARAMS.items()}
    complexion = DEFAULT_COMPLEXION
    if look:
        bp = store.load_meta(look).blend_params or {}
        for k, v in bp.get("regions", {}).items():
            regions.setdefault(k, {}).update(v)
        complexion = float(bp.get("complexion", complexion))
    return (*(float(regions[r].get("strength", 1.0)) for r in _REGIONS), complexion)  # type: ignore[return-value]


# --- colour chips -----------------------------------------------------------------------------
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


def _look_colors(look: str) -> dict[str, str]:
    """``region -> #hex`` for a look: parsed off the reference face (engine looks) or sampled
    from the UV layer (legacy looks). Empty when neither is possible."""
    if (ref := store.load_reference(look)) is not None:
        if not _engine_available()[0]:
            return {}
        return reference_swatches(ref)
    layer = store.load_layer(look)
    regions = _uv_regions(layer.shape[0])
    out = {}
    for name in _REGIONS:
        if (weights := regions.get(name)) is None:
            continue
        if (sample := _region_pigment(layer, weights)) is not None:
            out[name] = sample[0]
    return out


def _swatch_html(look: str | None) -> str:
    """The look's real colours, sampled per region — what you get before you apply it."""
    if not look:
        return ""
    try:
        colors = _look_colors(look)
    except Exception:
        return ""
    chips = []
    for name in _REGIONS:
        if (hex_color := colors.get(name)) is None:
            continue
        chips.append(f'<span class="chip" title="{hex_color}">'
                     f'<span class="chip-dot" style="background:{hex_color}"></span>'
                     f'{name}</span>')
    if not chips:
        return ""
    return (f'<div class="chips"><span class="chips-title">{_title(look)}</span>'
            f'{"".join(chips)}</div>')


def _look_panel(look: str | None):
    """Everything that follows from selecting a look: its intensity sliders + its colours."""
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


# --- try-on callbacks -------------------------------------------------------------------------
def do_select_look(evt: gr.SelectData):
    names = _look_names()
    look = names[evt.index] if 0 <= evt.index < len(names) else None
    return (look, *_look_panel(look))


def do_try(look, target_path, lips, eyes, brows, skin, complexion,
           quality=_INSTANT, guidance=DEFAULT_GUIDANCE, strength=DEFAULT_STRENGTH,
           match_skin=DEFAULT_MATCH_SKIN, progress=gr.Progress()):
    """Apply as soon as both halves are there; until then, say what's missing."""
    if not look or not target_path:
        hint = ("Choose a look in step 1." if not look else
                "Add a photo in step 2 — or tap one of the sample faces.")
        return gr.skip(), gr.skip(), gr.skip(), hint
    strengths = {"lips": float(lips), "eyes": float(eyes),
                 "brows": float(brows), "skin": float(skin)}
    if store.has_reference(look) and not (ok := _engine_available())[0]:
        return gr.skip(), gr.skip(), gr.skip(), f"**The makeup engine isn't set up here.** {ok[1]}"
    diffusion = quality == _REALISTIC and store.has_reference(look)
    if diffusion and not (ok := _diffusion_available())[0]:
        return gr.skip(), gr.skip(), gr.skip(), f"**Realistic mode isn't set up here.** {ok[1]}"

    def on_step(i, n):
        progress(i / n, desc=f"Painting the look · step {i} of {n}")

    try:
        if diffusion:
            progress(0.0, desc="Warming up the diffusion model (first run takes longer)")
        result_bgr, debug = apply_look(
            look, target_path, strengths=strengths, store=store, complexion=float(complexion),
            match_skin=float(match_skin),
            backend="diffusion" if diffusion else "csd-mt",
            guidance=float(guidance) if diffusion else None,
            strength=float(strength) if diffusion else None,
            progress=on_step if diffusion else None,
        )
    except ValueError:
        return (gr.skip(), gr.skip(), gr.skip(),
                "**We couldn't find a face in that photo.** Try a clear, front-facing photo "
                "in good light.")
    out_path = WEBAPP_OUT / f"makeup-{look}.png"
    cv2.imwrite(str(out_path), result_bgr)
    how = " · generated with diffusion" if diffusion else ""
    return (gr.update(visible=False),
            gr.update(value=(_bgr2rgb(debug["target"]), _bgr2rgb(result_bgr)), visible=True),
            gr.update(value=str(out_path), visible=True),
            f"Wearing **{_title(look)}**{how} · drag the handle to compare")


# --- create-a-look callbacks ------------------------------------------------------------------
def _created(name: str, photo_path: str | None, message: str):
    """Shared tail of both extract paths: thumbnail, refreshed galleries, a way forward."""
    _save_thumb(name, photo_path)
    return (message, gr.update(visible=True), name,
            _gallery(name), _gallery(name))


def do_create(with_path, name):
    if not with_path:
        raise gr.Error("Add a photo of someone wearing the makeup.")
    name = (name or "").strip()
    if not name:
        raise gr.Error("Give your look a name.")
    ok, why = _engine_available()
    if not ok:
        raise gr.Error(f"The makeup engine isn't set up here. {why}")
    try:
        ref, _ = create_look(with_path, name, store=store)
    except ValueError as e:  # detect_face raises ValueError when no face is found
        raise gr.Error(f"We couldn't read that photo ({e}). Use a front-facing, well-lit face.")
    return (gr.update(value=ref, visible=True),
            *_created(name, with_path, f"**{_title(name)}** is saved to your looks."))


def do_go_try(look):
    """Jump to the Try-on screen with ``look`` selected."""
    return (gr.Tabs(selected="try"), look, _gallery(look), *_look_panel(look))


# --- my-looks callbacks -----------------------------------------------------------------------
def do_select_library(evt: gr.SelectData):
    names = _look_names()
    look = names[evt.index] if 0 <= evt.index < len(names) else None
    return (look, gr.update(visible=look is not None),
            f"### {_title(look)}" if look else "", gr.update(visible=False),
            gr.update(visible=False))


def do_export(look):
    if not look:
        raise gr.Error("Choose a look first.")
    out = WEBAPP_OUT / f"{look}.zip"
    store.export_zip(look, out)
    return gr.update(value=str(out), visible=True)


def do_ask_delete(look):
    return gr.update(visible=bool(look), value=f"Delete **{_title(look)}**? This can't be undone.")


def do_delete(look, try_look):
    if not look:
        raise gr.Error("Choose a look first.")
    store.delete(look)
    (THUMBS / f"{look}.jpg").unlink(missing_ok=True)
    try_look = None if try_look == look else try_look
    return (None, gr.update(visible=False), gr.update(visible=False), gr.update(visible=False),
            try_look, _gallery(try_look), _gallery(None))


def do_reload(try_look):
    return _gallery(try_look), _gallery(None)


# --- look & feel ------------------------------------------------------------------------------
# "Compact case": looks are round pans, the result sits in a ringed mirror. White paper,
# cornflower ink, powder-blue tints — the makeup on the faces is the only warm colour.
_PAPER, _MIST, _LINE, _POWDER = "#ffffff", "#f3f7fb", "#e6edf5", "#dcebf9"
_CORNFLOWER, _CORNFLOWER_DEEP, _INK, _SLATE = "#2f74bb", "#265f9b", "#1e2633", "#6a7686"

THEME = gr.themes.Base(
    primary_hue=gr.themes.Color(
        c50="#f1f7fd", c100="#e3effb", c200="#c8e0f6", c300="#9fc8ef",
        c400="#6aa9e3", c500="#3a8ad6", c600="#2f74bb", c700="#265f9b",
        c800="#224f7e", c900="#1f4367", c950="#152c45",
    ),
    neutral_hue=gr.themes.Color(
        c50="#f8fafc", c100="#f3f7fb", c200="#e6edf5", c300="#d2dce7",
        c400="#9ba8b7", c500="#6a7686", c600="#515d6c", c700="#3c4655",
        c800="#2a3240", c900="#1e2633", c950="#131922",
    ),
    font=(gr.themes.GoogleFont("Figtree"), "ui-sans-serif", "system-ui", "sans-serif"),
    font_mono=(gr.themes.GoogleFont("IBM Plex Mono"), "ui-monospace", "monospace"),
    radius_size=gr.themes.sizes.radius_md,
).set(
    body_background_fill=_PAPER,
    body_text_color=_INK,
    body_text_color_subdued=_SLATE,
    background_fill_primary=_PAPER,
    background_fill_secondary=_MIST,
    block_background_fill=_PAPER,
    block_border_color=_LINE,
    block_border_width="1px",
    block_radius="16px",
    block_label_background_fill=_PAPER,
    block_label_text_color=_SLATE,
    block_label_text_weight="600",
    block_shadow="none",
    input_background_fill=_MIST,
    input_border_color=_LINE,
    input_radius="12px",
    button_primary_background_fill=_CORNFLOWER,
    button_primary_background_fill_hover=_CORNFLOWER_DEEP,
    button_primary_text_color="#ffffff",
    button_secondary_background_fill="#eaf3fc",
    button_secondary_background_fill_hover=_POWDER,
    button_secondary_text_color=_CORNFLOWER,
    button_large_radius="999px",
    button_small_radius="999px",
    slider_color="#3a8ad6",
)
# Light-only: make every dark-mode token identical to its light twin, so an OS in dark mode
# can't turn the page black even before (or without) the _FORCE_LIGHT script running.
for _attr in list(vars(THEME)):
    if _attr.endswith("_dark") and hasattr(THEME, _attr[:-5]):
        setattr(THEME, _attr, getattr(THEME, _attr[:-5]))

CSS = """
:root {
  --display: 'Bricolage Grotesque', ui-sans-serif, system-ui, sans-serif;
  --paper: #ffffff; --mist: #f3f7fb; --line: #e6edf5; --powder: #dcebf9;
  --cornflower: #2f74bb; --ink: #1e2633; --slate: #6a7686;
}
body, gradio-app, .gradio-container, .main { background: var(--paper) !important; }
.gradio-container { max-width: 1120px !important; color: var(--ink); }
footer { display: none !important; }
button:focus-visible, [role="tab"]:focus-visible, input:focus-visible {
  outline: 2px solid var(--cornflower) !important; outline-offset: 2px; }

/* masthead: the app says what it's for, once ------------------------------------ */
#topbar { padding: 1.8rem 0 .4rem; }
#topbar .brand { font-family: var(--display); font-weight: 700; font-size: .92rem;
  letter-spacing: .01em; color: var(--cornflower); margin: 0 0 .4rem; }
#topbar h1 { font-family: var(--display); font-weight: 600; letter-spacing: -.03em;
  font-size: clamp(1.9rem, 4.4vw, 2.9rem); line-height: 1.04; color: var(--ink); margin: 0; }

/* tabs: quiet text with an underline --------------------------------------------- */
.tab-wrapper, .tab-container { border: 0 !important; }
.tab-container { border-bottom: 1px solid var(--line) !important; gap: 1.4rem !important; }
.tab-container button {
  font-family: var(--display) !important; font-weight: 600 !important; font-size: 1rem !important;
  color: var(--slate) !important; background: transparent !important; border: 0 !important;
  border-radius: 0 !important; padding: .7rem 0 !important; }
.tab-container button.selected { color: var(--ink) !important;
  box-shadow: inset 0 -2px 0 var(--cornflower) !important; }
.tab-container button::after { display: none !important; }

/* steps: open space, no boxes — only the mirror gets a frame ------------------------ */
.card { background: transparent !important; border: 0 !important; box-shadow: none !important;
  border-radius: 0 !important; padding: 1.2rem 0 1.4rem !important; gap: .95rem !important; }
.card + .card { border-top: 1px solid var(--line) !important; }
.card > *:not(button) { background: transparent !important; }
.card > .form, .card > .block:not(.looks):not(.image-container) { border: 0 !important; }
.step { display: flex; gap: .7rem; align-items: flex-start; }
.step .num { flex: none; width: 1.7rem; height: 1.7rem; border-radius: 999px; display: grid;
  place-items: center; font-family: var(--display); font-weight: 700; font-size: .85rem;
  background: var(--powder); color: var(--cornflower); }
.step h3 { font-family: var(--display); font-weight: 600; font-size: 1.2rem;
  letter-spacing: -.01em; margin: .05rem 0 .15rem; color: var(--ink); }
.step p { margin: 0; color: var(--slate); font-size: .93rem; }

/* looks as palette pans: round faces, name underneath ------------------------------ */
.looks { border: 0 !important; background: transparent !important; }
.looks, .looks .wrap, .looks .gallery-container, .looks .grid-wrap {
  height: auto !important; min-height: 0 !important; max-height: none !important;
  overflow: visible !important; background: transparent !important; }
.looks .grid-wrap { padding: .3rem .1rem 0 !important; }
.looks .grid-container { grid-template-columns: repeat(4, minmax(0, 1fr)) !important;
  gap: .9rem !important; height: auto !important; }
#library-gallery .grid-container {
  grid-template-columns: repeat(auto-fill, minmax(130px, 1fr)) !important; gap: 1.4rem !important; }
.looks .gallery-item { width: auto !important; height: auto !important; padding-bottom: 1.7rem; }
.looks .thumbnail-item { width: 100% !important; height: auto !important; aspect-ratio: 1;
  border-radius: 999px !important; border: 0 !important; box-shadow: none !important;
  background: transparent !important; overflow: visible !important; }
.looks .thumbnail-item img { width: 100% !important; height: 100% !important; object-fit: cover;
  border-radius: 999px; box-shadow: 0 0 0 1px var(--line);
  transition: box-shadow .18s ease, transform .18s ease; }
.looks .thumbnail-item:hover img { transform: translateY(-2px); }
.looks .thumbnail-item.selected img {
  box-shadow: 0 0 0 3px var(--paper), 0 0 0 5px var(--cornflower); }
.looks .caption-label { position: absolute !important; top: calc(100% + .45rem) !important;
  left: 0 !important; right: 0 !important; bottom: auto !important; max-width: none !important;
  background: transparent !important; color: var(--ink) !important; padding: 0 !important;
  font-family: var(--display); font-size: .82rem; font-weight: 600; text-align: center;
  white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.looks .thumbnail-item.selected .caption-label { color: var(--cornflower) !important; }
@media (max-width: 700px) {
  .looks .grid-container { grid-template-columns: repeat(3, minmax(0, 1fr)) !important; }
}

/* the look's colours, laid out like pans in a palette ------------------------------ */
.chips { display: flex; flex-wrap: wrap; align-items: flex-end; gap: 1rem; }
.chips-title { font-family: var(--display); font-weight: 600; color: var(--ink);
  margin-right: .2rem; align-self: center; }
.chip { display: inline-flex; flex-direction: column; align-items: center; gap: .3rem;
  font-size: .74rem; color: var(--slate); }
.chip-dot { width: 1.55rem; height: 1.55rem; border-radius: 999px; display: inline-block;
  box-shadow: inset 0 0 0 1px rgba(30, 38, 51, .12), 0 1px 3px rgba(30, 38, 51, .12); }

/* the mirror: the one framed thing on the page ------------------------------------- */
#result-card { background: var(--mist) !important; border-radius: 28px !important;
  padding: 1.4rem !important; margin-top: 1.2rem;
  box-shadow: 0 0 0 6px var(--paper), 0 0 0 7px var(--powder) !important; }
#result-card + * { margin-top: 0; }
@media (min-width: 900px) { #result-card { position: sticky; top: 1.2rem; } }
#result-card .block { background: transparent !important; border: 0 !important; }
#result-card .accordion { border-top: 1px solid var(--line) !important; border-radius: 0 !important; }
.empty { min-height: 440px; display: grid; place-content: center; text-align: center; padding: 2rem; }
.empty .ring { width: 124px; height: 124px; border-radius: 999px; margin: 0 auto 1.3rem;
  background: radial-gradient(circle at 34% 28%, #ffffff 0 18%, var(--powder) 70%);
  box-shadow: 0 0 0 6px var(--paper), 0 0 0 7px var(--powder); }
.empty p { font-family: var(--display); font-weight: 600; font-size: 1.15rem; color: var(--ink);
  margin: 0 0 .3rem; }
.empty span { color: var(--slate); font-size: .92rem; }
.status p { margin: 0; color: var(--slate); text-align: center; }
.status strong { color: var(--ink); }
.hint p { color: var(--slate); font-size: .88rem; margin: 0; }

/* buttons ------------------------------------------------------------------------ */
.go, .go button { font-family: var(--display) !important; font-weight: 600 !important; }
button.stop { background: transparent !important; color: #c04a66 !important;
  border: 1px solid #f1c9d3 !important; }
button.stop:hover { background: #fdf2f5 !important; }
.accordion .label-wrap span { font-family: var(--display); font-weight: 600; }

.foot p { color: var(--slate); font-size: .82rem; text-align: center; }
.foot a { color: var(--cornflower) !important; }
@media (prefers-reduced-motion: reduce) { * { transition: none !important; } }
"""

_FONT_LINK = (
    '<link rel="preconnect" href="https://fonts.googleapis.com">'
    '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
    '<link href="https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,500..700'
    '&display=swap" rel="stylesheet">'
)

# Belt and braces with the token mirroring above: drop Gradio's dark class if it appears.
_FORCE_LIGHT = """() => {
  const light = () => [document.documentElement, document.body]
    .forEach(el => el.classList.remove('dark'));
  light();
  new MutationObserver(light).observe(document.body, {attributes: true, attributeFilter: ['class']});
}"""

# Gradio 6 takes theme/css/head/js at launch() — the Space entrypoint passes these too.
LAUNCH_KWARGS = {"theme": THEME, "css": CSS, "head": _FONT_LINK, "js": _FORCE_LIGHT}

_EMPTY_RESULT = (
    '<div class="empty"><div class="ring"></div><p>Your result appears here</p>'
    '<span>Choose a look and add a photo. It applies automatically.</span></div>'
)


def _step(number: str, title: str, sub: str) -> str:
    return (f'<div class="step"><span class="num">{number}</span>'
            f'<div><h3>{title}</h3><p>{sub}</p></div></div>')


# --- UI --------------------------------------------------------------------------------------
def build_demo() -> gr.Blocks:
    names = _look_names()
    first = names[0] if names else None
    l0, e0, b0, s0, c0 = _look_strengths(first)
    engine_ok, engine_why = _engine_available()
    diffusion_ok, diffusion_why = _diffusion_available()

    with gr.Blocks(title="Makeup Transfer") as demo:
        gr.HTML('<div id="topbar"><p class="brand">Makeup Transfer</p>'
                '<h1>Try a look before you wear it.</h1></div>')

        look = gr.State(first)          # the look being worn on the Try-on screen
        lib_pick = gr.State(None)       # the look selected in My looks

        with gr.Tabs() as tabs:
            # --- Try on --------------------------------------------------------------------
            with gr.Tab("Try on", id="try"):
                with gr.Row(equal_height=False):
                    with gr.Column(scale=5):
                        with gr.Column(elem_classes="card"):
                            gr.HTML(_step("1", "Choose a look", "Tap a face to borrow its makeup."))
                            a_gallery = gr.Gallery(
                                value=_gallery(first)["value"], selected_index=0 if first else None,
                                show_label=False, columns=4, height="auto", object_fit="cover",
                                allow_preview=False, buttons=[], elem_classes="looks",
                            )
                            a_swatch = gr.HTML(_swatch_html(first))

                        with gr.Column(elem_classes="card"):
                            gr.HTML(_step("2", "Add your photo",
                                          "Front-facing, good light, nothing covering the face."))
                            a_target = gr.Image(type="filepath", show_label=False, height=260,
                                                sources=["upload", "webcam", "clipboard"])
                            target_rows = _sample_rows([["002386_bare.jpg"], ["003125_bare.jpg"],
                                                        ["003417_bare.jpg"]])
                            if target_rows:
                                gr.Examples(target_rows, inputs=a_target,
                                            label="No photo handy? Use a sample face",
                                            examples_per_page=3)

                    with gr.Column(scale=6, elem_id="result-card", elem_classes="card"):
                        gr.HTML(_step("3", "See the result", "Drag the handle to compare before and after."))
                        a_empty = gr.HTML(_EMPTY_RESULT)
                        a_side = gr.ImageSlider(show_label=False, visible=False, type="numpy",
                                                slider_position=50, height=520)
                        a_status = gr.Markdown(elem_classes="status")
                        a_quality = gr.Radio(
                            [_INSTANT, _REALISTIC] if diffusion_ok else [_INSTANT],
                            value=_INSTANT, label="Quality",
                            info=("Instant previews in a second. Realistic repaints the face "
                                  "with a diffusion model for natural-looking makeup."
                                  if diffusion_ok else
                                  "Realistic (diffusion) mode isn't installed here — run the "
                                  "app locally with `fetch_weights --diffusion` to get it."))
                        a_file = gr.DownloadButton("Download photo", variant="primary", size="lg",
                                                   visible=False, elem_classes="go")
                        with gr.Accordion("Adjust intensity", open=False):
                            with gr.Row():
                                a_lips = gr.Slider(0.0, 1.5, value=l0, step=0.05, label="Lips")
                                a_eyes = gr.Slider(0.0, 1.5, value=e0, step=0.05, label="Eyes")
                            with gr.Row():
                                a_brows = gr.Slider(0.0, 1.5, value=b0, step=0.05, label="Brows")
                                a_skin = gr.Slider(0.0, 1.5, value=s0, step=0.05, label="Skin")
                            a_complexion = gr.Slider(
                                0.0, 1.0, value=c0, step=0.05, label="Borrow their complexion",
                                info="0 keeps your own skin tone; 1 takes the look's "
                                     "foundation and lighting too.")
                            a_match = gr.Slider(
                                0.0, 1.0, value=DEFAULT_MATCH_SKIN, step=0.05,
                                label="Match neck and body",
                                info="Carries the face's new skin tone onto your ears, neck "
                                     "and other bare skin so the face doesn't stand apart. "
                                     "0 touches the face only.")
                            a_guidance = gr.Slider(
                                1.0, 2.5, value=DEFAULT_GUIDANCE, step=0.05,
                                label="Makeup weight (Realistic only)", visible=diffusion_ok,
                                info="About 1.1 for a light look, 2 for a heavy one. "
                                     "Changing it repaints the face.")
                            a_strength = gr.Slider(
                                0.4, 1.0, value=DEFAULT_STRENGTH, step=0.05,
                                label="Creative freedom (Realistic only)", visible=diffusion_ok,
                                info="Lower stays closer to your own face; higher lets the "
                                     "model repaint more. Changing it repaints the face.")
                            a_reset = gr.Button("Reset to the original look", size="sm")

            # --- Create a look -------------------------------------------------------------
            with gr.Tab("Create a look", id="create"):
                with gr.Column(elem_classes="card"):
                    gr.HTML(_step("✦", "Turn a photo into a look",
                                  "One photo of anyone wearing makeup you like — a friend, a "
                                  "magazine page, a screenshot. It becomes a look you can try on."))
                    if not engine_ok:
                        gr.Markdown(f"**Not available here:** {engine_why}", elem_classes="hint")
                    with gr.Row():
                        p_with = gr.Image(label="1 · Photo with makeup", type="filepath",
                                          height=300)
                        p_preview = gr.Image(label="The face we'll borrow the look from",
                                             height=300, visible=False)
                    photo_rows = _sample_rows([
                        ["002386_makeup_03.jpg", "rose-satin"],
                        ["003125_makeup_02.jpg", "warm-nude"],
                        ["003417_makeup_05.jpg", "deep-plum"],
                    ])
                    p_name = gr.Textbox(label="2 · Name your look", placeholder="e.g. Sunday brunch")
                    if photo_rows:
                        gr.Examples(photo_rows, inputs=[p_with, p_name],
                                    label="Or try it with a sample photo", examples_per_page=3)
                    p_btn = gr.Button("Create look", variant="primary", size="lg",
                                      interactive=engine_ok, elem_classes="go")
                    p_status = gr.Markdown(elem_classes="status")
                    p_go = gr.Button("Try it on →", variant="primary", size="lg", visible=False,
                                     elem_classes="go")

            # --- My looks ------------------------------------------------------------------
            with gr.Tab("My looks", id="library"):
                with gr.Column(elem_classes="card"):
                    gr.HTML(_step("♡", "Your looks", "Tap a look to try it on, download it, "
                                                     "or remove it."))
                    l_gallery = gr.Gallery(value=_gallery()["value"], show_label=False, columns=5,
                                           height="auto", object_fit="cover", allow_preview=False,
                                           buttons=[], elem_classes="looks",
                                           elem_id="library-gallery")
                    with gr.Column(visible=False) as l_actions:
                        l_title = gr.Markdown()
                        with gr.Row():
                            l_try = gr.Button("Try it on", variant="primary", elem_classes="go")
                            l_exp = gr.Button("Download (.zip)")
                            l_del = gr.Button("Delete", variant="stop")
                        l_file = gr.File(show_label=False, visible=False)
                        with gr.Row(visible=False) as l_confirm:
                            l_confirm_msg = gr.Markdown()
                            l_cancel = gr.Button("Keep it", size="sm")
                            l_yes = gr.Button("Delete", variant="stop", size="sm")

        gr.Markdown(
            "Sample faces from [FFHQ-Makeup](https://huggingface.co/datasets/cyberagent/FFHQ-Makeup) "
            "(CC BY-NC-SA) · Photos are processed only to make your result · "
            "[Source code](https://github.com/Edward9292/ai-makeup-transfer)",
            elem_classes="foot",
        )

        # --- wiring (all components exist now) ----------------------------------------------
        sliders = [a_lips, a_eyes, a_brows, a_skin, a_complexion]
        panel = [*sliders, a_swatch]
        try_in = [look, a_target, *sliders, a_quality, a_guidance, a_strength, a_match]
        try_out = [a_empty, a_side, a_file, a_status]

        a_gallery.select(do_select_look, None, [look, *panel]).then(do_try, try_in, try_out)
        a_target.change(do_try, try_in, try_out)
        for s in (*sliders, a_guidance, a_strength, a_match):
            s.release(do_try, try_in, try_out)
        a_quality.change(do_try, try_in, try_out)
        a_reset.click(_look_panel, look, panel).then(
            lambda: DEFAULT_MATCH_SKIN, None, a_match).then(do_try, try_in, try_out)

        created = [look, a_gallery, l_gallery]
        p_btn.click(do_create, [p_with, p_name], [p_preview, p_status, p_go, *created])

        go_out = [tabs, look, a_gallery, *panel]
        p_go.click(do_go_try, look, go_out).then(do_try, try_in, try_out)
        l_try.click(do_go_try, lib_pick, go_out).then(do_try, try_in, try_out)

        l_gallery.select(do_select_library, None,
                         [lib_pick, l_actions, l_title, l_file, l_confirm])
        l_exp.click(do_export, lib_pick, l_file)
        l_del.click(do_ask_delete, lib_pick, l_confirm_msg).then(
            lambda: gr.update(visible=True), None, l_confirm)
        l_cancel.click(lambda: gr.update(visible=False), None, l_confirm)
        l_yes.click(do_delete, [lib_pick, look],
                    [lib_pick, l_actions, l_confirm, l_file, look, a_gallery, l_gallery])

        demo.load(do_reload, look, [a_gallery, l_gallery])

    return demo


def main() -> None:
    build_demo().launch(**LAUNCH_KWARGS)


if __name__ == "__main__":
    main()
