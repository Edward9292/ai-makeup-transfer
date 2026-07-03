"""Phase 3: Gradio web demo (``app.py``).

A thin browser UI over the already-built pipeline — it only calls existing public functions
(``extract_look`` / ``extract_look_single`` / ``apply_look`` / ``LookStore``), reimplementing no
pipeline logic. Four tabs:

1. **Extract from a pair** — with-makeup + bare (same person) -> saved RGBA look (CPU).
2. **Extract from one photo** — with-makeup only; SSAT removal synthesizes the bare face
   (Phase 2.5). **Requires CUDA.**
3. **Apply a look** — pick a saved look + a target face, per-region intensity sliders -> result.
4. **Look library** — browse / delete / zip-export saved looks.

Uploads are handled as file *paths* (``gr.Image(type="filepath")``) to match the pipeline's
path-based API. Pipeline outputs are OpenCV BGR arrays, converted to RGB for display here.

Run (from the repo root)::

    $env:PYTHONPATH = "src"          # package lives under src/
    python -m makeup_transfer.app
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import gradio as gr

from .apply import DEFAULT_REGION_PARAMS, apply_look
from .extract import _checker_preview, extract_look, extract_look_single
from .storage import LookStore

_ROOT = Path(__file__).resolve().parents[2]
WEBAPP_OUT = _ROOT / "preprocess_out" / "webapp"
WEBAPP_OUT.mkdir(parents=True, exist_ok=True)

store = LookStore()

# Slider order used everywhere strengths are read/written: lips, eyes, brows, skin.
_REGIONS = ["lips", "eyes", "brows", "skin"]


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
    return (
        _preview_rgb(layer),
        f"Saved look '{name}'. It's now available in the Apply and Library tabs.",
        gr.update(choices=looks, value=name),
        gr.update(choices=looks, value=name),
        _gallery_items(),
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
            "On a CPU-only machine, use the 'Extract from a pair' tab instead."
        )
    except ValueError as e:
        raise gr.Error(f"Could not process the photo: {e}. Use a frontal, well-lit face.")
    pseudo = cv2.imread(str(pseudo_path))
    looks = _looks()
    return (
        _bgr2rgb(pseudo) if pseudo is not None else None,
        _preview_rgb(layer),
        f"Saved look '{name}' (single-photo). Available in the Apply and Library tabs.",
        gr.update(choices=looks, value=name),
        gr.update(choices=looks, value=name),
        _gallery_items(),
    )


def do_apply(look, target_path, lips, eyes, brows, skin, composite):
    if not look:
        raise gr.Error("Pick a saved look.")
    if not target_path:
        raise gr.Error("Upload a target face photo.")
    strengths = {"lips": float(lips), "eyes": float(eyes),
                 "brows": float(brows), "skin": float(skin)}
    try:
        result_bgr, debug = apply_look(
            look, target_path, strengths=strengths, composite=composite, store=store,
        )
    except ValueError as e:
        raise gr.Error(f"Could not apply the look: {e}")
    result = _bgr2rgb(result_bgr)
    side = np.hstack([_bgr2rgb(debug["target"]), result])
    out_path = WEBAPP_OUT / f"{look}_on_target.png"
    cv2.imwrite(str(out_path), result_bgr)
    return side, str(out_path)


def do_delete(look):
    if not look:
        raise gr.Error("Pick a look to delete.")
    store.delete(look)
    looks = _looks()
    return (
        gr.update(choices=looks, value=None),
        gr.update(choices=looks, value=None),
        _gallery_items(),
        f"Deleted '{look}'.",
    )


def do_export(look):
    if not look:
        raise gr.Error("Pick a look to export.")
    out = WEBAPP_OUT / f"{look}.zip"
    store.export_zip(look, out)
    return str(out)


def do_refresh():
    looks = _looks()
    return gr.update(choices=looks), gr.update(choices=looks), _gallery_items()


# --- UI --------------------------------------------------------------------------------------
def build_demo() -> gr.Blocks:
    initial = _looks()
    l0, e0, b0, s0 = _look_strengths(initial[0] if initial else None)

    with gr.Blocks(title="AI Makeup Transfer") as demo:
        gr.Markdown(
            "# 💄 AI Makeup Transfer\n"
            "Extract makeup from a face as a translucent **RGBA layer in UV space**, then "
            "re-apply it to any face so skin tone shows through and the look adapts per person.\n\n"
            "*All models are pretrained — nothing is trained here. The single-photo tab needs a "
            "CUDA GPU; the other tabs run on CPU.*"
        )

        with gr.Tabs():
            # --- Tab 1: extract from a pair -------------------------------------------------
            with gr.Tab("Extract from a pair"):
                gr.Markdown(
                    "Two photos of the **same person** — one with makeup, one bare. The difference "
                    "in UV space becomes a reusable makeup layer."
                )
                with gr.Row():
                    p_with = gr.Image(label="With makeup", type="filepath")
                    p_bare = gr.Image(label="Bare (same person)", type="filepath")
                with gr.Row():
                    p_name = gr.Textbox(label="Look name", placeholder="e.g. smoky-red")
                    p_sym = gr.Slider(0.0, 1.0, value=0.5, step=0.05,
                                      label="Shadow suppression (UV symmetry)")
                p_btn = gr.Button("Extract look", variant="primary")
                p_status = gr.Markdown()
                p_preview = gr.Image(label="Extracted makeup layer (on checkerboard)")

            # --- Tab 2: extract from one photo ----------------------------------------------
            with gr.Tab("Extract from one photo"):
                gr.Markdown(
                    "One photo only. SSAT makeup **removal** synthesizes the bare face, then the "
                    "same pair pipeline runs. **Requires a CUDA GPU.**"
                )
                with gr.Row():
                    s_with = gr.Image(label="With makeup", type="filepath")
                    s_ref = gr.Image(label="Bare reference (optional)", type="filepath")
                with gr.Row():
                    s_name = gr.Textbox(label="Look name", placeholder="e.g. stolen-look")
                    s_sym = gr.Slider(0.0, 1.0, value=0.5, step=0.05,
                                      label="Shadow suppression (UV symmetry)")
                s_btn = gr.Button("Extract look (GPU)", variant="primary")
                s_status = gr.Markdown()
                with gr.Row():
                    s_pseudo = gr.Image(label="Synthesized bare face")
                    s_preview = gr.Image(label="Extracted makeup layer (on checkerboard)")

            # --- Tab 3: apply a look --------------------------------------------------------
            with gr.Tab("Apply a look"):
                gr.Markdown("Apply a saved look to a target face. Sliders scale each region.")
                with gr.Row():
                    with gr.Column():
                        a_look = gr.Dropdown(choices=initial, label="Saved look",
                                             value=initial[0] if initial else None)
                        a_target = gr.Image(label="Target face", type="filepath")
                        a_lips = gr.Slider(0.0, 1.5, value=l0, step=0.05, label="Lips")
                        a_eyes = gr.Slider(0.0, 1.5, value=e0, step=0.05, label="Eyes")
                        a_brows = gr.Slider(0.0, 1.5, value=b0, step=0.05, label="Brows")
                        a_skin = gr.Slider(0.0, 1.5, value=s0, step=0.05, label="Skin")
                        a_comp = gr.Radio(["delta", "plain"], value="delta",
                                          label="Composite (delta = grain-preserving)")
                        a_btn = gr.Button("Apply", variant="primary")
                    with gr.Column():
                        a_side = gr.Image(label="Target  |  Result")
                        a_file = gr.File(label="Download result")

            # --- Tab 4: look library --------------------------------------------------------
            with gr.Tab("Look library"):
                gr.Markdown("Every saved look. Delete or export (zip) below.")
                l_gallery = gr.Gallery(value=_gallery_items(), label="Saved looks",
                                       columns=4, height="auto")
                with gr.Row():
                    l_pick = gr.Dropdown(choices=initial, label="Select a look",
                                         value=initial[0] if initial else None)
                    l_del = gr.Button("Delete")
                    l_exp = gr.Button("Export .zip")
                l_status = gr.Markdown()
                l_file = gr.File(label="Download zip")
                l_refresh = gr.Button("Refresh")

        # --- wiring (all components exist now) ----------------------------------------------
        p_btn.click(do_extract_pair, [p_with, p_bare, p_name, p_sym],
                    [p_preview, p_status, a_look, l_pick, l_gallery])
        s_btn.click(do_extract_single, [s_with, s_ref, s_name, s_sym],
                    [s_pseudo, s_preview, s_status, a_look, l_pick, l_gallery])

        # When a look is selected in the Apply tab, seed the sliders from its recipe.
        a_look.change(_look_strengths, a_look, [a_lips, a_eyes, a_brows, a_skin])
        a_btn.click(do_apply, [a_look, a_target, a_lips, a_eyes, a_brows, a_skin, a_comp],
                    [a_side, a_file])

        l_del.click(do_delete, l_pick, [a_look, l_pick, l_gallery, l_status])
        l_exp.click(do_export, l_pick, l_file)
        l_refresh.click(do_refresh, None, [a_look, l_pick, l_gallery])

    return demo


def main() -> None:
    build_demo().launch(theme=gr.themes.Soft())


if __name__ == "__main__":
    main()
