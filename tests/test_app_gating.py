"""The single-photo (GPU) panel must be visible but disabled when SSAT/CUDA is absent."""
import gradio as gr

from makeup_transfer import app


def _gpu_buttons(demo):
    return [b for b in demo.blocks.values()
            if isinstance(b, gr.Button) and b.value == "Extract look on GPU"]


def test_single_photo_available_returns_tuple():
    ok, why = app._single_photo_available()
    assert isinstance(ok, bool)
    assert isinstance(why, str)
    if not ok:
        assert why  # a disabled panel must say why


def test_gpu_button_disabled_when_unavailable(monkeypatch):
    monkeypatch.setattr(app, "_single_photo_available",
                        lambda: (False, "needs a CUDA GPU"))
    demo = app.build_demo()
    btns = _gpu_buttons(demo)
    assert len(btns) == 1
    assert btns[0].interactive is False


def test_gpu_button_enabled_when_available(monkeypatch):
    monkeypatch.setattr(app, "_single_photo_available", lambda: (True, ""))
    demo = app.build_demo()
    btns = _gpu_buttons(demo)
    assert len(btns) == 1
    assert btns[0].interactive is not False  # None or True both mean enabled
