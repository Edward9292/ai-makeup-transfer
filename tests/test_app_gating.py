"""The Create-a-look button must be visible but disabled when the engine weights are absent."""
import gradio as gr

from makeup_transfer import app


def _create_buttons(demo):
    return [b for b in demo.blocks.values()
            if isinstance(b, gr.Button) and b.value == "Create look"]


def test_engine_available_returns_tuple():
    ok, why = app._engine_available()
    assert isinstance(ok, bool)
    assert isinstance(why, str)
    if not ok:
        assert why  # a disabled panel must say why


def test_create_button_disabled_when_unavailable(monkeypatch):
    monkeypatch.setattr(app, "_engine_available", lambda: (False, "weights missing"))
    demo = app.build_demo()
    btns = _create_buttons(demo)
    assert len(btns) == 1
    assert btns[0].interactive is False


def test_create_button_enabled_when_available(monkeypatch):
    monkeypatch.setattr(app, "_engine_available", lambda: (True, ""))
    demo = app.build_demo()
    btns = _create_buttons(demo)
    assert len(btns) == 1
    assert btns[0].interactive is not False  # None or True both mean enabled
