from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from gui import AnnotatorApp


def make_app(platform="win32", legacy=True):
    app = AnnotatorApp.__new__(AnnotatorApp)
    app.root = Mock()
    app.root.tk.call.side_effect = lambda *args: {
        ("tk", "windowingsystem"): platform,
        ("package", "provide", "Tk"): "8.6.15" if legacy else "9.0",
        ("package", "vcompare", "8.6.15", "8.7"): -1,
        ("package", "vcompare", "9.0", "8.7"): 1,
    }[args]
    app.canvas = Mock()
    app.filtered_image_paths = ["first", "second", "third"]
    app.current_index = 1
    app.load_image = Mock()
    app._focus_is_text_input = Mock(return_value=False)
    app._annotation_edit_in_progress = Mock(return_value=False)
    app._bind_mouse_navigation()
    return app


@pytest.mark.parametrize("platform,legacy,buttons", [
    ("win32", True, {4, 5, 8, 9}),
    ("win32", False, {8, 9}),
    ("x11", True, {8, 9}),
    ("aqua", False, {8, 9}),
])
def test_platform_mapping_preserves_wheel_bindings(platform, legacy, buttons):
    app = make_app(platform, legacy)
    assert set(app._mouse_nav_buttons) == buttons
    sequences = [call.args[0] for call in app.canvas.bind.call_args_list]
    assert sequences == (["<ButtonPress>", "<ButtonPress-4>", "<ButtonPress-5>"]
                         if platform == "win32" and legacy else ["<ButtonPress>"])


@pytest.mark.parametrize("button,index", [(4, 0), (5, 2), (8, 0), (9, 2), ("9", 2)])
def test_side_buttons_use_existing_navigation(button, index):
    app = make_app()
    assert app._on_mouse_navigation(SimpleNamespace(num=button)) == "break"
    app.load_image.assert_called_once_with(index)


@pytest.mark.parametrize("button", [1, 2, 3, 6, 7, 10, None, "??"])
def test_other_buttons_do_not_navigate(button):
    app = make_app()
    assert app._on_mouse_navigation(SimpleNamespace(num=button)) is None
    app.load_image.assert_not_called()


@pytest.mark.parametrize("guard", ["_focus_is_text_input", "_annotation_edit_in_progress"])
def test_side_buttons_respect_shortcut_guards(guard):
    app = make_app()
    getattr(app, guard).return_value = True
    assert app._on_mouse_navigation(SimpleNamespace(num=5)) == "break"
    app.load_image.assert_not_called()


@pytest.mark.parametrize("button,start,expected", [(4, 0, 2), (5, 2, 0)])
def test_side_buttons_preserve_wraparound(button, start, expected):
    app = make_app()
    app.current_index = start
    app._on_mouse_navigation(SimpleNamespace(num=button))
    app.load_image.assert_called_once_with(expected)


def test_side_buttons_with_no_images():
    app = make_app()
    app.filtered_image_paths = []
    app._on_mouse_navigation(SimpleNamespace(num=5))
    app.load_image.assert_not_called()


def test_d_key_still_navigates_and_releases():
    app = make_app()
    app.nav_held_key = None
    app.nav_timer_id = None
    app.rapid_mode = False
    event = SimpleNamespace(keysym="d")
    app._on_nav_key_press(event)
    app.load_image.assert_called_once_with(2)
    app._on_nav_key_release(event)
    assert app.nav_held_key is None
