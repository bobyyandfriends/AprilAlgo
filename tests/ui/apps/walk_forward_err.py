"""Streamlit runner: walk-forward lab CLI error path."""

from unittest.mock import MagicMock

import aprilalgo.ui.pages.walk_forward_lab as wfl
from aprilalgo.ui.pages.walk_forward_lab import render


def _fake_err(*args, **kwargs):
    return MagicMock(returncode=1, stdout="", stderr="config not found")


wfl.subprocess.run = _fake_err  # type: ignore[method-assign]

render()
